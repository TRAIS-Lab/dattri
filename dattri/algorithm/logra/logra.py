"""LoGra Attributor - Influence Function with gradient projection."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from dattri.task import AttributionTask

import logging

import torch
from torch import nn
from tqdm import tqdm

from dattri.algorithm.base import BaseAttributor

# Import LoGra utilities
from dattri.algorithm.logra.core.hook import HookManager
from dattri.algorithm.logra.core.metadata import MetadataManager
from dattri.algorithm.logra.offload import create_offload_manager
from dattri.algorithm.logra.utils.common import stable_inverse
from dattri.algorithm.logra.utils.projector import setup_model_projectors
from dattri.algorithm.utils import _check_shuffle

logger = logging.getLogger(__name__)


class LoGraAttributor(BaseAttributor):
    """LoGra Attributor.

    Computes influence scores using projected gradients for efficiency.
    Uses hooks to capture per-sample gradients and applies random projections.
    """

    def __init__(
        self,
        task: "AttributionTask",
        layer_names: Optional[
            Union[str, List[str]]
        ] = None,  # Maybe support layer class as input?
        hessian: Literal["Identity", "eFIM"] = "eFIM",
        damping: Optional[float] = None,
        device: str = "cpu",
        projector_kwargs: Optional[Dict[str, Any]] = None,
        offload: Literal["none", "cpu", "disk"] = "cpu",
        cache_dir: Optional[str] = None,
        chunk_size: int = 16,
    ) -> None:
        """Initialize LoGra attributor.

        Args:
            task: Attribution task containing model, loss function, and checkpoints
            layer_names: Names of layers where gradients will be collected.
                If None, uses all Linear layers.
                You can check the names using model.named_modules().
                HookManager will register hooks to named layers.
            hessian: Type of Hessian approximation ("Identity", "eFIM"). For
                "Identity", the hessian will be taken as the identity matrix.
                For "eFIM", the hessian will be computed as the empirical
                fisher information matrix.
            damping: Damping factor for Hessian inverse (when hessian="eFIM")
            device: Device to run computations on
            projector_kwargs: Arguments for random projection (proj_dim, method, etc.)
            offload: Memory management strategy ("none", "cpu", "disk"), stating
                the place to offload the gradients.
                "cpu": stores gradients on CPU and moves to device when needed.
                "disk": stores gradients on disk and moves to device when needed.
            cache_dir: Directory for caching (required when offload="disk").
            chunk_size: Chunk size for processing in disk offload.

        Raises:
            ValueError: If cache_dir is None when offload="disk".
        """
        self.task = task
        self.device = device
        self.hessian = hessian
        self.damping = damping
        self.projector_kwargs = projector_kwargs or {}
        self.offload = offload
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size

        # Get model from task and setup
        task._load_checkpoints(0)
        self.model = task.get_model()
        self.model.to(device)
        self.model.eval()

        # Determine layer names
        if layer_names is None:
            # Default to all Linear layers
            self.layer_names = [
                name
                for name, module in self.model.named_modules()
                if isinstance(module, nn.Linear)
            ]
        elif isinstance(layer_names, str):
            self.layer_names = [layer_names]
        else:
            self.layer_names = layer_names

        logger.info(
            "LoGra initialized with %d layers: %s",
            len(self.layer_names),
            self.layer_names,
        )

        # Create offload manager
        self.offload_manager = create_offload_manager(
            offload_type=offload,
            device=device,
            layer_names=self.layer_names,
            cache_dir=cache_dir,
            chunk_size=chunk_size,
        )

        # Initialize metadata manager
        self.metadata = MetadataManager(cache_dir or ".", self.layer_names)

        # Initialize other components
        self.full_train_dataloader: Optional["DataLoader"] = None
        self.hook_manager: Optional[HookManager] = None
        self.projectors: Optional[List[Any]] = None
        self.layer_dims: Optional[List[int]] = None
        self.total_proj_dim: Optional[int] = None

        # Validation
        if offload == "disk" and cache_dir is None:
            msg = "cache_dir must be provided when offload='disk'"
            raise ValueError(msg)

    def _setup_projectors(self, train_dataloader: "DataLoader") -> None:
        """Set up projectors for the model layers.

        Args:
            train_dataloader: DataLoader for training data used to set up projectors.
        """
        if not self.projector_kwargs:
            self.projectors = []
            return

        logger.info("Setting up projectors...")
        self.projectors = setup_model_projectors(
            self.model,
            self.layer_names,
            self.projector_kwargs,
            train_dataloader,
            device=self.device,
        )

    def _sync_layer_dims(self) -> None:
        """Synchronize layer dimensions between components."""
        if self.layer_dims is None:
            if self.metadata.layer_dims is not None:
                self.layer_dims = self.metadata.layer_dims
                self.total_proj_dim = self.metadata.total_proj_dim
            elif (
                hasattr(self.offload_manager, "layer_dims")
                and self.offload_manager.layer_dims is not None
            ):
                self.layer_dims = self.offload_manager.layer_dims
                self.total_proj_dim = self.offload_manager.total_proj_dim

        # Sync to all components
        if self.layer_dims is not None:
            if hasattr(self.offload_manager, "layer_dims"):
                self.offload_manager.layer_dims = self.layer_dims
                self.offload_manager.total_proj_dim = self.total_proj_dim
            if self.metadata.layer_dims is None:
                self.metadata.set_layer_dims(self.layer_dims)

    def cache(self, full_train_dataloader: "DataLoader") -> None:
        """Cache gradients and IFVP for the full training dataset.

        Args:
            full_train_dataloader: DataLoader for full training data
        """
        logger.info("Caching gradients with LoGra (offload: %s)", self.offload)

        self.full_train_dataloader = full_train_dataloader

        # Setup projectors if not already done
        if self.projectors is None:
            self._setup_projectors(full_train_dataloader)

        # Initialize metadata for complete dataset
        self.metadata.initialize_complete_dataset(
            full_train_dataloader,
            is_master_worker=True,
        )

        # Create hook manager
        if self.hook_manager is None:
            self.hook_manager = HookManager(
                self.model,
                self.layer_names,
            )
            if self.projectors:
                self.hook_manager.set_projectors(self.projectors)

        # Collect gradients using hooks
        logger.info("Computing gradients using hooks...")
        for batch_idx, batch in enumerate(
            tqdm(full_train_dataloader, desc="Computing gradients"),
        ):
            self.model.zero_grad()

            # Prepare inputs and compute loss
            loss = self.task.original_loss_func(self.model, batch, self.device)
            batch_size = full_train_dataloader.batch_size

            # Update metadata
            self.metadata.add_batch_info(batch_idx, batch_size)

            # Backward pass
            loss.backward()

            # Get projected gradients from hooks
            with torch.no_grad():
                compressed_grads = self.hook_manager.get_compressed_grads()

                # Detect layer dimensions on first batch
                if self.layer_dims is None:
                    self.layer_dims = []
                    for grad in compressed_grads:
                        if grad is not None and grad.numel() > 0:
                            self.layer_dims.append(
                                grad.shape[1] if grad.dim() > 1 else grad.numel(),
                            )
                        else:
                            self.layer_dims.append(0)

                    self.total_proj_dim = sum(self.layer_dims)
                    logger.info(
                        "Detected layer dimensions: %d layers, total dimension=%d",
                        len(self.layer_dims),
                        self.total_proj_dim,
                    )

                    # Save to metadata manager
                    self.metadata.set_layer_dims(self.layer_dims)

                    # Pass to offload manager if needed
                    if hasattr(self.offload_manager, "layer_dims"):
                        self.offload_manager.layer_dims = self.layer_dims
                        self.offload_manager.total_proj_dim = self.total_proj_dim

                # Store gradients using offload manager
                self.offload_manager.store_gradients(
                    batch_idx,
                    compressed_grads,
                    is_test=False,
                )

            # Memory management
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        # Save metadata
        self.metadata.save_metadata()

        # Clean up hooks
        self.hook_manager.remove_hooks()
        self.hook_manager = None

        # Compute preconditioners and IFVP if needed
        if self.hessian == "eFIM":
            logger.info("Computing preconditioners...")
            self.compute_preconditioners()
            logger.info("Computing IFVP...")
            self.compute_ifvp()

        logger.info("Caching completed")

    def compute_preconditioners(self, damping: Optional[float] = None) -> None:
        """Compute preconditioners (inverse Hessian) from gradients.

        Args:
            damping: Damping factor for numerical stability

        Raises:
            ValueError: If layer dimensions are not found.
        """
        logger.info("Computing preconditioners...")

        if damping is None:
            damping = self.damping

        # Load batch information if needed
        if not self.metadata.batch_info:
            self.metadata._load_metadata_if_exists()

        # Synchronize layer dimensions
        self._sync_layer_dims()

        if self.layer_dims is None:
            msg = (
                "Layer dimensions not found. "
                "Ensure gradients have been computed and stored."
            )
            raise ValueError(
                msg,
            )

        # If hessian type is "Identity", no preconditioners needed
        if self.hessian == "Identity":
            logger.info("Hessian type is 'none', skipping preconditioner computation")
            for layer_idx in range(len(self.layer_names)):
                self.offload_manager.store_preconditioner(layer_idx, None)
            return

        total_samples = self.metadata.get_total_samples()
        logger.info("Computing preconditioners from %s total samples", total_samples)

        # Initialize Hessian accumulators
        hessian_accumulators = []
        sample_counts = [0] * len(self.layer_names)

        for layer_idx in range(len(self.layer_names)):
            layer_dim = self.layer_dims[layer_idx]
            if layer_dim > 0:
                hessian_accumulators.append(
                    torch.zeros(
                        layer_dim,
                        layer_dim,
                        device=self.device,
                        dtype=torch.float32,
                    ),
                )
            else:
                hessian_accumulators.append(None)

        # Use tensor-based dataloader for efficient processing
        dataloader = self.offload_manager.create_gradient_dataloader(
            data_type="gradients",
            batch_size=4,
            pin_memory=True,
        )

        for chunk_tensor, _batch_mapping in tqdm(
            dataloader,
            desc="Computing preconditioners",
        ):
            # Move chunk to device
            chunk_tensor_device = self.offload_manager.move_to_device(chunk_tensor)

            # Process each layer
            for layer_idx in range(len(self.layer_names)):
                # Extract layer slice
                start_col = sum(self.layer_dims[:layer_idx])
                end_col = start_col + self.layer_dims[layer_idx]
                layer_data = chunk_tensor_device[:, start_col:end_col].detach()

                if layer_data.numel() == 0 or hessian_accumulators[layer_idx] is None:
                    continue

                sample_counts[layer_idx] += layer_data.shape[0]

                # In-place accumulation: H += g^T @ g
                hessian_accumulators[layer_idx].addmm_(layer_data.t(), layer_data)

            torch.cuda.empty_cache()

        # Compute preconditioners from accumulated Hessians
        computed_count = 0
        for layer_idx in tqdm(range(len(self.layer_names)), desc="Computing inverses"):
            hessian_accumulator = hessian_accumulators[layer_idx]
            sample_count = sample_counts[layer_idx]

            if hessian_accumulator is not None and sample_count > 0:
                # Normalize by total number of samples
                hessian = hessian_accumulator / sample_count

                # Compute inverse
                if self.hessian == "eFIM":
                    precond = stable_inverse(hessian, damping=damping)
                    self.offload_manager.store_preconditioner(layer_idx, precond)

                computed_count += 1
                torch.cuda.empty_cache()

        self.offload_manager.wait_for_async_operations()
        logger.info("Computed %s preconditioners", computed_count)

    def compute_ifvp(self) -> None:
        """Compute inverse-Hessian-vector products (IFVP).

        Here we use empirical fisher information matrix.

        Raises:
            ValueError: If layer dimensions are not found.
        """
        logger.info("Computing IFVP...")

        # Load batch information if needed
        if not self.metadata.batch_info:
            self.metadata._load_metadata_if_exists()

        # Synchronize layer dimensions
        self._sync_layer_dims()

        if self.layer_dims is None:
            msg = "Layer dimensions not found."
            raise ValueError(msg)

        # Return raw gradients if Hessian type is "Identity"
        if self.hessian == "Identity":
            logger.debug("Using raw gradients as IFVP since hessian type is 'Identity'")
            self._copy_gradients_as_ifvp()
            return

        # Load all preconditioners
        preconditioners = []
        for layer_idx in range(len(self.layer_names)):
            precond = self.offload_manager.retrieve_preconditioner(layer_idx)
            preconditioners.append(precond)

        # Get batch mapping
        batch_to_sample_mapping = self.metadata.get_batch_to_sample_mapping()
        processed_batches = 0
        processed_samples = 0

        # Use tensor-based dataloader
        dataloader = self.offload_manager.create_gradient_dataloader(
            data_type="gradients",
            batch_size=4,
            pin_memory=True,
        )

        for chunk_tensor, batch_mapping in tqdm(dataloader, desc="Computing IFVP"):
            # Move chunk to device
            chunk_tensor_device = self.offload_manager.move_to_device(chunk_tensor)

            # Process each batch in the chunk
            for batch_idx, (start_row, end_row) in batch_mapping.items():
                if batch_idx not in batch_to_sample_mapping:
                    continue

                batch_tensor = chunk_tensor_device[start_row:end_row]
                batch_ifvp = []

                # Process each layer
                for layer_idx in range(len(self.layer_names)):
                    if preconditioners[layer_idx] is None:
                        batch_ifvp.append(
                            torch.zeros(
                                batch_tensor.shape[0],
                                self.layer_dims[layer_idx],
                            ),
                        )
                        continue

                    # Extract layer data
                    start_col = sum(self.layer_dims[:layer_idx])
                    end_col = start_col + self.layer_dims[layer_idx]
                    layer_grad = batch_tensor[:, start_col:end_col]

                    if layer_grad.numel() == 0:
                        batch_ifvp.append(
                            torch.zeros(
                                batch_tensor.shape[0],
                                self.layer_dims[layer_idx],
                            ),
                        )
                        continue

                    # Get preconditioner
                    device_precond = self.offload_manager.move_to_device(
                        preconditioners[layer_idx],
                    )
                    device_precond = device_precond.to(dtype=layer_grad.dtype)

                    # Compute IFVP: H^{-1} @ g
                    ifvp = torch.matmul(device_precond, layer_grad.t()).t()
                    batch_ifvp.append(ifvp)

                # Store IFVP for this batch
                self.offload_manager.store_ifvp(batch_idx, batch_ifvp)
                processed_batches += 1
                processed_samples += batch_tensor.shape[0]

            torch.cuda.empty_cache()

        self.offload_manager.wait_for_async_operations()
        logger.info(
            "Computed IFVP for %s batches, %s samples",
            processed_batches,
            processed_samples,
        )

    def _copy_gradients_as_ifvp(self) -> None:
        """Copy gradients as IFVP when hessian type is 'none'."""
        # Ensure layer dimensions are loaded
        if self.layer_dims is None:
            self._sync_layer_dims()

        batch_to_sample_mapping = self.metadata.get_batch_to_sample_mapping()
        processed_batches = 0
        processed_samples = 0

        # Process using tensor dataloader
        dataloader = self.offload_manager.create_gradient_dataloader(
            data_type="gradients",
            batch_size=1,
            pin_memory=True,
        )

        for chunk_tensor, batch_mapping in tqdm(
            dataloader,
            desc="Copying gradients as IFVP",
        ):
            for batch_idx, (start_row, end_row) in batch_mapping.items():
                if batch_idx not in batch_to_sample_mapping:
                    continue

                # Extract batch and split into layers
                batch_tensor = chunk_tensor[start_row:end_row]
                gradients = []

                for layer_idx in range(len(self.layer_names)):
                    start_col = sum(self.layer_dims[:layer_idx])
                    end_col = start_col + self.layer_dims[layer_idx]
                    gradients.append(batch_tensor[:, start_col:end_col].contiguous())

                self.offload_manager.store_ifvp(batch_idx, gradients)
                processed_batches += 1
                processed_samples += batch_tensor.shape[0]

            torch.cuda.empty_cache()

        logger.info("Copied %s batches as IFVP", processed_batches)

    def compute_self_attribution(self) -> torch.Tensor:
        """Compute self-influence scores.

        Returns:
            Self-influence scores
        """
        logger.info("Computing self-influence scores")

        if not self.metadata.batch_info:
            self.metadata._load_metadata_if_exists()

        # Synchronize layer dimensions
        self._sync_layer_dims()

        # Make sure IFVP is computed
        if not self.offload_manager.has_ifvp():
            logger.info("IFVP not found, computing it now...")
            self.compute_ifvp()

        # Get batch mapping
        batch_to_sample_mapping = self.metadata.get_batch_to_sample_mapping()
        total_samples = self.metadata.get_total_samples()
        self_influence = torch.zeros(total_samples, device="cpu")

        # Use tensor dataloaders for both gradients and IFVP
        grad_dataloader = self.offload_manager.create_gradient_dataloader(
            data_type="gradients",
            batch_size=4,
            pin_memory=True,
        )

        ifvp_dataloader = self.offload_manager.create_gradient_dataloader(
            data_type="ifvp",
            batch_size=4,
            pin_memory=True,
        )

        if grad_dataloader and ifvp_dataloader:
            # Process in parallel
            for (grad_tensor, grad_mapping), (ifvp_tensor, ifvp_mapping) in tqdm(
                zip(grad_dataloader, ifvp_dataloader),
                desc="Computing self-influence",
                total=len(grad_dataloader),
            ):
                # Move to device
                grad_tensor_device = self.offload_manager.move_to_device(grad_tensor)
                ifvp_tensor_device = self.offload_manager.move_to_device(ifvp_tensor)

                # Process each batch
                for batch_idx in grad_mapping:
                    if (
                        batch_idx not in batch_to_sample_mapping
                        or batch_idx not in ifvp_mapping
                    ):
                        continue

                    sample_start, sample_end = batch_to_sample_mapping[batch_idx]

                    # Extract batch slices
                    grad_start, grad_end = grad_mapping[batch_idx]
                    ifvp_start, ifvp_end = ifvp_mapping[batch_idx]

                    batch_grad = grad_tensor_device[grad_start:grad_end]
                    batch_ifvp = ifvp_tensor_device[ifvp_start:ifvp_end]

                    # Compute dot product
                    batch_influence = torch.sum(batch_grad * batch_ifvp, dim=1).cpu()
                    self_influence[sample_start:sample_end] = batch_influence

                torch.cuda.empty_cache()

        return self_influence

    def attribute(
        self,
        train_dataloader: "DataLoader",
        test_dataloader: "DataLoader",
    ) -> torch.Tensor:
        """Compute influence scores between training and test samples.

        Args:
            train_dataloader: Training data (can be subset if cache was called)
            test_dataloader: Test data to compute influence for

        Returns:
            Influence score tensor of shape (num_train, num_test)
        """
        logger.info("Computing influence attribution with LoGra")

        # Validation
        _check_shuffle(train_dataloader)
        _check_shuffle(test_dataloader)

        # Handle training data
        if self.full_train_dataloader is None:
            warnings.warn(
                "The full training data loader was NOT cached. "
                "Treating the train_dataloader as the full training data loader.",
                stacklevel=2,
            )
            self.cache(train_dataloader)

        # Load batch information if needed
        if not self.metadata.batch_info:
            self.metadata._load_metadata_if_exists()

        # Synchronize layer dimensions
        self._sync_layer_dims()

        # Set up projectors if needed
        if self.projectors is None:
            self._setup_projectors(test_dataloader)

        # Get or compute IFVP
        use_cached_ifvp = self.offload_manager.has_ifvp()
        if use_cached_ifvp:
            logger.info("Using cached IFVP")
        else:
            logger.info("Computing IFVP")
            self.compute_ifvp()

        # Compute test gradients
        logger.info("Computing test gradients")
        test_grads_tensor, _test_batch_mapping = self._compute_test_gradients(
            test_dataloader,
        )

        if test_grads_tensor is None or test_grads_tensor.numel() == 0:
            logger.warning("No test gradients computed")
            num_train = (
                len(train_dataloader.sampler)
                if hasattr(train_dataloader, "sampler")
                else len(train_dataloader.dataset)
            )
            num_test = (
                len(test_dataloader.sampler)
                if hasattr(test_dataloader, "sampler")
                else len(test_dataloader.dataset)
            )
            return torch.zeros(num_train, num_test)

        # Get batch mappings
        batch_to_sample_mapping = self.metadata.get_batch_to_sample_mapping()
        total_train_samples = self.metadata.get_total_samples()
        test_sample_count = test_grads_tensor.shape[0]

        # Initialize result
        if_score = torch.zeros(
            total_train_samples,
            test_sample_count,
            device=self.device,
        )

        # Create dataloader for IFVP
        train_ifvp_dataloader = self.offload_manager.create_gradient_dataloader(
            data_type="ifvp",
            batch_size=2,
            pin_memory=True,
        )

        logger.info("Starting influence computation")

        # Configure test batching for memory efficiency
        test_batch_size = min(32, test_sample_count)

        # Single pass through training IFVP data with nested test batching
        for chunk_tensor, batch_mapping in tqdm(
            train_ifvp_dataloader,
            desc="Computing attribution",
        ):
            # Move train chunk to device
            chunk_tensor_device = self.offload_manager.move_to_device(chunk_tensor).to(
                dtype=test_grads_tensor.dtype,
            )

            # Process test gradients in batches to save memory
            for test_start in range(0, test_sample_count, test_batch_size):
                test_end = min(test_start + test_batch_size, test_sample_count)
                test_batch = test_grads_tensor[test_start:test_end]

                # Move test batch to device
                test_batch_device = self.offload_manager.move_to_device(test_batch)

                # Efficient batched matrix multiplication
                chunk_scores = torch.matmul(chunk_tensor_device, test_batch_device.t())

                # Map chunk results back to global sample indices
                for batch_idx, (start_row, end_row) in batch_mapping.items():
                    if batch_idx not in batch_to_sample_mapping:
                        continue

                    train_start, train_end = batch_to_sample_mapping[batch_idx]
                    batch_scores = chunk_scores[start_row:end_row]
                    if_score[train_start:train_end, test_start:test_end] = (
                        batch_scores.to(if_score.device)
                    )

                torch.cuda.empty_cache()

            torch.cuda.empty_cache()

        logger.info(
            "Attribution computation completed. Result shape: %s",
            if_score.shape,
        )
        return if_score

    def _compute_test_gradients(
        self,
        test_dataloader: "DataLoader",
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute gradients for test data using hooks.

        Args:
            test_dataloader: DataLoader for test data to compute gradients for.

        Returns:
            Tuple containing:
                - Tensor of concatenated test gradients
                  Shape:(num_test_samples, total_proj_dim)
                - Dictionary mapping batch indices to (start_row, end_row) positions.

        Raises:
            ValueError: If model can't return loss for dict-style inputs.
        """
        # Create hook manager if needed
        if self.hook_manager is None:
            self.hook_manager = HookManager(
                self.model,
                self.layer_names,
            )
            if self.projectors:
                self.hook_manager.set_projectors(self.projectors)

        all_test_grads = []
        test_batch_mapping = {}
        current_row = 0

        for batch_idx, batch in enumerate(
            tqdm(test_dataloader, desc="Computing test gradients"),
        ):
            self.model.zero_grad()

            # Prepare inputs and compute loss
            if isinstance(batch, dict):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                batch_size = next(iter(batch.values())).shape[0]
                outputs = self.model(**inputs)
                if hasattr(outputs, "loss"):
                    loss = outputs.loss
                else:
                    msg = "Model must return loss for dict-style inputs"
                    raise ValueError(msg)
            else:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                batch_size = inputs.shape[0]
                outputs = self.model(inputs)
                if hasattr(outputs, "loss"):
                    loss = outputs.loss
                else:
                    loss = nn.functional.cross_entropy(outputs, targets)

            loss.backward()

            # Get projected gradients
            with torch.no_grad():
                compressed_grads = self.hook_manager.get_compressed_grads()

                # Concatenate layer gradients
                batch_features = []
                for grad_idx, grad in enumerate(compressed_grads):
                    if grad is not None and grad.numel() > 0:
                        batch_features.append(grad.cpu())
                    elif self.layer_dims and grad_idx < len(self.layer_dims):
                        batch_features.append(
                            torch.zeros(batch_size, self.layer_dims[grad_idx]),
                        )

                if batch_features:
                    batch_tensor = torch.cat(batch_features, dim=1)
                    all_test_grads.append(batch_tensor)
                    test_batch_mapping[batch_idx] = (
                        current_row,
                        current_row + batch_size,
                    )
                    current_row += batch_size

            torch.cuda.empty_cache()

        # Clean up hooks
        self.hook_manager.remove_hooks()
        self.hook_manager = None

        # Combine all test gradients
        if all_test_grads:
            return torch.cat(all_test_grads, dim=0), test_batch_mapping
        return torch.empty(0, self.total_proj_dim or 0), {}
