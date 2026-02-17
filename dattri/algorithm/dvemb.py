"""Data Value Embedding (DVEmb) attributor for trajectory-specific data influence."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Union

from dattri.params.projection import DVEmbProjectionParams, RandomProjectionParams

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Dict, List, Optional, Tuple

    from torch import Tensor
    from torch.utils.data import DataLoader

    from dattri.task import AttributionTask

import inspect
import math

import torch
from torch import nn
from tqdm import tqdm

from dattri.func.projection import random_project
from dattri.func.utils import flatten_params


class DVEmbAttributor:
    """Data Value Embedding (DVEmb) attributor.

    DVEmb captures temporal dependence in training by computing data value embeddings
    that approximate trajectory-specific leave-one-out influence. This implementation
    stores embeddings for each epoch separately, allowing for epoch-specific analysis.
    """

    def __init__(  # noqa: PLR0912, PLR0915
        self,
        task: AttributionTask,
        proj_params: Optional[DVEmbProjectionParams] = None,
        factorization_type: Optional[str] = "none",
        layer_names: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """Initializes the DVEmb attributor.

        Args:
            task: Task to attribute. Must be an instance of `AttributionTask`.
                  Note: The checkpoint functionality of the task is not used by DVEmb.
                  The loss function of the task must follow specific formats:
                    - If `factorization_type` is "none", the loss function should follow
                      the signature of the following example:
                        ```python
                        def f(params, data):
                            image, label = data
                            loss = nn.CrossEntropyLoss()
                            yhat = torch.func.functional_call(model, params, image)
                            return loss(yhat, label)
                        ```.
                    - If `factorization_type` is not "none", the loss function should
                      follow the signature of the following example:
                        ```python
                        def f(model, data, device):
                            image, label = data
                            loss = nn.CrossEntropyLoss()
                            yhat = model(image.to(device))
                            return loss(yhat, label.to(device))
                        ```.
            proj_params: Projection config. If None, no projection (proj_dim=None).
                Use DVEmbProjectionParams(proj_dim=...) for projection.
            factorization_type: Type of gradient factorization to use. Options are
                                "none" (default),
                                "kronecker" (same as in the paper),
                                or "elementwise" (efficiently projects Kronecker
                                products via factorized elementwise products).
            layer_names: Names of layers where gradients will be collected.
                If None, uses all Linear layers.
                You can check the names using model.named_modules().
                Hooks will be registered on these layers to collect gradients.
                Will only be used when factorization_type is not "none".

        Raises:
            ValueError: If an unknown factorization type is provided
                        or if no Linear layers are found for factorization or
                        if the loss function format is incorrect.
        """
        self.task = task
        self.model = task.get_model()
        self.device = next(self.model.parameters()).device
        self.projector = None
        self.use_factorization = factorization_type != "none"
        self.factorization_type = factorization_type
        self.proj_params = proj_params or DVEmbProjectionParams(
            proj_dim=None,
            proj_max_batch_size=64,
        )
        self.projection_dim = self.proj_params.proj_dim

        if layer_names is None:
            self.layer_names = None
        elif isinstance(layer_names, str):
            self.layer_names = [layer_names]
        else:
            self.layer_names = layer_names

        # Check loss function format
        sig = inspect.signature(self.task.original_loss_func)
        count = len([p for p in sig.parameters.values() if p.default == p.empty])

        # TODO: Use a more robust way to check function signature
        error_msg = "Wrong loss function format for factorization.\
                     Please refer to the docstring."
        if self.use_factorization:
            if count != 3:  # noqa: PLR2004
                raise ValueError(error_msg)
        elif count != 2:  # noqa: PLR2004
            raise ValueError(error_msg)

        # Create meta-information for factorized gradient caching
        if self.use_factorization:
            self.cached_factors: Dict[int, List[List[Dict[str, Tensor]]]] = {}
            self._linear_layers = self._get_target_linear_layers()
            if not self._linear_layers:
                msg = "No Linear layers found for gradient factorization."
                raise ValueError(msg)
            if self.projection_dim is None:
                self._params_dim = sum(
                    p.numel()
                    for layer in self._linear_layers
                    for p in layer.parameters()
                )
            elif self.factorization_type == "kronecker":
                self.projection_dim = int(
                    math.sqrt(self.projection_dim / len(self._linear_layers)),
                )
                self._params_dim = (
                    len(self._linear_layers) * self.projection_dim * self.projection_dim
                )
            elif self.factorization_type == "elementwise":
                self.projection_dim = int(
                    self.projection_dim / len(self._linear_layers),
                )
                self._params_dim = len(self._linear_layers) * self.projection_dim
            else:
                msg = f"Unknown factorization type: {self.factorization_type}"
                raise ValueError(msg)

        self.cached_gradients: Dict[int, List[Tensor]] = {}
        self.learning_rates: Dict[int, List[float]] = {}
        self.data_indices: Dict[int, List[Tensor]] = {}
        self.embeddings: Dict[int, Tensor] = {}

    def _setup_projectors(self, batch_size: int) -> None:
        """Sets up random projectors for each Linear layer based on the projection type.

        Args:
            batch_size: The batch size to be used for creating projectors.

        Creates self.random_projectors as a list of tuples containing
        (input_projector, output_projector) for each Linear layer.
        """
        self.random_projectors = []
        for layer in self._linear_layers:
            output_dim = layer.weight.shape[0]
            input_dim = layer.weight.shape[1]
            project_output = random_project(
                feature=torch.zeros((batch_size, output_dim)),
                proj_params=RandomProjectionParams(
                    feature_batch_size=batch_size,
                    device=self.device,
                    proj_dim=self.projection_dim,
                    proj_seed=self.proj_params.proj_seed,
                    proj_max_batch_size=self.proj_params.proj_max_batch_size,
                ),
            )
            project_input = random_project(
                feature=torch.zeros((batch_size, input_dim)),
                proj_params=RandomProjectionParams(
                    feature_batch_size=batch_size,
                    device=self.device,
                    proj_dim=self.projection_dim,
                    proj_seed=self.proj_params.proj_seed,
                    proj_max_batch_size=self.proj_params.proj_max_batch_size,
                ),
            )
            self.random_projectors.append((project_input, project_output))

    def _get_target_linear_layers(self) -> List[nn.Module]:
        """Gets the target Linear layers based on specified layer names.

        If `self.layer_names` is specified, it selects layers by name.
        Otherwise, it defaults to selecting all `nn.Linear` layers.

        Returns:
            A list of PyTorch modules to apply hooks to.
        """
        if self.layer_names is None:
            return [m for m in self.model.modules() if isinstance(m, nn.Linear)]

        target_layers = []
        model_layers = dict(self.model.named_modules())
        for name in self.layer_names:
            if name in model_layers:
                target_layers.append(model_layers[name])
            else:
                warnings.warn(
                    f"Layer with name '{name}' not found in the model.",
                    stacklevel=2,
                )
        return target_layers

    def _register_factorization_hooks(self) -> Tuple[list, list]:
        """Register forward/backward hooks on each Linear layer to collect factors.

        Returns:
            A tuple containing lists of handles and caches for the hooks.
        """
        handles, caches = [], []
        caches.extend(
            [{"A": None, "B": None, "has_bias": False} for _ in self._linear_layers],
        )  # A is input and B is pre-activation gradient

        # Use saved_tensors_hooks instead
        # https://github.com/TRAIS-Lab/dattri/pull/193#discussion_r2683833611
        def fwd_hook(idx: int) -> Callable:
            def _hook(
                layer: nn.Module,
                inputs: Tuple[Tensor, ...],
                _output: Tensor,
            ) -> None:
                a = inputs[0].detach()
                # Keep original dimensions (2D or 3D) for proper gradient computation
                caches[idx]["A"] = a
                caches[idx]["has_bias"] = layer.bias is not None

            return _hook

        def bwd_hook(idx: int) -> Callable:
            def _hook(
                _layer: nn.Module,
                _grad_inputs: Tuple[Tensor, ...],
                grad_outputs: Tuple[Tensor, ...],
            ) -> None:
                b = grad_outputs[0].detach()
                # Keep original dimensions (2D or 3D) for proper gradient computation
                caches[idx]["B"] = b

            return _hook

        for i, layer in enumerate(self._linear_layers):
            handles.extend(
                [
                    layer.register_forward_hook(fwd_hook(i)),
                    layer.register_full_backward_hook(bwd_hook(i)),
                ],
            )
        return handles, caches

    def cache_gradients(
        self,
        epoch: int,
        batch_data: tuple[Tensor, ...],
        indices: Tensor,
        learning_rate: float,
    ) -> None:
        """Cache per-sample gradients for a specific epoch and training step.

        Args:
            epoch: The current epoch number.
            batch_data: A tuple containing the batch of inputs and targets,
                        e.g., (inputs, labels).
            indices: A tensor containing the original indices for the samples
                     in batch_data.
            learning_rate: The learning rate for this step.
        """
        # Set up projectors if not already done
        if (
            hasattr(self, "random_projectors") is False
            and self.use_factorization
            and self.projection_dim is not None
        ):
            self._setup_projectors(batch_size=len(indices))

        if self.use_factorization:
            self._cache_factored_gradients(epoch, batch_data, indices, learning_rate)
        else:
            self._cache_full_gradients(epoch, batch_data, indices, learning_rate)

    def _calculate_full_gradients(
        self,
        batch_data: tuple[Tensor, ...],
    ) -> Tensor:
        """Calculate full per-sample gradients.

        Args:
            batch_data: A tuple of data tensors for a batch.

        Returns:
            A tensor of per-sample gradients, possibly projected.
        """
        batch_data_tensors = tuple(d.to(self.device) for d in batch_data)

        in_dims = (None, tuple(0 for _ in batch_data_tensors))
        grad_loss_fn = self.task.get_grad_loss_func(in_dims=in_dims)

        model_params = flatten_params(
            {k: v.detach() for k, v in self.model.named_parameters()},
        )
        per_sample_grads = grad_loss_fn(model_params, batch_data_tensors)

        if self.projection_dim is not None:
            # Create projector if needed
            if self.projector is None:
                num_features = per_sample_grads.shape[1]
                batch_size = per_sample_grads.shape[0]
                self.projector = random_project(
                    torch.zeros((batch_size, num_features)),
                    proj_params=RandomProjectionParams(
                        feature_batch_size=batch_size,
                        device=self.device,
                        proj_dim=self.projection_dim,
                        proj_seed=self.proj_params.proj_seed,
                        proj_max_batch_size=self.proj_params.proj_max_batch_size,
                    ),
                )
            # Project gradients
            per_sample_grads = self.projector(per_sample_grads)

        return per_sample_grads

    def _cache_full_gradients(
        self,
        epoch: int,
        batch_data: tuple[Tensor, ...],
        indices: Tensor,
        learning_rate: float,
    ) -> None:
        """Cache full per-sample gradients. Offload to CPU to save GPU memory.

        Args:
            epoch: The current epoch number.
            batch_data: A tuple of data tensors for a batch.
            indices: Original indices of the samples in the batch.
            learning_rate: The learning rate for the current step.
        """
        if epoch not in self.cached_gradients:
            self.cached_gradients[epoch] = []
            self.learning_rates[epoch] = []
            self.data_indices[epoch] = []

        per_sample_grads = self._calculate_full_gradients(batch_data)

        self.cached_gradients[epoch].append(per_sample_grads.cpu())
        self.learning_rates[epoch].append(learning_rate / len(indices))
        self.data_indices[epoch].append(indices.cpu())

    def _calculate_gradient_factors(
        self,
        batch_data: tuple[Tensor, ...],
    ) -> List[Dict[str, Tensor]]:
        """Calculate per-layer gradient factors.

        Args:
            batch_data: A tuple of data tensors for a batch.

        Returns:
            A list of cached factor dictionaries for each linear layer.
        """
        self.model.zero_grad()
        handles, caches = self._register_factorization_hooks()

        batch_data_tensors = [d.to(self.device) for d in batch_data]
        loss = self.task.original_loss_func(self.model, batch_data_tensors, self.device)
        loss *= batch_data_tensors[0].shape[0]  # Scale loss by batch size
        loss.backward()

        for h in handles:
            h.remove()

        return caches

    def _reconstruct_gradients(  # noqa: PLR6301
        self,
        gradient_factors: List[Dict[str, Tensor]],
    ) -> Tensor:
        """Reconstruct per-sample gradients from their factors.

        Supports both 2D (B, D) and 3D (B, L, D) activations/gradients.
        For 3D inputs, uses einsum to properly sum over the sequence dimension.

        Args:
            gradient_factors: A list of factor dictionaries for each layer.

        Returns:
            A tensor of reconstructed per-sample gradients.
        """
        projected_grads_parts = []
        for item in gradient_factors:
            a = item["A"]  # Activations: (B, D_in) or (B, L, D_in)
            b = item["B"]  # Grad Output: (B, D_out) or (B, L, D_out)

            if a.dim() == 3:  # noqa: PLR2004
                # 3D case: (B, L, D_in) and (B, L, D_out)
                # grad_W = sum_l (B_l^T @ A_l), result shape: (B, D_out, D_in)
                grad_w = torch.einsum("bli,blo->boi", a, b)
                grad_b = b.sum(dim=1) if item["has_bias"] else None
            else:
                # 2D case: (B, D_in) and (B, D_out)
                # grad_W = B^T @ A, using outer product
                grad_w = b.unsqueeze(2) @ a.unsqueeze(1)  # (B, D_out, D_in)
                grad_b = b if item["has_bias"] else None

            projected_grads_parts.append(grad_w.flatten(start_dim=1))
            if grad_b is not None:
                projected_grads_parts.append(grad_b)

        return torch.cat(projected_grads_parts, dim=1)

    def _project_gradients_factors_kronecker(
        self,
        gradient_factors: List[Dict[str, Tensor]],
    ) -> Tensor:
        """Project gradient factors using Kronecker product.

        Supports both 2D (B, D) and 3D (B, L, D) activations/gradients.
        For 3D inputs, reshapes to (B*L, D) for projection, then uses einsum
        to compute outer product and sum over L.

        Args:
            gradient_factors: A list of factor dictionaries for each layer.

        Returns:
            A tensor of projected gradients with shape
            (B, proj_dim * proj_dim * n_layers).
        """
        projected_grads_parts = []
        for (proj_a, proj_b), item in zip(
            self.random_projectors,
            gradient_factors,
        ):
            a = item["A"]  # (B, D_in) or (B, L, D_in)
            b = item["B"]  # (B, D_out) or (B, L, D_out)

            if a.dim() == 3:  # noqa: PLR2004
                batch_size, seq_len, d_in = a.shape
                d_out = b.shape[2]

                a_flat = a.reshape(batch_size * seq_len, d_in)
                b_flat = b.reshape(batch_size * seq_len, d_out)

                a_proj_flat = proj_a(a_flat)
                b_proj_flat = proj_b(b_flat)

                k = a_proj_flat.shape[1]
                a_proj = a_proj_flat.reshape(batch_size, seq_len, k)
                b_proj = b_proj_flat.reshape(batch_size, seq_len, k)

                grad = torch.einsum("blk,blj->bkj", a_proj, b_proj)
            else:
                # 2D case: (B, D)
                a_proj = proj_a(a)
                b_proj = proj_b(b)
                # Outer product: (B, K, K)
                grad = b_proj.unsqueeze(2) @ a_proj.unsqueeze(1)

            projected_grads_parts.append(grad.flatten(start_dim=1))

        return torch.cat(projected_grads_parts, dim=1)

    def _project_gradients_factors_elementwise(
        self,
        gradient_factors: List[Dict[str, Tensor]],
    ) -> Tensor:
        """Project gradient factors using element-wise product.

        Supports both 2D (B, D) and 3D (B, L, D) activations/gradients.
        For 3D inputs, reshapes to (B*L, D) for projection, then sums over L.

        Args:
            gradient_factors: A list of factor dictionaries for each layer.

        Returns:
            A tensor of projected gradients with shape (B, proj_dim * n_layers).
        """
        projected_grads_parts = []
        for (proj_a, proj_b), item in zip(
            self.random_projectors,
            gradient_factors,
        ):
            a = item["A"]  # (B, D_in) or (B, L, D_in)
            b = item["B"]  # (B, D_out) or (B, L, D_out)

            if a.dim() == 3:  # noqa: PLR2004
                batch_size, seq_len, d_in = a.shape
                d_out = b.shape[2]

                a_flat = a.reshape(batch_size * seq_len, d_in)
                b_flat = b.reshape(batch_size * seq_len, d_out)

                a_proj_flat = proj_a(a_flat)
                b_proj_flat = proj_b(b_flat)

                k = a_proj_flat.shape[1]
                a_proj = a_proj_flat.reshape(batch_size, seq_len, k)
                b_proj = b_proj_flat.reshape(batch_size, seq_len, k)

                grad = (a_proj * b_proj).sum(dim=1) * math.sqrt(self.projection_dim)
            else:
                # 2D case: (B, D)
                a_proj = proj_a(a)
                b_proj = proj_b(b)
                grad = a_proj.mul(b_proj) * math.sqrt(self.projection_dim)

            projected_grads_parts.append(grad.flatten(start_dim=1))

        return torch.cat(projected_grads_parts, dim=1)

    def _cache_factored_gradients(
        self,
        epoch: int,
        batch_data: tuple[Tensor, ...],
        indices: Tensor,
        learning_rate: float,
    ) -> None:
        """Cache per-sample gradients using factorization.

        Args:
            epoch: The current epoch number.
            batch_data: A tuple of data tensors for a batch.
            indices: Original indices of the samples in the batch.
            learning_rate: The learning rate for the current step.
        """
        if epoch not in self.cached_factors:
            self.cached_factors[epoch] = []
            self.cached_gradients[epoch] = []
            self.learning_rates[epoch] = []
            self.data_indices[epoch] = []

        caches = self._calculate_gradient_factors(batch_data)

        if self.projection_dim is None:
            self.cached_factors[epoch].append(caches)
        elif self.factorization_type == "kronecker":
            # kronecker with projection returns projected gradients directly
            projected_grads = self._project_gradients_factors_kronecker(caches)
            self.cached_gradients[epoch].append(projected_grads.cpu())
        else:
            projected_grads = self._project_gradients_factors_elementwise(caches)
            self.cached_gradients[epoch].append(projected_grads.cpu())

        self.learning_rates[epoch].append(learning_rate / len(indices))
        self.data_indices[epoch].append(indices.cpu())

    def clear_cache(self) -> None:
        """Clears cached gradients and factors to free memory."""
        if self.use_factorization:
            self.cached_factors.clear()
        self.cached_gradients.clear()
        self.learning_rates.clear()
        self.data_indices.clear()

    def cache(  # noqa: PLR0912, PLR0914, PLR0915
        self,
        gradients: Optional[Dict[int, List[Tensor]]] = None,
        learning_rates: Optional[Dict[int, List[float]]] = None,
        memory_saving: Optional[bool] = True,
    ) -> None:
        """Computes data value embeddings for each epoch separately.

        Args:
            gradients: Optional external gradients instead of cached ones
                (e.g., (epoch -> list of per-sample gradients)).
            learning_rates: Optional external learning rates instead of cached ones
                (e.g., (epoch -> list of learning rates)).
            memory_saving: If True, cached gradients will be cleared from memory
                         after computation to save space.

        Raises:
            ValueError: If no gradients are cached before computation,
                or if NaN values are detected during computation,
                or if external gradients are provided when using gradient factorization.
        """
        if self.use_factorization and gradients is not None:
            msg = (
                "External gradients are not supported when using gradient "
                "factorization."
            )
            raise ValueError(msg)

        if gradients is not None:
            self.cached_gradients = gradients
        if learning_rates is not None:
            self.learning_rates = learning_rates

        has_full_grads = self.cached_gradients and any(self.cached_gradients.values())
        has_gradient_factors = (
            self.use_factorization
            and self.cached_factors
            and any(self.cached_factors.values())
        )
        if not has_full_grads and not has_gradient_factors:
            msg = "No gradients cached. Call `cache_gradients` during training first."
            raise ValueError(msg)

        all_indices = torch.cat(
            [torch.cat(indices) for indices in self.data_indices.values()],
        )
        num_total_samples = all_indices.max().item() + 1

        # Determine gradient dimension and data type from cached data
        if self.use_factorization:
            grad_dim = self._params_dim
            if self.projection_dim is None:
                # No projection: use cached factors
                grad_dtype = self.cached_factors[0][0][0]["A"].dtype
            else:
                # With projection (kronecker or elementwise): use cached gradients
                grad_dtype = self.cached_gradients[0][0].dtype
        else:
            grad_dim = self.cached_gradients[0][0].shape[1]
            grad_dtype = self.cached_gradients[0][0].dtype

        # Create weighting matrix
        m_matrix = torch.zeros(
            (grad_dim, grad_dim),
            device=self.device,
            dtype=grad_dtype,
        )

        # Calculate data embeddings for each epoch in reverse order
        sorted_epochs = sorted(self.learning_rates.keys(), reverse=True)
        for epoch in tqdm(sorted_epochs, desc="Computing DVEmb per Epoch"):
            epoch_embeddings = torch.zeros(
                (num_total_samples, grad_dim),
                device=self.device,
                dtype=grad_dtype,
            )
            num_iterations_in_epoch = len(self.learning_rates[epoch])

            # Calculate data embeddings in reverse order within the epoch
            for t in reversed(range(num_iterations_in_epoch)):
                eta_t = self.learning_rates[epoch][t]
                indices_t = self.data_indices[epoch][t]

                if self.use_factorization:
                    if self.projection_dim is None:
                        # No projection: reconstruct from cached factors
                        gradient_factors = self.cached_factors[epoch][t]
                        grads_t = self._reconstruct_gradients(gradient_factors).to(
                            self.device,
                        )
                    else:
                        # With projection (kronecker or elementwise)
                        grads_t = self.cached_gradients[epoch][t].to(self.device)
                else:
                    grads_t = self.cached_gradients[epoch][t].to(self.device)

                # Update data value embedding
                dvemb_t = eta_t * grads_t - eta_t * (grads_t @ m_matrix.T)

                epoch_embeddings.index_add_(0, indices_t.to(self.device), dvemb_t)
                m_matrix += dvemb_t.T @ grads_t

                # Check for NaN values; if found, report error
                if torch.isnan(m_matrix).any() or torch.isnan(dvemb_t).any():
                    msg = (
                        f"NaN detected at epoch {epoch}, iteration {t}. "
                        "Learning rate may be too high or gradients are unstable."
                    )
                    raise ValueError(msg)

            self.embeddings[epoch] = epoch_embeddings

        if memory_saving:
            self.clear_cache()

    def attribute(
        self,
        test_dataloader: DataLoader,
        epoch: Optional[int] = None,
        train_data_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Calculates influence scores for a test set for one or all epochs.

        Args:
            test_dataloader: A dataloader for the test set.
            epoch: Optional. If specified, returns scores using embeddings from that
                   epoch. If None, returns scores based on the sum of embeddings
                   across all epochs.
            train_data_indices: Optional. A list of training sample indices for which
                               to compute influence. If None, computes for all.

        Returns:
            A tensor of influence scores.

        Raises:
            RuntimeError: If embeddings have not been computed by calling
                          `cache` first, or if a projection dimension
                          was specified but the projector is not initialized.
            ValueError: If embeddings for the specified `epoch` are not found.
        """
        if not self.embeddings:
            msg = "Embeddings not computed. Call cache first."
            raise RuntimeError(msg)

        # Create test gradients
        self.model.eval()
        test_grads = []
        for batch_data in tqdm(test_dataloader, desc="Calculating test gradients"):
            if self.use_factorization:
                factors = self._calculate_gradient_factors(batch_data)
                if self.projection_dim is None:
                    grads = self._reconstruct_gradients(factors)
                elif self.factorization_type == "kronecker":
                    # kronecker with projection returns projected gradients directly
                    grads = self._project_gradients_factors_kronecker(factors)
                else:
                    grads = self._project_gradients_factors_elementwise(factors)
            else:
                grads = self._calculate_full_gradients(batch_data)
            test_grads.append(grads)
        test_grads_tensor = torch.cat(test_grads, dim=0).to(self.device)

        # Select active embeddings (according to epoch if specified)
        if epoch is not None:
            if epoch not in self.embeddings:
                msg = f"Embeddings for epoch {epoch} not found."
                raise ValueError(msg)
            active_embeddings = self.embeddings[epoch]
        else:
            active_embeddings = torch.stack(list(self.embeddings.values())).sum(dim=0)

        # Select training embeddings according to specified indices
        if train_data_indices is not None:
            train_embeddings = active_embeddings[train_data_indices]
        else:
            train_embeddings = active_embeddings

        return torch.matmul(train_embeddings.to(self.device), test_grads_tensor.T)
