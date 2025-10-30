"""Data Value Embedding (DVEmb) attributor for trajectory-specific data influence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from yaml import warnings

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Dict, List, Optional, Tuple

    from torch import Tensor
    from torch.utils.data import DataLoader

    from dattri.task import AttributionTask

import math

import torch
from torch import nn
from tqdm import tqdm

from dattri.func.utils import flatten_params


class DVEmbAttributor:
    """Data Value Embedding (DVEmb) attributor.

    DVEmb captures temporal dependence in training by computing data value embeddings
    that approximate trajectory-specific leave-one-out influence. This implementation
    stores embeddings for each epoch separately, allowing for epoch-specific analysis.
    """

    def __init__(
        self,
        task: AttributionTask,
        criterion: nn.Module,
        proj_dim: Optional[int] = None,
        factorization_type: Optional[str] = "none",
        layer_names: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """Initializes the DVEmb attributor.

        Args:
            task: Task to attribute. Must be an instance of `AttributionTask`.
                  Note: The checkpoint functionality of the task is not used by DVEmb.
            criterion: The loss function module (e.g., nn.CrossEntropyLoss())
                       used for gradient factorization.
            proj_dim: The dimension for projection (if used).
            factorization_type: Type of gradient factorization to use. Options are
                                "none"(default),
                                "kronecker"(same with paper),
                                or "elementwise"(better performance while
                                using same projection dimension).
            layer_names: Names of layers where gradients will be collected.
                If None, uses all Linear layers.
                You can check the names using model.named_modules().
                Hooks will be registered on these layers to collect gradients.
                Only available when factorization_type is not "none".

        Raises:
            ValueError: If an unknown factorization type is provided
                        or if no Linear layers are found for factorization.
        """
        self.task = task
        self.model = task.get_model()
        self.device = next(self.model.parameters()).device
        self.criterion = criterion
        self.projector = None
        self.use_factorization = factorization_type != "none"
        self.factorization_type = factorization_type
        self.projection_dim = proj_dim

        if layer_names is None:
            self.layer_names = None
        elif isinstance(layer_names, str):
            self.layer_names = [layer_names]
        else:
            self.layer_names = layer_names

        if self.use_factorization:
            self.cached_factors: Dict[int, List[List[Dict[str, Tensor]]]] = {}
            self._linear_layers = self._get_target_linear_layers()
            if not self._linear_layers:
                msg = "No Linear layers found for gradient factorization."
                raise ValueError(msg)
            if proj_dim is None:
                self._params_dim = sum(
                    p.numel()
                    for layer in self._linear_layers
                    for p in layer.parameters()
                )
            else:
                if self.factorization_type == "kronecker":
                    self.projection_dim = int(
                        math.sqrt(proj_dim / len(self._linear_layers)),
                    )
                    self._params_dim = (
                        len(self._linear_layers)
                        * self.projection_dim
                        * self.projection_dim
                    )
                elif self.factorization_type == "elementwise":
                    self.projection_dim = int(proj_dim / len(self._linear_layers))
                    self._params_dim = len(self._linear_layers) * self.projection_dim
                else:
                    msg = f"Unknown factorization type: {self.factorization_type}"
                    raise ValueError(msg)
                # projectors for each gradient factor (d1 x k) and (d2 x k)
                self.random_projectors = []
                for layer in self._linear_layers:
                    output_dim = layer.weight.shape[0]
                    input_dim = layer.weight.shape[1]
                    project_output = self._generate_projector(
                        output_dim,
                        self.projection_dim,
                    )
                    project_input = self._generate_projector(
                        input_dim,
                        self.projection_dim,
                    )
                    self.random_projectors.append((project_input, project_output))

        self.cached_gradients: Dict[int, List[Tensor]] = {}
        self.learning_rates: Dict[int, List[float]] = {}
        self.data_indices: Dict[int, List[Tensor]] = {}
        self.embeddings: Dict[int, Tensor] = {}

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
                warnings.warn(f"Layer with name '{name}' not found in the model.")
        return target_layers

    @staticmethod
    def _generate_projector(dim: int, proj_dim: int) -> Tensor:
        """Generates a random projection matrix.

        Args:
            dim: The original dimension of the features.
            proj_dim: The dimension to project to.

        Returns:
            A random projection matrix.
        """
        return torch.randn(dim, proj_dim) / torch.sqrt(
            torch.tensor(proj_dim, dtype=torch.float32),
        )

    def _register_factorization_hooks(self) -> Tuple[list, list]:
        """Register forward/backward hooks on each Linear layer to collect factors.

        Args:
            model: The PyTorch model.

        Returns:
            A tuple containing lists of handles and caches for the hooks.
        """
        handles, caches = [], []
        caches.extend(
            [{"A": None, "B": None, "has_bias": False} for _ in self._linear_layers],
        )

        def fwd_hook(idx: int) -> Callable:
            def _hook(
                layer: nn.Module,
                inputs: Tuple[Tensor, ...],
                _output: Tensor,
            ) -> None:
                a = inputs[0].detach()

                if a.dim() > 2:  # noqa: PLR2004
                    a = a.reshape(a.size(0), -1)
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
                if b.dim() > 2:  # noqa: PLR2004
                    b = b.reshape(b.size(0), -1)
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
            if self.projector is None:
                num_features = per_sample_grads.shape[1]
                self.projector = self._generate_projector(
                    num_features,
                    self.projection_dim,
                ).to(
                    device=self.device,
                    dtype=per_sample_grads.dtype,
                )
            per_sample_grads @= self.projector

        return per_sample_grads

    def _cache_full_gradients(
        self,
        epoch: int,
        batch_data: tuple[Tensor, ...],
        indices: Tensor,
        learning_rate: float,
    ) -> None:
        """Cache full per-sample gradients.

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

        self.cached_gradients[epoch].append(per_sample_grads.cpu() / len(indices))
        self.learning_rates[epoch].append(learning_rate)
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
        outputs = self.model(batch_data_tensors[0])
        loss = self.criterion(outputs, batch_data_tensors[1])
        loss.backward()

        for h in handles:
            h.remove()

        return caches

    def _reconstruct_gradients(  # noqa: PLR6301
        self,
        gradient_factors: List[Dict[str, Tensor]],
    ) -> Tensor:
        """Reconstruct per-sample gradients from their factors.

        Args:
            gradient_factors: A list of factor dictionaries for each layer.

        Returns:
            A tensor of reconstructed per-sample gradients.
        """
        projected_grads_parts = []
        for item in gradient_factors:
            a = item["A"]
            b = item["B"]
            c = a.unsqueeze(2) @ b.unsqueeze(1)
            projected_grads_parts.append(c.flatten(start_dim=1))
            if item["has_bias"]:
                projected_grads_parts.append(b)
        return torch.cat(projected_grads_parts, dim=1)

    def _project_gradients_factors_kronecker(
        self,
        gradient_factors: List[Dict[str, Tensor]],
    ) -> List[Dict[str, Tensor]]:
        """Project gradient factors using Kronecker product.

        Args:
            gradient_factors: A list of factor dictionaries for each layer.

        Returns:
            A list of projected gradient factor dictionaries.
        """
        proj_factors = []
        for (proj_a, proj_b), item in zip(
            self.random_projectors,
            gradient_factors,
        ):
            a_proj = item["A"] @ proj_a.to(self.device)
            b_proj = item["B"] @ proj_b.to(self.device)
            proj_factors.append({"A": a_proj, "B": b_proj, "has_bias": False})
        return proj_factors

    def _project_gradients_factors_elementwise(
        self,
        gradient_factors: List[Dict[str, Tensor]],
    ) -> Tensor:
        """Project gradient factors using element-wise product.

        Args:
            gradient_factors: A list of factor dictionaries for each layer.

        Returns:
            A tensor of projected gradients.
        """
        projected_grads_parts = []
        for (proj_a, proj_b), item in zip(
            self.random_projectors,
            gradient_factors,
        ):
            a_proj = item["A"] @ proj_a.to(self.device)
            b_proj = item["B"] @ proj_b.to(self.device)
            grad = a_proj.mul(b_proj)
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
            proj_factors = self._project_gradients_factors_kronecker(caches)
            self.cached_factors[epoch].append(proj_factors)
        else:
            projected_grads = self._project_gradients_factors_elementwise(caches)
            self.cached_gradients[epoch].append(projected_grads.cpu() / len(indices))

        self.learning_rates[epoch].append(learning_rate)
        self.data_indices[epoch].append(indices.cpu())

    def compute_embeddings(  # noqa: PLR0912, PLR0914
        self,
        gradients: Optional[Dict[int, List[Tensor]]] = None,
        learning_rates: Optional[Dict[int, List[float]]] = None,
    ) -> None:
        """Computes data value embeddings for each epoch separately.

        Args:
            gradients: Optional external gradients instead of cached ones
                (e.g., (epoch -> list of per-sample gradients)).
            learning_rates: Optional external learning rates instead of cached ones
                (e.g., (epoch -> list of learning rates)).

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
            msg = "No gradients cached. Call cache_gradients during training first."
            raise ValueError(msg)

        all_indices = torch.cat(
            [torch.cat(indices) for indices in self.data_indices.values()],
        )
        num_total_samples = all_indices.max().item() + 1

        # Determine gradient dimension and data type from cached data
        if self.use_factorization:
            grad_dim = self._params_dim
            if self.projection_dim is None or self.factorization_type == "kronecker":
                grad_dtype = self.cached_factors[0][0][0]["A"].dtype
            else:
                grad_dtype = self.cached_gradients[0][0].dtype
        else:
            grad_dim = self.cached_gradients[0][0].shape[1]
            grad_dtype = self.cached_gradients[0][0].dtype

        m_matrix = torch.zeros(
            (grad_dim, grad_dim),
            device=self.device,
            dtype=grad_dtype,
        )

        sorted_epochs = sorted(self.learning_rates.keys(), reverse=True)
        for epoch in tqdm(sorted_epochs, desc="Computing DVEmb per Epoch"):
            epoch_embeddings = torch.zeros(
                (num_total_samples, grad_dim),
                device=self.device,
                dtype=grad_dtype,
            )
            num_iterations_in_epoch = len(self.learning_rates[epoch])

            for t in reversed(range(num_iterations_in_epoch)):
                eta_t = self.learning_rates[epoch][t]
                indices_t = self.data_indices[epoch][t]

                if self.use_factorization:
                    if (
                        self.projection_dim is None
                        or self.factorization_type == "kronecker"
                    ):
                        gradient_factors = self.cached_factors[epoch][t]
                        grads_t = self._reconstruct_gradients(gradient_factors).to(
                            self.device,
                        )
                    else:
                        grads_t = self.cached_gradients[epoch][t].to(self.device)
                else:
                    grads_t = self.cached_gradients[epoch][t].to(self.device)

                dvemb_t = eta_t * grads_t - eta_t * (grads_t @ m_matrix)

                epoch_embeddings.index_add_(0, indices_t.to(self.device), dvemb_t)
                m_matrix += dvemb_t.T @ grads_t

                if torch.isnan(m_matrix).any() or torch.isnan(dvemb_t).any():
                    msg = (
                        f"NaN detected at epoch {epoch}, iteration {t}. "
                        "Learning rate may be too high or gradients are unstable."
                    )
                    raise ValueError(msg)

            self.embeddings[epoch] = epoch_embeddings

    def attribute(
        self,
        test_dataloader: DataLoader,
        epoch: Optional[int] = None,
        traindata_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Calculates influence scores for a test set for one or all epochs.

        Args:
            test_dataloader: A dataloader for the test set.
            epoch: Optional. If specified, returns scores using embeddings from that
                   epoch. If None, returns scores based on the sum of embeddings
                   across all epochs.
            traindata_indices: Optional. A list of training sample indices for which
                               to compute influence. If None, computes for all.

        Returns:
            A tensor of influence scores.

        Raises:
            RuntimeError: If embeddings have not been computed by calling
                          `compute_embeddings` first, or if a projection dimension
                          was specified but the projector is not initialized.
            ValueError: If embeddings for the specified `epoch` are not found.
        """
        if not self.embeddings:
            msg = "Embeddings not computed. Call compute_embeddings first."
            raise RuntimeError(msg)

        self.model.eval()
        test_grads = []
        for batch_data in tqdm(test_dataloader, desc="Calculating test gradients"):
            if self.use_factorization:
                factors = self._calculate_gradient_factors(batch_data)
                if self.projection_dim is None:
                    grads = self._reconstruct_gradients(factors)
                elif self.factorization_type == "kronecker":
                    proj_factors = self._project_gradients_factors_kronecker(factors)
                    grads = self._reconstruct_gradients(proj_factors)
                else:
                    grads = self._project_gradients_factors_elementwise(factors)
            else:
                grads = self._calculate_full_gradients(batch_data)
            test_grads.append(grads)
        test_grads_tensor = torch.cat(test_grads, dim=0).to(self.device)

        if epoch is not None:
            if epoch not in self.embeddings:
                msg = f"Embeddings for epoch {epoch} not found."
                raise ValueError(msg)
            active_embeddings = self.embeddings[epoch]
        else:
            active_embeddings = torch.stack(list(self.embeddings.values())).sum(dim=0)

        if traindata_indices is not None:
            train_embeddings = active_embeddings[traindata_indices]
        else:
            train_embeddings = active_embeddings

        return torch.matmul(train_embeddings.to(self.device), test_grads_tensor.T)
