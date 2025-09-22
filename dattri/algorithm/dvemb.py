"""Data Value Embedding (DVEmb) attributor for trajectory-specific data influence."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Dict, List, Optional

    from torch import Tensor
    from torch.utils.data import DataLoader

import math

import torch
from torch import nn
from torch.func import grad, vmap
from tqdm import tqdm

from dattri.func.projection import random_project
from dattri.func.utils import flatten_params


class DVEmbAttributor:
    """Data Value Embedding (DVEmb) attributor.

    DVEmb captures temporal dependence in training by computing data value embeddings
    that approximate trajectory-specific leave-one-out influence. This implementation
    stores embeddings for each epoch separately, allowing for epoch-specific analysis.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_func: Callable,
        device: str = "cpu",
        use_projection: bool = False,
        projection_dim: Optional[int] = None,
    ) -> None:
        """Initializes the DVEmb attributor.

        Args:
            model: The PyTorch model to be attributed.
            loss_func: A per-sample loss function. It should take model parameters
                       and a single data sample (unbatched) as input.
            device: The device to run computations on (e.g., "cpu", "cuda").
            use_projection: Specifies if cached gradients should be projected.
            projection_dim: The dimension for projection (if used).
        """
        self.model = model
        self.device = device
        self.per_sample_loss_fn = loss_func
        self.use_projection = use_projection
        self.projection_dim = projection_dim
        self.projector = None

        # Data structures are now dictionaries keyed by epoch
        self.cached_gradients: Dict[int, List[Tensor]] = {}
        self.learning_rates: Dict[int, List[float]] = {}
        self.data_indices: Dict[int, List[Tensor]] = {}
        self.embeddings: Dict[int, Tensor] = {}

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
        self.model.eval()

        if epoch not in self.cached_gradients:
            self.cached_gradients[epoch] = []
            self.learning_rates[epoch] = []
            self.data_indices[epoch] = []

        batch_data_tensors = [d.to(self.device) for d in batch_data]

        def grad_wrapper(
            params: Tensor,
            *single_data_tensors: tuple[Tensor, ...],
        ) -> Tensor:
            return grad(self.per_sample_loss_fn)(params, single_data_tensors)

        per_sample_grad_fn = vmap(
            grad_wrapper,
            in_dims=(None, *([0] * len(batch_data_tensors))),
        )

        model_params = flatten_params(
            {k: v.detach() for k, v in self.model.named_parameters()},
        )
        per_sample_grads = per_sample_grad_fn(model_params, *batch_data_tensors)

        if self.use_projection:
            if self.projector is None:
                self.projector = random_project(
                    per_sample_grads,
                    per_sample_grads.shape[0],
                    proj_dim=self.projection_dim,
                    proj_max_batch_size=per_sample_grads.shape[0],
                    device=self.device,
                )
            projected_grads = self.projector(per_sample_grads)

            scaling_factor = 1.0 / math.sqrt(self.projection_dim)
            per_sample_grads = projected_grads * scaling_factor

        self.cached_gradients[epoch].append(per_sample_grads.cpu())
        self.learning_rates[epoch].append(learning_rate)
        self.data_indices[epoch].append(indices.cpu())

    def compute_embeddings(self) -> None:
        """Computes data value embeddings for each epoch separately."""
        if not self.cached_gradients:
            msg = "No gradients cached. Call cache_gradients during training first."
            raise ValueError(msg)

        all_indices = torch.cat(
            [torch.cat(indices) for indices in self.data_indices.values()],
        )
        num_total_samples = all_indices.max().item() + 1
        grad_dim = self.cached_gradients[0][0].shape[1]
        grad_dtype = self.cached_gradients[0][0].dtype

        m_matrix = torch.zeros(
            (grad_dim, grad_dim),
            device=self.device,
            dtype=grad_dtype,
        )

        sorted_epochs = sorted(self.cached_gradients.keys(), reverse=True)

        for epoch in tqdm(sorted_epochs, desc="Computing DVEmb per Epoch"):
            epoch_embeddings = torch.zeros(
                (num_total_samples, grad_dim),
                device=self.device,
                dtype=grad_dtype,
            )
            num_iterations_in_epoch = len(self.cached_gradients[epoch])

            for t in reversed(range(num_iterations_in_epoch)):
                eta_t = self.learning_rates[epoch][t]
                grads_t = self.cached_gradients[epoch][t].to(self.device)
                indices_t = self.data_indices[epoch][t]

                dvemb_t = eta_t * grads_t - eta_t * (m_matrix @ grads_t.T).T
                if torch.isnan(dvemb_t).any():
                    msg = f"NaN detected in dvemb_t at epoch {epoch}, iteration {t}"
                    raise ValueError(msg)

                epoch_embeddings.index_add_(0, indices_t.to(self.device), dvemb_t)
                m_matrix += dvemb_t.T @ grads_t
                if torch.isnan(m_matrix).any():
                    msg = f"NaN detected in m_matrix at epoch {epoch}, iteration {t}"
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
        """
        if not self.embeddings:
            msg = "Embeddings not computed. Call compute_embeddings first."
            raise RuntimeError(msg)

        self.model.eval()

        test_grads = []
        for batch_data in test_dataloader:
            batch_data_tensors = [d.to(self.device) for d in batch_data]

            def grad_wrapper(
                params: Tensor,
                *single_data_tensors: tuple[Tensor, ...],
            ) -> Tensor:
                return grad(self.per_sample_loss_fn)(params, single_data_tensors)

            per_sample_grad_fn = vmap(
                grad_wrapper,
                in_dims=(None, *([0] * len(batch_data_tensors))),
            )

            model_params = flatten_params(
                {k: v.detach() for k, v in self.model.named_parameters()},
            )
            grads = per_sample_grad_fn(model_params, *batch_data_tensors)

            if self.use_projection:
                if self.projector is None:
                    msg = "Projector is not initialized."
                    raise RuntimeError(msg)
                grads = self.projector(grads)
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

        return torch.matmul(train_embeddings, test_grads_tensor.T)
