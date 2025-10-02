"""Projection matrix constructions.

This file contains functions to (1) construct random projection matrices (entries
are normal or Rademacher random variables) for dimension reduction and (2) perform
eigen decomposition on the inverse Hessian matrix using Arnoldi iteration and derive
the corresponding projection matrix.

Typically, the feature will correspond to gradient w.r.t model parameters.

The code is mainly adapted from https://github.com/MadryLab/trak/blob/main/trak/
and https://github.com/google-research/jax-influence/blob/main/jax_influence/arnoldi.py,
correspondingly.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Dict, List, Union

import warnings

import numpy as np
import torch
from torch import Tensor

from dattri.func.hessian import hvp_at_x

from .utils import _vectorize as vectorize
from .utils import get_parameter_chunk_sizes


class ProjectionType(str, Enum):
    """Projection type used for projectors."""

    normal: str = "normal"
    rademacher: str = "rademacher"


class AbstractProjector(ABC):
    """Base Class for projectors."""

    @abstractmethod
    def __init__(
        self,
        feature_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: Union[str, ProjectionType],
        device: Union[str, torch.device],
    ) -> None:
        """Initializes hyperparameters for the projection.

        Args:
            feature_dim (int): Dimension of the features to be projected.
                Typically, this equals the number of parameters in the model
                (dimension of the gradient vectors).
            proj_dim (int): Dimension after the projection.
            seed (int): Random seed for the generation of the sketching
                (projection) matrix.
            proj_type (Union[str, ProjectionType]): The random projection (JL
                transform) guarantees that distances will be approximately
                preserved for a variety of choices of the random matrix. Here,
                we provide an implementation for matrices with iid Gaussian
                entries and iid Rademacher entries.
            device (Union[str, torch.device]): CUDA device to use.
        """
        self.feature_dim = feature_dim
        self.proj_dim = proj_dim
        self.seed = seed
        self.proj_type = proj_type
        self.device = device

    @abstractmethod
    def project(self, features: Union[dict, Tensor], ensemble_id: int) -> Tensor:
        """Performs the random projection on feature matrix.

        This function will take features and an ensemble_id, which allows us
        to generate different projection matrices, and output the projected
        matrix.

        Args:
            features (Union[dict, Tensor]): A batch of features or a dictionary
                of batch of features.
            ensemble_id (int): A unique ID for this ensemble.

        Returns:
            Tensor: The projected features.
        """

    @abstractmethod
    def free_memory(self) -> None:
        """Frees up memory used by the projector."""


class BasicProjector(AbstractProjector):
    """A simple block-wise implementation of the projection.

    The projection matrix is generated on-device in blocks.
    The accumulated result across blocks is returned.

    Note: This class will be significantly slower and have a larger memory
    footprint than the CudaProjector. It is recommended that you use this method
    only if the CudaProjector is not available to you -- e.g. if you don't have
    a CUDA-enabled device with compute capability >=7.0 (see
    https://developer.nvidia.com/cuda-gpus).
    """

    def __init__(
        self,
        feature_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: Union[str, ProjectionType],
        device: torch.device,
        block_size: int = 100,
        dtype: torch.dtype = torch.float32,
        ensemble_id: int = 0,
    ) -> None:
        """Initializes hyperparameters for BasicProjector.

        Args:
            feature_dim (int): Dimension of the features to be projected.
                Typically, this equals the number of parameters in the model
                (dimension of the gradient vectors).
            proj_dim (int): Dimension after the projection.
            seed (int): Random seed for the generation of the sketching
                (projection) matrix.
            proj_type (Union[str, ProjectionType]): The random projection (JL
                transform) guarantees that distances will be approximately
                preserved for a variety of choices of the random matrix. Here,
                we provide an implementation for matrices with iid Gaussian
                entries and iid Rademacher entries.
            device (torch.device): CUDA device to use.
            block_size (int): Maximum number of projection dimension allowed.
                Thus, min(block_size, proj_dim) will be used as the actual
                projection dimension.
            dtype (torch.dtype): The dtype of the projected matrix.
            ensemble_id (int): A unique ID for this ensemble.
        """
        super().__init__(feature_dim, proj_dim, seed, proj_type, device)

        self.block_size = min(self.proj_dim, block_size)
        self.num_blocks = math.ceil(self.proj_dim / self.block_size)
        self.dtype = dtype
        self.proj_type = proj_type
        self.ensemble_id = ensemble_id

        self.proj_matrix = torch.empty(
            self.feature_dim,
            self.block_size,
            dtype=self.dtype,
            device=self.device,
        )

        self.proj_matrix_available = True

        self.generator = torch.Generator(device=self.device)

        self.get_generator_states()
        self.generate_sketch_matrix(self.generator_states[0])

    def free_memory(self) -> None:
        """Delete the projection matrix."""
        del self.proj_matrix
        self.proj_matrix_available = False

    def get_generator_states(self) -> None:
        """Set generator seeds for each block."""
        self.generator_states = []
        self.seeds = []
        self.jl_size = self.feature_dim * self.block_size

        for i in range(self.num_blocks):
            s = self.seed + int(1e3) * i + int(1e5) * self.ensemble_id
            self.seeds.append(s)
            self.generator = self.generator.manual_seed(s)
            self.generator_states.append(self.generator.get_state())

    def generate_sketch_matrix(self, generator_state: List) -> None:
        """Set generator states and generate sketch matrices.

        Args:
            generator_state (List): A list of generator states. Usually each
                block will be given a unique generator states.

        Raises:
            KeyError: Projection type is not recognized.
        """
        if not self.proj_matrix_available:
            self.proj_matrix = torch.empty(
                self.feature_dim,
                self.block_size,
                dtype=self.dtype,
                device=self.device,
            )
            self.proj_matrix_available = True

        self.generator.set_state(generator_state)

        if self.proj_type in {ProjectionType.normal, "normal"}:
            self.proj_matrix.normal_(generator=self.generator)
        elif self.proj_type in {ProjectionType.rademacher, "rademacher"}:
            self.proj_matrix.bernoulli_(p=0.5, generator=self.generator)
            self.proj_matrix *= 2.0
            self.proj_matrix -= 1.0
        else:
            msg = f"Projection type {self.proj_type} not recognized."
            raise KeyError(msg)

    def project(self, features: Union[dict, Tensor], ensemble_id: int) -> Tensor:
        """Performs the random projection on the feature matrix.

        Args:
            features (Union[dict, Tensor]): A batch of features or a dictionary
                of batch of features.
            ensemble_id (int): A unique ID for this ensemble.

        Returns:
            Tensor: The projected features.
        """
        if isinstance(features, dict):
            features = vectorize(features, device=self.device)
        elif features.device.type != self.device:
            features = features.to(self.device)
        features = features.to(dtype=self.dtype)
        sketch = torch.zeros(
            size=(features.size(0), self.proj_dim),
            dtype=self.dtype,
            device=self.device,
        )

        if ensemble_id != self.ensemble_id:
            self.ensemble_id = ensemble_id
            self.get_generator_states()  # regenerate random seeds for new ensemble_id
            if self.num_blocks == 1:
                self.generate_sketch_matrix(self.generator_states[0])

        if self.num_blocks == 1:
            torch.matmul(features.data, self.proj_matrix, out=sketch)
        else:
            for ind in range(self.num_blocks):
                self.generate_sketch_matrix(self.generator_states[ind])

                st = ind * self.block_size
                ed = min((ind + 1) * self.block_size, self.proj_dim)
                sketch[:, st:ed] = (
                    features.type(self.dtype) @ self.proj_matrix[:, : (ed - st)]
                )
        return sketch.type(features.dtype)


class CudaProjector(AbstractProjector):
    """Projector implemented using CUDA.

    A performant implementation of the projection
    for CUDA with compute capability >= 7.0.
    """

    def __init__(
        self,
        feature_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: ProjectionType,
        device: str,
        max_batch_size: int,
    ) -> None:
        """Initializes hyperparameters for CudaProjector.

        Args:
            feature_dim (int): Dimension of the features to be projected.
                Typically, this equals the number of parameters in the model
                (dimension of the gradient vectors).
            proj_dim (int): Dimension we project *to* during the projection step
            seed (int): Random seed.
            proj_type (ProjectionType): Type of randomness to use for
                projection matrix (rademacher or normal).
            device (str): CUDA device to use.
            max_batch_size (int): Explicitly constrains the batch size of
                the CudaProjector is going to use for projection.
                Set this if you get a 'The batch size of the CudaProjector is
                too large for your GPU' error. Must be either 8, 16, or 32.

        Raises:
            ValueError: When attempting to use this on a non-CUDA device.
            ModuleNotFoundError: When fast_jl is not installed.
        """
        super().__init__(feature_dim, proj_dim, seed, proj_type, device)
        self.max_batch_size = max_batch_size

        if isinstance(device, str):
            device = torch.device(device)

        if device.type != "cuda":
            err = "CudaProjector only works on a CUDA device; \
            Either switch to a CUDA device, or use the BasicProjector"
            raise ValueError(err)

        self.num_sms = torch.cuda.get_device_properties(
            device.index,
        ).multi_processor_count

        try:
            import fast_jl

            # test run to catch at init time if projection goes through
            fast_jl.project_rademacher_8(
                torch.zeros(8, 1_000, device="cuda"),
                512,
                0,
                self.num_sms,
            )
        except ImportError:
            msg = "You should make sure to install the CUDA projector \
            (the fast_jl library)."
            raise ModuleNotFoundError(msg) from None

    def project(
        self,
        features: Union[dict, Tensor],
        ensemble_id: int,
    ) -> Tensor:
        """Performs the random projection on the feature matrix.

        Args:
            features (Union[dict, Tensor]): A batch of features or a dictionary
                of batch of features.
            ensemble_id (int): A unique ID for this ensemble.

        Raises:
            RuntimeError: The batch size of the CudaProjector is too large for
                your GPU.
            RuntimeError: Too many resources requested for launch CUDA.

        Returns:
            Tensor: The projected features.
        """
        import fast_jl

        if isinstance(features, dict):
            features = vectorize(features, device=self.device)
        elif features.device.type != self.device:
            features = features.to(self.device)
        batch_size = features.shape[0]

        effective_batch_size = 32
        min_proj_batch_size = 8
        if batch_size <= min_proj_batch_size:
            effective_batch_size = min_proj_batch_size
        elif batch_size <= min_proj_batch_size * 2:
            effective_batch_size = min_proj_batch_size * 2

        effective_batch_size = min(self.max_batch_size, effective_batch_size)

        function_name = f"project_{self.proj_type}_{effective_batch_size}"

        fn = getattr(fast_jl, function_name)

        try:
            result = fn(
                features,
                self.proj_dim,
                self.seed + int(1e4) * ensemble_id,
                self.num_sms,
            )
        except RuntimeError as e:
            if "CUDA error: too many resources requested for launch" in str(e):
                # provide a more helpful error message
                msg = "The batch size of the CudaProjector is too large for your GPU. \
                    Reduce it by using the proj_max_batch_size argument.\
                    \nOriginal error:"
                raise RuntimeError(msg) from e
            raise e from None

        return result

    def free_memory(self) -> None:
        """A no-op method."""


class ChunkedCudaProjector:
    """Chunked CudaProjector implemented using CUDA.

    This projector is used when (# dim of features)*(# batch size) is too large.
    If the features are gradients, then (# dim of features) equals to the number
    of parameters in the model.
    """

    def __init__(
        self,
        projector_per_chunk: list,
        max_chunk_size: int,
        dim_per_chunk: list,
        feature_batch_size: int,
        proj_max_batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Initializes hyperparameters for ChunkedCudaProjector.

        Args:
            projector_per_chunk (list): A list of projectors. Specifying
                the projector used by each chunk.
            max_chunk_size (int): The maximum size of each chunk.
            dim_per_chunk (list): The number of feature dimensions per chunk.
            feature_batch_size (int): The batch size of input feature.
            proj_max_batch_size (int): The maximum batch size for each projector.
            device (torch.device): Device to use. Will be "cuda" or "cpu".
            dtype (torch.dtype): The dtype of the projected matrix.
        """
        self.projector_per_chunk = projector_per_chunk
        self.proj_dim = self.projector_per_chunk[0].proj_dim
        self.proj_type = self.projector_per_chunk[0].proj_type
        self.dim_per_chunk = dim_per_chunk
        self.feature_batch_size = feature_batch_size
        self.max_chunk_size = max_chunk_size
        self.proj_max_batch_size = proj_max_batch_size
        self.device = device
        self.dtype = dtype
        self.input_allocated = False

    def allocate_input(self) -> None:
        """Allocate zero tensor for input."""
        if self.input_allocated:
            return

        self.ch_input = torch.zeros(
            size=(self.feature_batch_size, self.max_chunk_size),
            device=self.device,
            dtype=self.dtype,
        )

        self.input_allocated = True

    def free_memory(self) -> None:
        """Frees up memory used by the projector."""
        if not self.input_allocated:
            return

        del self.ch_input
        self.input_allocated = False

    def dict_project(self, features: Union[dict, Tensor], ensemble_id: int) -> Tensor:
        """Performs the random projection on the feature matrix.

        Args:
            features (Union[dict, Tensor]): A batch of features or a dictionary
                of batch of features.
            ensemble_id (int): A unique ID for this ensemble.

        Raises:
            ValueError: The number of accumulated #feature dim does not match
                dim_per_chunk.

        Returns:
            Tensor: The projected features.
        """
        self.allocate_input()
        ch_output = torch.zeros(
            size=(self.feature_batch_size, self.proj_dim),
            device=self.device,
            dtype=self.dtype,
        )
        pointer = 0
        # iterate over feature dimenions, keep a counter of #dim so far, and when prev
        # chunk reaches max_chunk_size, project and accumulate.
        projector_index = 0
        vector_dim = 1
        for _, p in enumerate(features.values()):
            # check the shape of p, if vector then unsqueeze.
            if len(p.shape) <= vector_dim:
                p_flat = p.data.unsqueeze(-1)
            else:
                p_flat = p.data.flatten(start_dim=1)

            feature_dim_size = p_flat.size(1)
            # if current accumulated params exceed max_chunk_size,
            # then stop accumulation.
            if pointer + feature_dim_size > self.max_chunk_size:
                # fill remaining entries with 0
                if pointer != self.dim_per_chunk[projector_index]:
                    msg = "Current number of accumulated #dim does not match \
                    the #feature dim of current chunk."
                    raise ValueError(msg)
                # project and accumulate
                ch_output.add_(
                    self.projector_per_chunk[projector_index].project(
                        self.ch_input[:, :pointer].contiguous(),
                        ensemble_id=ensemble_id,
                    ),
                )
                # reset counter
                pointer = 0
                projector_index += 1

            # continue accumulation
            actual_bs = min(self.ch_input.size(0), p_flat.size(0))
            self.ch_input[:actual_bs, pointer : pointer + feature_dim_size].copy_(
                p_flat,
            )
            pointer += feature_dim_size

        # at the end, we need to project remaining items
        # fill remaining entries with 0
        if pointer != self.dim_per_chunk[projector_index]:
            msg = "Current number of accumulated #dim does not match \
                    the #feature dim of current chunk."
            raise ValueError(msg)

        # project and accumulate
        ch_output[:actual_bs].add_(
            self.projector_per_chunk[projector_index].project(
                self.ch_input[:actual_bs, :pointer].contiguous(),
                ensemble_id=ensemble_id,
            ),
        )

        return ch_output[:actual_bs]

    def project(self, features: Union[dict, Tensor], ensemble_id: int) -> Tensor:
        """Performs the random projection on the feature matrix.

        Args:
            features (Union[dict, Tensor]): A batch of features or a dictionary
                of batch of features.
            ensemble_id (int): A unique ID for this ensemble.

        Returns:
            Tensor: The projected features.
        """
        # allocate zero tensor for output
        ch_output = torch.zeros(
            size=(self.feature_batch_size, self.proj_dim),
            device=self.device,
            dtype=self.dtype,
        )
        # force the input to be Tensor for now
        # TODO: support dict input
        if isinstance(features, dict):
            features = vectorize(features, device=self.device)

        pointer = 0
        for chunk_idx, chunk_dim in enumerate(self.dim_per_chunk):
            ch_output.add_(
                self.projector_per_chunk[chunk_idx].project(
                    features[:, pointer : pointer + chunk_dim].contiguous(),
                    ensemble_id=ensemble_id,
                ),
            )

            pointer += chunk_dim

        return ch_output


class ArnoldiProjector(AbstractProjector):
    """Implementation of Arnoldi projector."""

    def __init__(
        self,
        feature_dim: int,
        proj_dim: int,
        func: Callable,
        x: Tuple,
        argnums: int = 0,
        max_iter: int = 100,
        norm_constant: float = 1.0,
        tol: float = 1e-7,
        mode: str = "rev-fwd",
        regularization: float = 0.0,
        seed: int = 0,
        device: torch.device = "cpu",
    ) -> None:
        """Initializes hyperparameters for ArnoldiProjector.

        Args:
            feature_dim (int): Dimension of the features to be projected.
                Typically, this equals the number of parameters in the model
                (dimension of the gradient vectors).
            proj_dim (int): Dimension after the projection. This corresponds to
                the number of top eigenvalues (top-k eigenvalues) to keep for
                the Hessian approximation.
            func (Callable): A Python function that takes one or more
                arguments. Must return a single-element Tensor. The Hessian
                will be calculated on this function.
            x (Tuple): List of arguments for `func`.
            argnums (int): An integer defaulting to 0. Specifies which argument
                of func to compute inverse Hessian with respect to.
            max_iter (int): An integer defaulting to 100. Specifies the maximum
                iteration to calculate the ihvp through Arnoldi Iteration.
            norm_constant (float): A float defaulting to 1.0. Specifies a
                constant value for the norm of each projection. In some
                situations (e.g. with a large number of parameters) it might be
                advisable to set norm_constant > 1 to avoid dividing projection
                components by a large normalization factor.
            tol (float): A float defaulting to 1e-7. Specifies the break
                condition that decides if the algorithm has converged. If the
                torch.norm of current basis vector is less than tol, then the
                arnoldi_iter algorithm is truncated.
            mode (str): The auto diff mode, which can have one of the following
                values:
                - rev-rev: calculate the Hessian with two reverse-mode
                  auto-diff. It has better compatibility while costing more
                  memory.
                - rev-fwd: calculate the Hessian with the composition of
                  reverse-mode and forward-mode. It's more memory-efficient but
                  may not be supported by some operators.
            regularization (float): A float defaulting to 0.0. Specifies the
                regularization term to be added to the Hessian vector product,
                which is useful for the later inverse calculation if the
                Hessian matrix is singular or ill-conditioned. Specifically,
                the regularization term is `regularization * v`.
            seed (int): Random seed for the generation of the random initial
                vector to build orthonormal basis for the Krylov subspaces.
            device (torch.device): Device to use. Defaults to cpu.
        """
        self.max_iter = max_iter
        self.norm_constant = norm_constant
        self.tol = tol
        # get hvp func
        self.hvp_at_x_func = hvp_at_x(
            func,
            x=(x,),
            argnums=argnums,
            mode=mode,
            regularization=regularization,
        )

        self.feature_dim = feature_dim
        self.proj_dim = proj_dim
        self.seed = seed
        self.device = device

        # init eig vals and vecs as None
        self.eigvals = None
        self.eigvecs = None

    def _arnoldi_iter(  # noqa: PLR6301
        self,
        hvp_func: Callable,
        start_vec: Tensor,
        n_iters: int,
        norm_constant: float,
        tol: float,
        device: str = "cpu",
    ) -> Tuple[Tensor, Tensor]:
        """Applies Arnoldi's algorithm.

        Args:
            hvp_func (Callable): A function that computes hvp.
            start_vec (Tensor): A random normalized vector for initialization.
            n_iters (int): The number of iterations.
            norm_constant (float): The norm normalization for each projection.
            tol (float): A tolerance value used to terminate iteration early.
            device (str): The device to run the algorithm. Defaults to "cpu".

        Returns:
            The result of the Arnoldi Iteration, containing a Hessenberg
            matrix H' approximating the Hessian matrix on its Krylov subspace K,
            and the projections onto K. If H is Hermitian,
            H' will be a tridiagonal matrix (up to numerical errors).
        """
        n_iters = min(start_vec.shape[0] + 1, n_iters)

        proj = []
        appr_mat = torch.zeros((n_iters, n_iters - 1)).to(device)

        start_vec /= torch.norm(start_vec)
        proj.append(start_vec)

        for n in range(n_iters - 1):
            h_vec = hvp_func(proj[n])

            for j, proj_vec in enumerate(proj):
                appr_mat[j][n] = torch.dot(h_vec, proj_vec) / norm_constant**2
                h_vec -= appr_mat[j][n] * proj_vec

            new_norm = torch.norm(h_vec)
            if new_norm < tol:
                appr_mat[n + 1][n] = 0
                proj.append(h_vec)
                appr_mat = appr_mat[: n + 2, : n + 1]
                break

            appr_mat[n + 1][n] = new_norm / norm_constant
            h_vec *= 1.0 / appr_mat[n + 1][n]
            proj.append(h_vec)

        return appr_mat, torch.stack(proj, dim=0)

    def _distill(  # noqa: PLR6301
        self,
        appr_mat: Tensor,
        proj: Tensor,
        top_k: int,
        *,
        force_hermitian: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """Distills result of Arnoldi iteration to top_k eigenvalues and eigenvectors.

        Args:
            appr_mat (Tensor): The first result from arnoldi_iter. This will be
                a Hessenberg matrix H' approximating the Hessian H.
            proj (Tensor): The second result from arnoldi_iter. This will be
                the projection vectors onto the Krylov subspace K of the
                Hessian H.
            top_k (int): Specifies how many eigenvalues and eigenvectors to
                distill.
            force_hermitian (bool): Whether to force the Hessian to Hermitian.
                Defaults to True.

        Returns:
            The distilled eigenvalues and eigenvectors.
        """
        appr_mat = appr_mat[:-1, :]

        if force_hermitian:
            appr_mat = torch.tril(appr_mat, diagonal=1)
            appr_mat = 0.5 * (appr_mat + appr_mat.T)
            eigvals, eigvecs = torch.linalg.eigh(appr_mat)
        else:
            eigvals, eigvecs = torch.linalg.eig(appr_mat)

        idx = torch.argsort(torch.abs(eigvals))
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        reduced_projections = torch.matmul(eigvecs[:, -top_k:].T, proj[:-1])

        return eigvals[-top_k:], reduced_projections

    def get_eigenspace(self) -> None:
        """Compute the approximate eigenspaces of inverse Hessian."""
        # set torch seed
        torch.manual_seed(self.seed)
        # init a random vector
        v0 = torch.rand(self.feature_dim).to(self.device)

        # perform arnoldi iteration
        appr_mat, proj = self._arnoldi_iter(
            self.hvp_at_x_func,
            v0,
            self.max_iter,
            self.norm_constant,
            self.tol,
            self.device,
        )

        # get top k eigenvalues/vectors and cache
        # proj dim is the value of k
        self.eigvals, self.eigvecs = self._distill(appr_mat, proj, self.proj_dim)

        # prevent from negative eigvals
        if self.proj_dim > torch.sum(self.eigvals > self.tol**2):
            # adjust proj_dim
            self.proj_dim = torch.sum(self.eigvals > self.tol**2).item()
            warnings.warn(
                "Encountered many negative eigenvalues and `proj_dim` is greater"
                " than the number of positive eigenvalues. Automatically adjusting"
                " `proj_dim` to the number of positive eigenvalues. Please consider"
                " increasing `regularization` to reduce negative eigenvalues.",
                stacklevel=1,
            )

    def project(self, features: Union[dict, Tensor]) -> Tensor:
        """Performs the random projection on the feature matrix.

        Args:
            features (Union[dict, Tensor]): A batch of features or a dictionary
                of batch of features.

        Returns:
            Tensor: The projected features.
        """
        # transform to tensors
        if isinstance(features, dict):
            features = vectorize(features, device=self.device)
        # have not compute the eigen space yet
        if self.eigvals is None or self.eigvecs is None:
            self.get_eigenspace()

        return features @ self.eigvecs.T * (1.0 / torch.sqrt(self.eigvals.unsqueeze(0)))

    def free_memory(self) -> None:
        """A no-op method."""


def make_random_projector(
    param_shape_list: List,
    feature_batch_size: int,
    proj_dim: int,
    proj_max_batch_size: int,
    device: str,
    proj_seed: int = 0,
    *,
    use_half_precision: bool = True,
) -> Tensor:
    """Initialize random projector by the info of feature about to be projected.

    Args:
        param_shape_list (List): A list of numbers indicating the total number of
            features to be projected. A typical example is a list of total parameter
            size of each module in a torch.nn.Module model. Total parameter size
            of each module equals to feature_batch_size * param_size of that module.
        feature_batch_size (int): The batch size of each tensor in the feature
            about to be projected. The typical type of feature are gradients of
            torch.nn.Module model but can be restricted to this.
        proj_dim (int): Dimension of the projected feature.
        proj_max_batch_size (int): The maximum batch size used by fast_jl if the
            CudaProjector is used. Must be a multiple of 8. The maximum
            batch size is 32 for A100 GPUs, 16 for V100 GPUs, 40 for H100 GPUs.
        device (str): "cuda" or "cpu".
        proj_seed (int): Random seed used by the projector. Defaults to 0.
        use_half_precision (bool): If True, torch.float16 will be used for all
            computations and arrays will be stored in torch.float16.

    Returns:
        The initialized projector object
        (CudaProjector, ChunkedCudaProjector, or BasicProjector).
    """
    using_cuda_projector = False
    dtype = torch.float16 if use_half_precision else torch.float32
    # the total feature dim
    feature_dim = sum(param_shape_list)
    if device == "cpu":
        projector = BasicProjector
        # Sampling from bernoulli distribution is not supported for
        # dtype float16 on CPU; playing it safe here by defaulting to
        # normal projection, rather than rademacher.
        proj_type = ProjectionType.normal
    else:
        try:
            import fast_jl

            test_feature = torch.ones(1, feature_dim).cuda()
            num_sms = torch.cuda.get_device_properties(
                "cuda",
            ).multi_processor_count
            fast_jl.project_rademacher_8(
                test_feature,
                proj_dim,
                0,
                num_sms,
            )
            projector = CudaProjector
            using_cuda_projector = True
            proj_type = ProjectionType.rademacher

        except (ImportError, RuntimeError, AttributeError, ModuleNotFoundError):
            projector = BasicProjector
            proj_type = ProjectionType.normal

    if using_cuda_projector:
        # TODO: make this support dict input
        # currently, only tensor input will be considered
        max_chunk_size, param_chunk_sizes = get_parameter_chunk_sizes(
            param_shape_list,
            proj_max_batch_size,
        )
        if len(param_chunk_sizes) > 1:  # we have to use the ChunkedCudaProjector
            rng = np.random.default_rng(proj_seed)
            # different seeds for each chunk
            seeds = rng.integers(
                low=0,
                high=500,
                size=len(param_chunk_sizes),
            )
            projector_per_chunk = [
                projector(
                    feature_dim=chunk_size,
                    proj_dim=proj_dim,
                    seed=seeds[i],
                    proj_type=proj_type,
                    max_batch_size=proj_max_batch_size,
                    device=device,
                )
                for i, chunk_size in enumerate(param_chunk_sizes)
            ]
            return ChunkedCudaProjector(
                projector_per_chunk,
                max_chunk_size,
                param_chunk_sizes,
                feature_batch_size,
                proj_max_batch_size,
                device,
                dtype,
            )

    if projector == CudaProjector:
        assigned_projector = projector(
            feature_dim=feature_dim,
            proj_dim=proj_dim,
            seed=proj_seed,
            proj_type=proj_type,
            max_batch_size=proj_max_batch_size,
            device=device,
        )
    elif projector == BasicProjector:
        assigned_projector = projector(
            feature_dim=feature_dim,
            proj_dim=proj_dim,
            seed=proj_seed,
            proj_type=proj_type,
            dtype=dtype,
            device=device,
        )

    return assigned_projector


def arnoldi_project(
    feature_dim: int,
    func: Callable,
    x: List,
    argnums: int = 0,
    proj_dim: int = 100,
    max_iter: int = 100,
    norm_constant: float = 1.0,
    tol: float = 1e-7,
    mode: str = "rev-fwd",
    regularization: float = 0.0,
    seed: int = 0,
    device: torch.device = "cpu",
) -> Callable:
    """Apply Arnoldi algorithm to approximate iHVP.

    Args:
        feature_dim (int): Dimension of the features to be projected. Typically,
            this equals the number of parameters in the model (dimension of the
            gradient vectors).
        func (Callable): A Python function that takes one or more arguments.
            Must return a single-element Tensor. The Hessian will be calculated
            on this function. The positional arguments to func must all be
            Tensors.
        x (List): List of arguments for `func`.
        argnums (int): An integer defaulting to 0. Specifies which argument of
            func to compute Hessian with respect to.
        proj_dim (int): Dimension after the projection. This corresponds to the
            number of top eigenvalues (top-k eigenvalues) to keep for the
            Hessian approximation.
        max_iter (int): An integer defaulting to 100. Specifies the maximum
            iteration to calculate the ihvp through Arnoldi Iteration.
        norm_constant (float): A float defaulting to 1.0. Specifies a constant
            value for the norm of each projection. In some situations (e.g.
            with a large number of parameters) it might be advisable to set
            norm_constant > 1 to avoid dividing projection components by a
            large normalization factor.
        tol (float): A float defaulting to 1e-7. Specifies the break condition
            that decides if the algorithm has converged. If the torch.norm of
            the current basis vector is less than tol, then the algorithm is
            truncated.
        mode (str): The auto diff mode, which can have one of the following
            values:
            - rev-rev: calculate the Hessian with two reverse-mode auto-diff.
              It has better compatibility while costing more memory.
            - rev-fwd: calculate the Hessian with the composition of
              reverse-mode and forward-mode. It's more memory-efficient but may
              not be supported by some operators.
        regularization (float): A float defaulting to 0.0. Specifies the
            regularization term to be added to the Hessian vector product,
            which is useful for the later inverse calculation if the Hessian
            matrix is singular or ill-conditioned. Specifically, the
            regularization term is `regularization * v`.
        seed (int): Random seed used by the projector. Defaults to 0.
        device (torch.device): "cuda" or "cpu". Defaults to "cpu".

    Returns:
        A function that applies Arnoldi algorithm on input feature.
    """
    # init arnoldi projector
    projector = ArnoldiProjector(
        feature_dim,
        proj_dim,
        func,
        x,
        argnums,
        max_iter,
        norm_constant,
        tol,
        mode,
        regularization,
        seed,
        device,
    )

    def _arnoldi_project_func(
        feature: Union[Dict[str, Tensor], Tensor],
    ) -> Tensor:
        """The projection function using constructed projector.

        Args:
            feature (Union[Dict[str, Tensor], Tensor]): The feature needs to be
                projected. This can simple be a tensor with size [feature_batch_size,
                feature_dim]. Or typically, if the this is gradient of some
                torch.nn.Module models, it will have the structure similar to the
                result of model.named_parameters().

        Returns:
            The projected result of feature, which is a tensor with size
                [feature_batch_size, proj_dim].
        """
        return projector.project(feature)

    return _arnoldi_project_func


def random_project(
    feature: Union[Dict[str, Tensor], Tensor],
    feature_batch_size: int,
    proj_dim: int,
    proj_max_batch_size: int,
    device: str,
    proj_seed: int = 0,
    *,
    use_half_precision: bool = True,
) -> Callable:
    """Randomly projects the features to a smaller dimension.

    Args:
        feature (Union[Dict[str, Tensor], Tensor]): The feature needs to be
            projected. This can simple be a tensor with size [feature_batch_size,
            feature_dim]. Or typically, if the this is gradient of some
            torch.nn.Module models, it will have the structure similar to the
            result of model.named_parameters().
        feature_batch_size (int): The batch size of each tensor in the feature
            about to be projected. The typical type of feature are gradients of
            torch.nn.Module model but can restricted to this.
        proj_dim (int): Dimension of the projected feature.
        proj_max_batch_size (int): The maximum batch size used by fast_jl if the
            CudaProjector is used. Must be a multiple of 8. The maximum
            batch size is 32 for A100 GPUs, 16 for V100 GPUs, 40 for H100 GPUs.
        device (str): "cuda" or "cpu".
        proj_seed (int): Random seed used by the projector. Defaults to 0.
        use_half_precision (bool): If True, torch.float16 will be used for all
            computations and arrays will be stored in torch.float16.

    Returns:
        A function that takes projects feature to a smaller dimension.
    """
    # check the type of feature
    if isinstance(feature, dict):
        param_shape_list = [
            feature[param_name].numel() // feature_batch_size for param_name in feature
        ]
    else:
        param_shape_list = [feature.numel() // feature_batch_size]

    projector = make_random_projector(
        param_shape_list=param_shape_list,
        feature_batch_size=feature_batch_size,
        proj_dim=proj_dim,
        proj_max_batch_size=proj_max_batch_size,
        device=device,
        proj_seed=proj_seed,
        use_half_precision=use_half_precision,
    )

    def _random_project_func(
        feature: Union[Dict[str, Tensor], Tensor],
        ensemble_id: int = 0,
    ) -> Tensor:
        """The projection function using constructed projector.

        Args:
            feature (Union[Dict[str, Tensor], Tensor]): The feature needs to be
                projected. This can simple be a tensor with size [feature_batch_size,
                feature_dim]. Or typically, if the this is gradient of some
                torch.nn.Module models, it will have the structure similar to the
                result of model.named_parameters().
            ensemble_id (int): A unique ID for this ensemble. Defaults to 0.

        Returns:
            The projected result of feature, which is a tensor with size
                [feature_batch_size, proj_dim].
        """
        return projector.project(feature, ensemble_id)

    return _random_project_func
