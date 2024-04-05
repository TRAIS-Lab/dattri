"""dattri.func.random_projection for some random projector."""
# Code adapted from https://github.com/MadryLab/trak/blob/main/trak/projectors.py
# Code adapted from https://github.com/MadryLab/trak/blob/main/trak/utils.py

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor

ch = torch

def vectorize(g: Dict[str, torch.Tensor], arr: Optional[torch.Tensor] = None,
              device: Optional[str] = "cuda") -> Tensor:
    """Vectorize gradient result (g) into arr.

    This function takes a dictionary of gradient and returns a flattened tensor
    of shape [batch_size, num_params].

    Args:
        g (dict of Tensors): A dictionary containing gradient tensors to be vectorized.
        arr (Tensor, optional): An optional pre-allocated tensor to store the
                                vectorized gradients. If provided, it must have the
                                shape `[batch_size, num_params]`, where `num_params`
                                is the total number of scalar parameters in all
                                gradient tensors combined. If `None`, a new tensor
                                is created. Defaults to None.
        device (str, optional): "cuda" or "cpu". Defaults to "cuda".

    Returns:
        Tensor: A 2D tensor of shape `[batch_size, num_params]`,
                where each row contains all the vectorized gradients
                for a single batch element.

    Raises:
        ValueError: Parameter size in g doesn't match batch size.
    """
    if arr is None:
        g_elt = g[next(iter(g.keys()))[0]]
        batch_size = g_elt.shape[0]
        num_params = 0
        for param in g.values():
            if  param.shape[0] != batch_size:
                msg = "Parameter row num doesn't match batch size."
                raise ValueError(msg)
            num_params += int(param.numel() / batch_size)
        arr = ch.empty(size=(batch_size, num_params), dtype=g_elt.dtype, device=device)

    pointer = 0
    vector_dim = 1
    for param in g.values():
        if len(param.shape) <=  vector_dim:
            num_param = 1
            p = param.data.reshape(-1, 1)
        else:
            num_param = param[0].numel()
            p = param.flatten(start_dim=1).data

        arr[:, pointer : pointer + num_param] = p.to(device)
        pointer += num_param

    return arr


class ProjectionType(str, Enum):
    """Projection type used for projectors."""
    normal: str = "normal"
    rademacher: str = "rademacher"


class AbstractProjector(ABC):
    """Base Class for projectors."""

    @abstractmethod
    def __init__(
        self,
        grad_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: Union[str, ProjectionType],
        device: Union[str, torch.device],
    ) -> None:
        """Initializes hyperparameters for the projection.

        Args:
            grad_dim (int):
                Number of parameters in the model (dimension of the gradient
                vectors).
            proj_dim (int):
                Dimension after the projection.
            seed (int):
                Random seed for the generation of the sketching (projection)
                matrix.
            proj_type (Union[str, ProjectionType]):
                The random projection (JL transform) guarantees that distances
                will be approximately preserved for a variety of choices of the
                random matrix (see e.g. https://arxiv.org/abs/1411.2404). Here,
                we provide an implementation for matrices with iid Gaussian
                entries and iid Rademacher entries.
            device (Union[str, torch.device]):
                CUDA device to use.

        """
        self.grad_dim = grad_dim
        self.proj_dim = proj_dim
        self.seed = seed
        self.proj_type = proj_type
        self.device = device

    @abstractmethod
    def project(self, grads: Tensor, model_id: int) -> Tensor:
        """Performs the random projection on gradient matrix.

        This function will take grads and a model ID, which allows us
        to generate different projection matrices, and output the projected
        matrix.

        Args:
            grads (Tensor): A batch of gradients to be projected.
            model_id (int): A unique ID for a checkpoint.

        Returns:
            Tensor: The projected gradients.
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
        grad_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: Union[str, ProjectionType],
        device: torch.device,
        block_size: int = 100,
        dtype: torch.dtype = ch.float32,
        model_id: int = 0,
        ) -> None:
        """Initializes hyperparameters for BasicProjector.

        Args:
            grad_dim (int):
                Number of parameters in the model (dimension of the gradient
                vectors).
            proj_dim (int):
                Dimension after the projection.
            seed (int):
                Random seed for the generation of the sketching (projection)
                matrix.
            proj_type (Union[str, ProjectionType]):
                The random projection (JL transform) guarantees that distances
                will be approximately preserved for a variety of choices of the
                random matrix (see e.g. https://arxiv.org/abs/1411.2404). Here,
                we provide an implementation for matrices with iid Gaussian
                entries and iid Rademacher entries.
            device (Union[str, torch.device]):
                CUDA device to use.
            block_size (int):
                Maximum number of projection dimension allowed. Thus,
                min(block_size, proj_dim) will be used as the true projection
                dimension.
            dtype (torch.dtype): The dtype of the projected matrix.
            model_id (int): A unique ID for a checkpoint.

        """
        super().__init__(grad_dim, proj_dim, seed, proj_type, device)

        self.block_size = min(self.proj_dim, block_size)
        self.num_blocks = math.ceil(self.proj_dim / self.block_size)
        self.dtype = dtype
        self.proj_type = proj_type
        self.model_id = model_id

        self.proj_matrix = ch.empty(
            self.grad_dim, self.block_size, dtype=self.dtype, device=self.device,
        )

        self.proj_matrix_available = True

        self.generator = ch.Generator(device=self.device)

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
        self.jl_size = self.grad_dim * self.block_size

        for i in range(self.num_blocks):
            s = self.seed + int(1e3) * i + int(1e5) * self.model_id
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
            self.proj_matrix = ch.empty(
                self.grad_dim, self.block_size, dtype=self.dtype, device=self.device,
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

    def project(self, grads: Tensor, model_id: int) -> Tensor:
        """Performs the random projection on gradient matrix.

        Args:
            grads (Tensor): A batch of gradients to be projected.
            model_id (int): A unique ID for a checkpoint.

        Returns:
            Tensor: The projected gradients.
        """
        if isinstance(grads, dict):
            grads = vectorize(grads, device=self.device)
        grads = grads.to(dtype=self.dtype)
        sketch = ch.zeros(
            size=(grads.size(0), self.proj_dim), dtype=self.dtype, device=self.device,
        )

        if model_id != self.model_id:
            self.model_id = model_id
            self.get_generator_states()  # regenerate random seeds for new model_id
            if self.num_blocks == 1:
                self.generate_sketch_matrix(self.generator_states[0])

        if self.num_blocks == 1:
            ch.matmul(grads.data, self.proj_matrix, out=sketch)
        else:
            for ind in range(self.num_blocks):
                self.generate_sketch_matrix(self.generator_states[ind])

                st = ind * self.block_size
                ed = min((ind + 1) * self.block_size, self.proj_dim)
                sketch[:, st:ed] = (
                    grads.type(self.dtype) @ self.proj_matrix[:, : (ed - st)]
                )
        return sketch.type(grads.dtype)



class CudaProjector(AbstractProjector):
    """Projector implemented using CUDA.

    A performant implementation of the projection
    for CUDA with compute capability >= 7.0.
    """

    def __init__(
        self,
        grad_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: ProjectionType,
        device: str,
        max_batch_size: int,
        ) -> None:
        """Initializes hyperparameters for CudaProjector.

        Args:
            grad_dim (int): Number of parameters.
            proj_dim (int): Dimension we project *to* during the projection step
            seed (int): Random seed.
            proj_type (ProjectionType): Type of randomness to use for
                                        projection matrix (rademacher or normal).
            device: CUDA device to use.
            max_batch_size (int): Explicitly constraints the batch size of
                the CudaProjector is going to use for projection.
                Set this if you get a 'The batch size of the CudaProjector is
                too large for your GPU' error. Must be either 8, 16, or 32.

        Raises:
            ValueError: When attempting to use this on a non-CUDA device
            msg: When fast_jl is not installed

        """
        super().__init__(grad_dim, proj_dim, seed, proj_type, device)
        self.max_batch_size = max_batch_size

        if isinstance(device, str):
            device = ch.device(device)

        if device.type != "cuda":
            err = "CudaProjector only works on a CUDA device; \
            Either switch to a CUDA device, or use the BasicProjector"
            raise ValueError(err)

        self.num_sms = ch.cuda.get_device_properties(device.index).multi_processor_count

        try:
            import fast_jl

            # test run to catch at init time if projection goes through
            fast_jl.project_rademacher_8(
                ch.zeros(8, 1_000, device="cuda"), 512, 0, self.num_sms,
            )
        except ImportError:
            msg = "You should make sure to install the CUDA projector \
            (the fast_jl library)."
            raise msg from ModuleNotFoundError

    def project(
        self,
        grads: Union[dict, Tensor],
        model_id: int,
        ) -> Tensor:
        """Performs the random projection on gradient matrix.

        Args:
            grads (Union[dict, Tensor]): A batch of gradients or a dictionary
                of batch of gradients.
            model_id (int): A unique ID for a checkpoint.

        Raises:
            msg: The batch size of the CudaProjector need to be reduced.
            RuntimeError: Too many resources requested for launch CUDA.

        Returns:
            Tensor: The projected gradients.
        """
        if isinstance(grads, dict):
            grads = vectorize(grads, device=self.device)
        batch_size = grads.shape[0]

        effective_batch_size = 32
        min_proj_batch_size = 8
        if batch_size <= min_proj_batch_size:
            effective_batch_size = min_proj_batch_size
        elif batch_size <= min_proj_batch_size * 2:
            effective_batch_size = min_proj_batch_size * 2

        effective_batch_size = min(self.max_batch_size, effective_batch_size)

        function_name = f"project_{self.proj_type.value}_{effective_batch_size}"
        import fast_jl

        fn = getattr(fast_jl, function_name)

        try:
            result = fn(
                grads, self.proj_dim, self.seed + int(1e4) * model_id, self.num_sms,
            )
        except RuntimeError as e:
            if "CUDA error: too many resources requested for launch" in str(e):
                # provide a more helpful error message
                msg = "The batch size of the CudaProjector is too large for your GPU. \
                    Reduce it by using the proj_max_batch_size argument.\
                    \nOriginal error:"
                raise msg from RuntimeError
            raise e from None

        return result

    def free_memory(self) -> None:
        """A no-op method."""


class ChunkedCudaProjector:
    """Chunked CudaProjector implemented using CUDA.

    This projector is used when (# params) * (# proj_dim) is too large.
    """
    def __init__(
        self,
        projector_per_chunk: list,
        max_chunk_size: int,
        params_per_chunk: list,
        proj_max_batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Initializes hyperparameters for ChunkedCudaProjector.

        Args:
            projector_per_chunk (list): A list of projectors. Specifying
                the projector used by each chunk.
            max_chunk_size (int): The maximum size of each chunk.
            params_per_chunk (list): The number of parameters per chunk.
            proj_max_batch_size (int): The maximum batch size or each projector.
            device (torch.device): Device to use. Will be "cuda" or "cpu".
            dtype (torch.dtype): The dtype of the projected matrix.
        """
        self.projector_per_chunk = projector_per_chunk
        self.proj_dim = self.projector_per_chunk[0].proj_dim
        self.proj_type = self.projector_per_chunk[0].proj_type
        self.params_per_chunk = params_per_chunk

        self.max_chunk_size = max_chunk_size
        self.proj_max_batch_size = proj_max_batch_size
        self.device = device
        self.dtype = dtype
        self.input_allocated = False

    def allocate_input(self) -> None:
        """Allocate zero tensor for input."""
        if self.input_allocated:
            return

        self.ch_input = ch.zeros(
            size=(self.proj_max_batch_size, self.max_chunk_size),
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

    def project(self, grads: Tensor, model_id: int) -> Tensor:
        """Performs the random projection on gradient matrix.

        Args:
            grads (Tensor): A batch of gradients to be projected.
            model_id (int): A unique ID for a checkpoint.

        Raises:
            ValueError: The number of accumulated params does not match
                params_per_chunk.
            ValueError: The number of accumulated params does not match
                params_per_chunk.

        Returns:
            Tensor: The projected gradients.
        """
        self.allocate_input()
        ch_output = ch.zeros(
            size=(self.proj_max_batch_size, self.proj_dim), device=self.device,
            dtype=self.dtype,
        )
        pointer = 0
        # iterate over params, keep a counter of params so far, and when prev
        # chunk reaches max_chunk_size, project and accumulate.
        projector_index = 0
        vector_dim = 1
        for _, p in enumerate(grads.values()):
            # check the shape of p, if vector then unsqueeze.
            if len(p.shape) <= vector_dim:
                p_flat = p.data.unsqueeze(-1)
            else:
                p_flat = p.data.flatten(start_dim=1)

            param_size = p_flat.size(1)
            # if current accumulated params exceed max_chunk_size,
            # then stop accumulation.
            if pointer + param_size > self.max_chunk_size:
                # fill remaining entries with 0
                if pointer != self.params_per_chunk[projector_index]:
                    msg = "Current number of accumulated params does not match \
                    the param number of current chunk."
                    raise ValueError(msg)
                # project and accumulate
                ch_output.add_(
                    self.projector_per_chunk[projector_index].project(
                        self.ch_input[:, :pointer].contiguous(),
                        model_id=model_id,
                    ),
                )
                # reset counter
                pointer = 0
                projector_index += 1

            # continue accumulation
            actual_bs = min(self.ch_input.size(0), p_flat.size(0))
            self.ch_input[:actual_bs, pointer : pointer + param_size].copy_(p_flat)
            pointer += param_size

        # at the end, we need to project remaining items
        # fill remaining entries with 0
        if pointer != self.params_per_chunk[projector_index]:
            msg = "Current number of accumulated params does not match \
                    the param number of current chunk."
            raise ValueError(msg)

        # project and accumulate
        ch_output[:actual_bs].add_(
            self.projector_per_chunk[projector_index].project(
                self.ch_input[:actual_bs, :pointer].contiguous(),
                model_id=model_id,
            ),
        )

        return ch_output[:actual_bs]
