"""dattri.func.random_projection for some random projector."""
# Code adapted from https://github.com/MadryLab/trak/blob/main/trak/projectors.py
# Code adapted from https://github.com/MadryLab/trak/blob/main/trak/utils.py

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Union

import torch
from torch import Tensor

ch = torch

def vectorize(g: Dict[str, torch.Tensor], arr: Optional[torch.Tensor] = None,
              device: str = "cuda") -> Tensor:
    """This function will vectorize gradient result (g) into arr.

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
                raise ValueError("Parameter row num doesn't match batch size.")
            num_params += int(param.numel() / batch_size)
        arr = ch.empty(size=(batch_size, num_params), dtype=g_elt.dtype, device=device)

    pointer = 0
    for param in g.values():
        if len(param.shape) < 2:
            num_param = 1
            p = param.data.reshape(-1, 1)
        else:
            num_param = param[0].numel()
            p = param.flatten(start_dim=1).data

        arr[:, pointer : pointer + num_param] = p.to(device)
        pointer += num_param

    return arr


class ProjectionType(str, Enum):
    normal: str = "normal"
    rademacher: str = "rademacher"


class AbstractProjector(ABC):
    """Implementations of the Projector class must implement the
    :meth:`AbstractProjector.project` method, which takes in model gradients and
    returns
    """

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
                number of parameters in the model (dimension of the gradient
                vectors)
            proj_dim (int):
                dimension after the projection
            seed (int):
                random seed for the generation of the sketching (projection)
                matrix
            proj_type (Union[str, ProjectionType]):
                the random projection (JL transform) guearantees that distances
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
        """Performs the random projection. Model ID is included
        so that we generate different projection matrices for every
        model ID.

        Args:
            grads (Tensor): a batch of gradients to be projected
            model_id (int): a unique ID for a checkpoint

        Returns:
            Tensor: the projected gradients
        """

    def free_memory(self):
        """Frees up memory used by the projector."""


class CudaProjector(AbstractProjector):
    """A performant implementation of the projection for CUDA with compute
    capability >= 7.0.
    """

    def __init__(
        self,
        grad_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: ProjectionType,
        device,
        max_batch_size: int,
        *args,
        **kwargs,
    ) -> None:
        """Args:
            grad_dim (int):
                Number of parameters
            proj_dim (int):
                Dimension we project *to* during the projection step
            seed (int):
                Random seed
            proj_type (ProjectionType):
                Type of randomness to use for projection matrix (rademacher or normal)
            device:
                CUDA device
            max_batch_size (int):
                Explicitly constraints the batch size the CudaProjector is going
                to use for projection. Set this if you get a 'The batch size of
                the CudaProjector is too large for your GPU' error. Must be
                either 8, 16, or 32.

        Raises:
            ValueError:
                When attempting to use this on a non-CUDA device
            ModuleNotFoundError:
                When fast_jl is not installed

        """
        super().__init__(grad_dim, proj_dim, seed, proj_type, device)
        self.max_batch_size = max_batch_size

        if isinstance(device, str):
            device = ch.device(device)

        if device.type != "cuda":
            err = "CudaProjector only works on a CUDA device; Either switch to a CUDA device, or use the BasicProjector"
            raise ValueError(err)

        self.num_sms = ch.cuda.get_device_properties(device.index).multi_processor_count

        try:
            import fast_jl

            # test run to catch at init time if projection goes through
            fast_jl.project_rademacher_8(
                ch.zeros(8, 1_000, device="cuda"), 512, 0, self.num_sms,
            )
        except ImportError:
            err = "You should make sure to install the CUDA projector for traker (called fast_jl).\
                  See the installation FAQs for more details."
            raise ModuleNotFoundError(err)

    def project(
        self,
        grads: Union[dict, Tensor],
        model_id: int,
        is_grads_dict: bool = True,
    ) -> Tensor:
        if is_grads_dict:
            grads = vectorize(grads, device=self.device)
        batch_size = grads.shape[0]

        effective_batch_size = 32
        if batch_size <= 8:
            effective_batch_size = 8
        elif batch_size <= 16:
            effective_batch_size = 16

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
                raise RuntimeError(
                    (
                        "The batch size of the CudaProjector is too large for your GPU. "
                        "Reduce it by using the proj_max_batch_size argument of the TRAKer.\nOriginal error:"
                    ),
                )
            else:
                raise e

        return result

    def free_memory(self):
        """A no-op method."""


class ChunkedCudaProjector:
    def __init__(
        self,
        projector_per_chunk: list,
        max_chunk_size: int,
        params_per_chunk: list,
        feat_bs: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.projector_per_chunk = projector_per_chunk
        self.proj_dim = self.projector_per_chunk[0].proj_dim
        self.proj_type = self.projector_per_chunk[0].proj_type
        self.params_per_chunk = params_per_chunk

        self.max_chunk_size = max_chunk_size
        self.feat_bs = feat_bs
        self.device = device
        self.dtype = dtype
        self.input_allocated = False

    def allocate_input(self):
        if self.input_allocated:
            return

        self.ch_input = ch.zeros(
            size=(self.feat_bs, self.max_chunk_size),
            device=self.device,
            dtype=self.dtype,
        )

        self.input_allocated = True

    def free_memory(self):
        if not self.input_allocated:
            return

        del self.ch_input
        self.input_allocated = False

    def project(self, grads, model_id):
        self.allocate_input()
        ch_output = ch.zeros(
            size=(self.feat_bs, self.proj_dim), device=self.device, dtype=self.dtype,
        )
        pointer = 0
        # iterate over params, keep a counter of params so far, and when prev
        # chunk reaches max_chunk_size, project and accumulate
        projector_index = 0
        for i, p in enumerate(grads.values()):
            if len(p.shape) < 2:
                p_flat = p.data.unsqueeze(-1)
            else:
                p_flat = p.data.flatten(start_dim=1)

            param_size = p_flat.size(1)
            if pointer + param_size > self.max_chunk_size:
                # fill remaining entries with 0
                assert pointer == self.params_per_chunk[projector_index]
                # project and accumulate
                ch_output.add_(
                    self.projector_per_chunk[projector_index].project(
                        self.ch_input[:, :pointer].contiguous(),
                        model_id=model_id,
                        is_grads_dict=False,
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
        assert pointer == self.params_per_chunk[projector_index]
        # project and accumulate
        ch_output[:actual_bs].add_(
            self.projector_per_chunk[projector_index].project(
                self.ch_input[:actual_bs, :pointer].contiguous(),
                model_id=model_id,
                is_grads_dict=False,
            ),
        )

        return ch_output[:actual_bs]
