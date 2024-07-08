"""Defines the task abstractions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.func import grad, vmap

from dattri.func.utils import flatten_func, flatten_params


class AttributionTask:
    """The abstraction of the abstraction task."""

    def __init__(
        self,
        target_func: Callable,
        model: nn.Module,
        checkpoints: Union[
            str,
            List[str],
            List[Dict[str, torch.Tensor]],
            Dict[str, torch.Tensor],
        ],
    ) -> None:
        """Initialize the AttributionTask.

        Args:
            target_func (Callable): The target function to be attributed.
                The function can be quite flexible in terms of what is calculate,
                but it should take the parameters and the data as input. Other than
                that, the forwarding of model should be in `torch.func` style.
                A typical example is as follows:
                ```python
                def f(params, data):
                    image, label = data
                    loss = nn.CrossEntropyLoss()
                    yhat = torch.func.functional_call(model, params, image)
                    return loss(yhat, label)
                ```.
                This examples calculates the CE loss of the model on the data.
            model (nn.Module): The model that the target function is based on.
                To be more specific, the model is the `model` used in the target
                function. Since only the computation graph of the model will be
                used, so it is allowed that this model is not loaded with a trained
                parameters.
            checkpoints:
                (Union[str, List[str], List[Dict[str, torch.Tensor]],
                Dict[str, torch.Tensor]]): The checkpoints
                of the model, both dictionary of the state_dict and the path to
                the checkpoint are supported. If ensemble is needed, a list of
                checkpoint is also supported.

        """
        self.model = model
        self.target_func = flatten_func(self.model)(target_func)

        if not isinstance(checkpoints, list):
            self.checkpoints = [checkpoints]
        else:
            self.checkpoints = checkpoints

        # current_checkpoint_idx is used to state
        # which checkpoint is currently loaded.
        self.current_checkpoint_idx = None

        # TODO: Make this more general, that is allow customized kwargs.
        self.grad_func = vmap(grad(self.target_func), in_dims=(None, 1))

    def _load_checkpoints(self, index: int) -> None:
        """This method load the checkpoint at the specified index.

        Args:
            index (int): The index of the checkpoint to be loaded.
        """
        if self.current_checkpoint_idx is None or self.current_checkpoint_idx != index:
            if isinstance(self.checkpoints[index], str):
                self.model.load_state_dict(torch.load(self.checkpoints[index]))
            else:
                self.model.load_state_dict(self.checkpoints[index])
            self.current_checkpoint_idx = index
            self.named_parameters = {
                k: p for k, p in self.model.named_parameters() if p.requires_grad
            }

    @staticmethod
    def _genearte_param_layer_map(
        named_parameters: Dict[str, torch.Tensor],
    ) -> List[int]:
        """This function generate the param_layer_map automatically.

        Args:
            named_parameters (Dict[str, torch.Tensor]):
                The named parameters of the model.

        Returns:
            List[int]: The map from the parameter to the layer.
        """
        named_parameters_keys = named_parameters.keys()
        param_layer_map = []

        current_layer = None
        current_index = -1
        for key in named_parameters_keys:
            layer_name = ".".join(key.split(".")[:-1])
            if layer_name != current_layer:
                current_index += 1
                current_layer = layer_name
            param_layer_map.append(current_index)

        return param_layer_map

    def get_grad_target_func(
        self,
        layer_name: Optional[Union[str, List[str]]] = None,
    ) -> Callable:
        """Return a function that computes the gradient of the target function.

        TODO: support partial parameter gradient.

        Args:
            layer_name (Optional[Union[str, List[str]]]): This has not been supported.

        Returns:
            Callable: The function that computes the gradient of the target function.

        Raises:
            NotImplementedError: If layer_name is not None.
        """
        if layer_name is not None:
            error_msg = "layer_name has not been implemented yet."
            raise NotImplementedError(error_msg)
        return self.grad_func

    def get_target_func(
        self,
        layer_name: Optional[Union[str, List[str]]] = None,
    ) -> Callable:
        """Return a function that computes the gradient of the target function.

        TODO: support partial parameter gradient.

        Args:
            layer_name (Optional[Union[str, List[str]]]): This has not been supported.

        Returns:
            Callable: The target function itself.

        Raises:
            NotImplementedError: If layer_name is not None.
        """
        if layer_name is not None:
            error_msg = "layer_name has not been implemented yet."
            raise NotImplementedError(error_msg)
        return self.target_func

    def get_param(
        self,
        index: int = 0,
        layer_name: Optional[Union[str, List[str]]] = None,
        layer_split: Optional[bool] = False,
        param_layer_map: Optional[List[int]] = None,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Optional[List[int]]]:
        """Return the flattened parameter of the model.

        Args:
            index (int): The index of the checkpoint to be loaded.
            layer_name (Optional[Union[str, List[str]]]): The name of the layer to be
                extracted. If None, all the parameters will be returned. This should be
                a string or a list of strings if multiple layers are needed. The name
                of layer should follow the key of model.named_parameters().
            layer_split (Optional[bool]): If True, the parameters will be split
                into different layers and returned as a list of parameter tensors.
            param_layer_map (Optional[List[int]]): The map from the parameter
                to the layer. If None, the map will be generated automatically. Normally
                this should not be stated explicitly by the user, if needed it should
                be the same length as parameters tuple. For example,
                for a two layer model, params = (0.weights1, 0.bias, 1.weights, 1.bias),
                param_layer_map should be [0, 0, 1, 1],resulting in two layers
                as expected.

        Returns:
            Tuple[Union[torch.Tensor, List[torch.Tensor]], Optional[List[int]]]: The
                flattened parameters of the model and the layer map if layer_split
                is True. Flattened parameters will be a 1-dim tensor.

        Raises:
            ValueError: If the length of param_layer_map is not the same as the length
                of named_parameters
        """
        self._load_checkpoints(index)

        if layer_name:
            named_parameters = {
                k: self.named_parameters[k]
                for k in layer_name
                if k in self.named_parameters
            }
        else:
            named_parameters = self.named_parameters

        if layer_split:
            if param_layer_map:
                if len(param_layer_map) == len(
                    named_parameters,
                ):
                    error_msg = (
                        "param_layer_map must have the same len as named_parameters."
                    )
                    raise ValueError(error_msg)
                return tuple(
                    [param.flatten() for param in named_parameters.values()],
                ), param_layer_map
            return tuple(
                [param.flatten() for param in named_parameters.values()],
            ), self._genearte_param_layer_map(named_parameters)
        return flatten_params(named_parameters), None

    def register_forward_hook(  # noqa:PLR6301
        self,
        layer_name: Union[str, List[str]],  # noqa: ARG002
    ) -> Tuple[torch.Tensors, ...]:
        """Register forward hook to specified layer_name.

        Args:
            layer_name (Union[str, List[str]]): The name of the layer to be registered.

        Raises:
            NotImplementedError: This method has not been implemented yet.
        """
        error_msg = "This method has not been implemented yet."
        raise NotImplementedError(error_msg)
