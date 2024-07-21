"""Defines the task abstractions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Dict, List, Optional, Tuple, Union

from pathlib import PosixPath

import torch
from torch import nn
from torch.func import grad, vmap

from dattri.func.utils import flatten_func, flatten_params


class AttributionTask:
    """The abstraction of the attribution task information."""

    def __init__(
        self,
        loss_func: Callable,
        model: nn.Module,
        checkpoints: Union[
            str,
            List[str],
            List[Dict[str, torch.Tensor]],
            Dict[str, torch.Tensor],
        ],
        target_func: Optional[Callable] = None,
    ) -> None:
        """Initialize the AttributionTask.

        Args:
            loss_func (Callable): The loss function of the model training.
                The function can be quite flexible in terms of what is calculated,
                but it should take the parameters and the data as input. Other than
                that, the forwarding of model should be in `torch.func` style.
                It will be used as target function to be attributed if no other
                target function provided
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
            target_func (Callable): The target function to be attributed.
                This input is optional, if not provided, the target function will
                be the same as the loss function. The function can be quite flexible
                in terms of what is calculated,
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

        """
        self.model = model
        if target_func is None:
            target_func = loss_func

        self.original_loss_func = loss_func
        self.loss_func = flatten_func(self.model)(loss_func)
        self.original_target_func = target_func
        self.target_func = flatten_func(self.model)(target_func)

        if not isinstance(checkpoints, list):
            self.checkpoints = [checkpoints]
        else:
            self.checkpoints = checkpoints

        # current_checkpoint_idx is used to state
        # which checkpoint is currently loaded.
        self.current_checkpoint_idx = None

        # TODO: Make this more general, that is allow customized kwargs.
        self.grad_loss_func = vmap(grad(self.loss_func), in_dims=(None, 1))
        self.grad_loss_func_kwargs = {"in_dims": (None, 1)}
        self.grad_target_func = vmap(grad(self.target_func), in_dims=(None, 1))
        self.grad_target_func_kwargs = {"in_dims": (None, 1)}

    def _load_checkpoints(self, index: int) -> None:
        """This method load the checkpoint at the specified index.

        Args:
            index (int): The index of the checkpoint to be loaded.
        """
        if self.current_checkpoint_idx is None or self.current_checkpoint_idx != index:
            if isinstance(self.checkpoints[index], (str, PosixPath)):
                self.model.load_state_dict(torch.load(self.checkpoints[index]))
            else:
                self.model.load_state_dict(self.checkpoints[index])
            self.current_checkpoint_idx = index
            self.named_parameters = {
                k: p for k, p in self.model.named_parameters() if p.requires_grad
            }
        self.model.eval()

    @staticmethod
    def _genearte_param_layer_map(
        named_parameters: Dict[str, torch.Tensor],
    ) -> List[int]:
        """This function generate the param_layer_map automatically.

        Args:
            named_parameters (Dict[str, torch.Tensor]):
                The named parameters of the model.

        Returns:
            List[int]: The map from the parameter
                to the layer. If None, the map will be generated automatically. Normally
                this should not be stated explicitly by the user, if needed it should
                be the same length as parameters tuple. For example,
                for a two layer model, params = (0.weights1, 0.bias, 1.weights, 1.bias),
                param_layer_map should be [0, 0, 1, 1],resulting in two layers
                as expected.
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
        in_dims: Tuple[Union[None, int], ...] = (None, 1),
        layer_name: Optional[Union[str, List[str]]] = None,
    ) -> Callable:
        """Return a function that computes the gradient of the target function.

        TODO: support partial parameter gradient.

        Args:
            in_dims (Tuple[Union[None, int], ...]): The input dimensions of the target
                function. This should be a tuple of integers and None. The length of the
                tuple should be the same as the number of inputs of the target function.
                If the input is a scalar, the corresponding element should be None.
                If the input is a tensor, the corresponding element should be the
                dimension of the tensor.
            layer_name (Optional[Union[str, List[str]]]): This has not been supported.

        Returns:
            Callable: The function that computes the gradient of the target function.

        Raises:
            NotImplementedError: If layer_name is not None.
        """
        if layer_name is not None:
            error_msg = "layer_name has not been implemented yet."
            raise NotImplementedError(error_msg)

        if self.grad_target_func_kwargs != {"in_dims": in_dims}:
            self.grad_target_func = vmap(grad(self.target_func), in_dims=in_dims)
            self.grad_target_func_kwargs = {"in_dims": in_dims}
        return self.grad_target_func

    def get_target_func(
        self,
        flatten: bool = True,
        layer_name: Optional[Union[str, List[str]]] = None,
    ) -> Callable:
        """Return a function that computes the target function.

        TODO: support partial parameter gradient.

        Args:
            flatten (bool): If True, the target function will be flattened.
            layer_name (Optional[Union[str, List[str]]]): This has not been supported.

        Returns:
            Callable: The target function itself.

        Raises:
            NotImplementedError: If layer_name is not None.
        """
        if layer_name is not None:
            error_msg = "layer_name has not been implemented yet."
            raise NotImplementedError(error_msg)
        if not flatten:
            return self.original_target_func
        return self.target_func

    def get_grad_loss_func(
        self,
        in_dims: Tuple[Union[None, int], ...] = (None, 1),
        layer_name: Optional[Union[str, List[str]]] = None,
    ) -> Callable:
        """Return a function that computes the gradient of the loss function.

        TODO: support partial parameter gradient.

        Args:
            in_dims (Tuple[Union[None, int], ...]): The input dimensions of the loss
                function. This should be a tuple of integers and None. The length of the
                tuple should be the same as the number of inputs of the loss function.
                If the input is a scalar, the corresponding element should be None.
                If the input is a tensor, the corresponding element should be the
                dimension of the tensor.
            layer_name (Optional[Union[str, List[str]]]): This has not been supported.

        Returns:
            Callable: The function that computes the gradient of the loss function.

        Raises:
            NotImplementedError: If layer_name is not None.
        """
        if layer_name is not None:
            error_msg = "layer_name has not been implemented yet."
            raise NotImplementedError(error_msg)

        if self.grad_loss_func_kwargs != {"in_dims": in_dims}:
            self.grad_loss_func = vmap(grad(self.loss_func), in_dims=in_dims)
            self.grad_loss_func_kwargs = {"in_dims": in_dims}
        return self.grad_loss_func

    def get_loss_func(
        self,
        flatten: bool = True,
        layer_name: Optional[Union[str, List[str]]] = None,
    ) -> Callable:
        """Return a function that computes the gradient of the loss function.

        TODO: support partial parameter gradient.

        Args:
            flatten (bool): If True, the loss function will be flattened.
            layer_name (Optional[Union[str, List[str]]]): This has not been supported.

        Returns:
            Callable: The loss function itself.

        Raises:
            NotImplementedError: If layer_name is not None.
        """
        if layer_name is not None:
            error_msg = "layer_name has not been implemented yet."
            raise NotImplementedError(error_msg)
        if not flatten:
            return self.original_loss_func
        return self.loss_func

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

    def get_checkpoints(self) -> List[Union[Dict[str, torch.Tensor], str]]:
        """Return the checkpoints of the model.

        Returns:
            List[Union[Dict[str, torch.Tensor], str]]: The checkpoints of the task.
        """
        return self.checkpoints

    def get_model(self) -> nn.Module:
        """Return the model of the task.

        Returns:
            nn.Module: The model of the task.
        """
        return self.model

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
