"""Test for influence function."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dattri.benchmark.datasets.mnist import train_mnist_mlp
from dattri.task import AttributionTask


class TestTask:
    """Test for AttributionTask."""

    def test_task_partial(self):
        """Test for partial parameter."""
        train_dataset = TensorDataset(
            torch.randn(20, 1, 28, 28),
            torch.randint(0, 10, (20,)),
        )
        train_loader = DataLoader(train_dataset, batch_size=4)
        model = train_mnist_mlp(train_loader)

        def f(params, data_target_pair):
            image, label = data_target_pair
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image)
            return loss(yhat, label.long())

        task = AttributionTask(loss_func=f, model=model, checkpoints=model.state_dict())

        grad_func_partial = task.get_grad_loss_func(
            layer_name=["fc3.weight", "fc3.bias"],
            ckpt_idx=0,
        )
        grad_func_full = task.get_grad_loss_func()

        for train_batch_data_ in train_loader:
            train_batch_data = tuple(data.unsqueeze(0) for data in train_batch_data_)
            params_partial, _ = task.get_param(
                ckpt_idx=0,
                layer_name=["fc3.weight", "fc3.bias"],
            )
            params_full, _ = task.get_param(ckpt_idx=0)
            gradient_partial = grad_func_partial(params_partial, train_batch_data)
            gradient_full = grad_func_full(params_full, train_batch_data)
            assert torch.allclose(
                gradient_partial,
                gradient_full[:, -gradient_partial.size(1) :],
            )
