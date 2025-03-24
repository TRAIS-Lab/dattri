"""Test for RPS."""

import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from dattri.algorithm.rps import RPSAttributor
from dattri.benchmark.utils import SubsetSampler
from dattri.task import AttributionTask


class MLP(nn.Module):
    """A multi-layer perceptron model for MNIST dataset."""

    def __init__(
        self,
        input_size=28 * 28,
        hidden_size1=128,
        hidden_size2=64,
        num_classes=10,
    ):
        """Initialize the multi-layer perceptron model."""
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        """Forward pass of the multi-layer perceptron model."""
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class TestRPS:
    """Test for RPS."""

    def test_rps_multi(self):
        """Test for RPS."""
        # multi-class
        # init the model
        model = MLP(num_classes=10)
        dataset = TensorDataset(torch.randn(10, 1, 28, 28), torch.randint(0, 10, (10,)))
        full_train_loader = DataLoader(dataset, batch_size=2)
        test_loader = DataLoader(dataset, batch_size=2)
        train_loader = DataLoader(
            dataset,
            batch_size=2,
            sampler=SubsetSampler(range(4)),
        )

        # define the loss function
        def f(pre_activation_list, label_list):
            return functional.cross_entropy(pre_activation_list, label_list)

        task = AttributionTask(
            loss_func=f,
            model=model,
            checkpoints=model.state_dict(),
        )

        # define the RPS attributor w/ cache
        attributor = RPSAttributor(
            task=task,
            final_linear_layer_name="fc3",
            epoch=10,
        )
        attributor.cache(full_train_loader)

        # Original test
        result = attributor.attribute(train_loader, test_loader)
        assert result.shape == (4, 10)

        # define the RPS attributor w/o cache
        attributor = RPSAttributor(
            task=task,
            final_linear_layer_name="fc3",
            epoch=100,
        )
        result = attributor.attribute(train_loader, test_loader)
        assert result.shape == (4, 10)

    def test_rps_binary(self):
        """Test for RPS."""
        # binary-class
        # init the model
        model = MLP(num_classes=1)
        dataset = TensorDataset(torch.randn(10, 1, 28, 28), torch.randint(0, 2, (10,)))
        full_train_loader = DataLoader(dataset, batch_size=2)
        test_loader = DataLoader(dataset, batch_size=2)
        train_loader = DataLoader(
            dataset,
            batch_size=2,
            sampler=SubsetSampler(range(4)),
        )

        def f(pre_activation_list, label_list):
            return functional.binary_cross_entropy_with_logits(
                pre_activation_list,
                label_list,
            )

        task = AttributionTask(
            loss_func=f,
            model=model,
            checkpoints=model.state_dict(),
        )

        # define the RPS attributor w/ cache
        attributor = RPSAttributor(
            task=task,
            final_linear_layer_name="fc3",
            epoch=10,
        )
        attributor.cache(full_train_loader)
        result = attributor.attribute(train_loader, test_loader)
        assert result.shape == (4, 10)

        # define the RPS attributor w/o cache
        attributor = RPSAttributor(
            task=task,
            final_linear_layer_name="fc3",
            epoch=10,
        )
        result = attributor.attribute(train_loader, test_loader)
        assert result.shape == (4, 10)
