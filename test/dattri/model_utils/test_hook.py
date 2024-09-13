"""Test functions in hook.py file."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dattri.model_util.hook import get_final_layer_io


class MyNet(nn.Module):
    """A test nn.Module PyTorch model."""

    def __init__(self):
        """Initialze the layers of the model."""
        super(MyNet, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 30)
        self.layer3 = nn.Linear(30, 40)
        self.layer4 = nn.Linear(40, 50)

    def forward(self, x: torch.Tensor):
        """Compute the forward pass of the model.

        Args:
            x (torch.Tensor): model input.

        Returns:
            torch.Tensor: model output.
        """
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        return self.layer4(x)


class TestGetFinalLayerFeature:
    """Test get_final_linear_layer_input function."""

    def test_output_shape(self):
        """Test output shape of get_final_linear_layer_input function."""
        model = MyNet()
        data = torch.randn(10, 10)
        labels = torch.randint(0, 2, (10,))  # Dummy labels
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=2)

        # specify the layer name
        layer_name = "layer4"
        feat_dim_gt = 40
        out_dim_gt = 50
        feature, output = get_final_layer_io(model, layer_name, dataloader)
        assert feature.shape[0] == len(dataset)
        assert feature.shape[1] == feat_dim_gt
        assert output.shape[0] == len(dataset)
        assert output.shape[1] == out_dim_gt
