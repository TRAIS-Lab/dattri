"""Test functions in dropout.py file."""

import torch
from torch import nn

from dattri.model_util.dropout import activate_dropout


class TestDropout:
    """Test dropout function."""

    def test_basic_functionality(self):
        """Test basic functionality of dropout function."""

        class MLP(nn.Module):
            def __init__(
                self,
                input_dim=20,
                hidden_dim=10,
                output_dim=5,
                dropout_prob=0.5,
            ):
                super(MLP, self).__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.dropout1 = nn.Dropout(p=dropout_prob)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.dropout2 = nn.Dropout(p=dropout_prob)
                self.fc3 = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.dropout1(x)
                x = torch.relu(self.fc2(x))
                x = self.dropout2(x)
                return self.fc3(x)

        test_data = torch.rand(5, 20)
        model = MLP()

        model.eval()
        before_result_1 = model(test_data)
        before_result_2 = model(test_data)
        assert torch.allclose(before_result_1, before_result_2)

        # activate all
        model = activate_dropout(model, dropout_prob=0.2)

        after_result = model(test_data)
        assert not torch.allclose(before_result_1, after_result)

        # resume by calling eval again
        model.eval()
        resume_result = model(test_data)
        assert torch.allclose(before_result_1, resume_result)

        # activate specific
        model = activate_dropout(model, ["dropout2"], dropout_prob=0.2)
        spec_result = model(test_data)
        assert not torch.allclose(before_result_1, spec_result)
