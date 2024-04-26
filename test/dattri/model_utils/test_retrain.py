"""Test functions in retrain.py file."""

import shutil
from pathlib import Path

import torch
import yaml

from dattri.model_utils.retrain import retrain_lds


class TestRetrainLDS:
    """Test retrain_lds function."""

    def test_basic_functionality(self):
        """Test basic functionality of retrain_lds function."""

        def train_func(dataloader):  # noqa: ARG001
            return torch.nn.Linear(1, 1)

        dataset = torch.utils.data.TensorDataset(torch.randn(10, 1), torch.randn(10, 1))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        path = "test_retrain_lds"

        retrain_lds(train_func, dataloader, path, num_subsets=10)

        metadata_file = Path(path) / "metadata.yml"
        assert metadata_file.exists(), "metadata.yml file doesn't exist"

        metadata = yaml.safe_load(metadata_file.open("r"))
        assert metadata["mode"] == "lds", "mode doesn't match"
        assert metadata["data_length"] == 10, "data_length doesn't match"  # noqa: PLR2004
        assert metadata["train_func"] == "train_func", "train_func doesn't match"
        assert len(metadata["subset_dir_map"]) == 10, "subset_dir_map is incorrect"  # noqa: PLR2004

        for index in metadata["subset_dir_map"]:
            model_dir = Path(metadata["subset_dir_map"][index])
            assert model_dir.exists(), f"{model_dir} doesn't exist"
            model_weights = model_dir / "model_weights_0.pt"
            assert model_weights.exists(), f"{model_weights} doesn't exist"
            model = torch.nn.Linear(1, 1)
            model.load_state_dict(torch.load(model_weights))

        shutil.rmtree(path)
