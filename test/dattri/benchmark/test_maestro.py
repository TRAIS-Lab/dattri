"""Test MAESTRO functions."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from dattri.benchmark.datasets.maestro import (
    create_musictransformer_model,
    loss_maestro_musictransformer,
    train_maestro_musictransformer,
)


class TestMaestro:
    """Test MAESTRO functions."""

    train_seq1 = torch.randint(0, 300, (10, 256))
    train_seq2 = torch.randint(0, 300, (10, 256))
    test_seq1 = torch.randint(0, 300, (1, 256))
    test_seq2 = torch.randint(0, 300, (1, 256))

    train_dataset = TensorDataset(train_seq1, train_seq2)
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_dataset = TensorDataset(test_seq1, test_seq2)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    def test_create_musictransformer_model(self):
        """Test create_musictransformer_model."""
        model = create_musictransformer_model(device="cpu")
        assert isinstance(model, torch.nn.Module)

    def test_train_maestro_musictransformer(self):
        """Test train_maestro_musictransformer."""
        model = train_maestro_musictransformer(self.test_dataloader, device="cpu")
        assert isinstance(model, torch.nn.Module)

    def test_loss_maestro_musictransformer(self):
        """Test loss_maestro_musictransformer."""
        model = train_maestro_musictransformer(
            self.train_dataloader,
            num_epoch=1,
            device="cpu",
        )
        torch.save(model.state_dict(), "test_model.pt")
        loss = loss_maestro_musictransformer("test_model.pt", self.test_dataloader)
        assert isinstance(loss, torch.Tensor)

        # remove the saved model for clean up
        Path("test_model.pt").unlink(missing_ok=True)
