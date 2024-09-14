"""This module contains functions for creating the Shakespeare dataset."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import os
from pathlib import Path
import dattri
from torch.utils.data import  Dataset
import torch

if TYPE_CHECKING:
    from typing import Tuple


class CustomDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        ix = idx * self.block_size
        x = torch.from_numpy(self.data[ix:ix + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[ix + 1:ix + 1 + self.block_size].astype(np.int64))
        if len(y) < self.block_size:
            y = torch.cat([y, torch.zeros(self.block_size - len(y), dtype=torch.int64)])
        return x, y

    def get_subset(self, indices):
        subset_data = [self[i] for i in indices]
        subset_x = torch.stack([item[0] for item in subset_data])
        subset_y = torch.stack([item[1] for item in subset_data])
        return subset_x, subset_y


def create_shakespeare_dataset(
        data_path: str,
        block_size: int = 256,
    )-> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Create Shakespeare dataset for training and testing.

    Args:
        data_path: The path to the dataset.
        block_size: The size of each block in the dataset.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: The training dataset,
            and the testing dataset.
    """

    data_path = Path(dattri.__file__).parent / Path(
        f"benchmark/datasets/shakespeare_char",
    )

    val_data = np.memmap(os.path.join(data_path, 'val.bin'), dtype=np.uint16, mode='r')
    val_dataset = CustomDataset(val_data, block_size)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    train_data = np.memmap(os.path.join(data_path, 'train.bin'), dtype=np.uint16, mode='r')
    train_dataset = CustomDataset(train_data, block_size)
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    return train_dataset, val_dataset
