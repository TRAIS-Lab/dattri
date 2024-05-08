"""This module contains functions for data processing on MAESTRO dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from dattri.benchmark.models.MusicTransformer.dataset.e_piano import (
    create_epiano_datasets,
)
from dattri.benchmark.models.MusicTransformer.dataset.preprocess_midi import (
    prep_maestro_midi,
)

# Data Processing Hyper-parameters ############
MAX_SEQUENCE = 256  # Maximum midi sequence to consider
BATCH_SIZE = 64  # Batch size to use
N_WORKERS = 1  # Number of threads for the dataloader


def create_maestro_datasets(
    dataset_path: str,
    train_size: int = 5000,
    val_size: int = 500,
    test_size: int = 500,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing.

    Args:
        dataset_path (str): Root directory of the MAESTRO Dataset. Should be
            downloaded from https://magenta.tensorflow.org/datasets/maestro#v200.
            The data is called "maestro-v2.0.0-midi.zip". Please unzip the file.
        train_size (int): The train loader size. Default to 5000.
        val_size (int): The validation loader size. Default to 500.
        test_size (int): The test loader size. Default to 500.
        seed (int): The random seed used to sample these examples.

    Returns:
        A tuple of three DataLoader objects for train/val/test.
    """
    # set seed
    torch.manual_seed(seed)
    # read unzipped files and preprocess the files
    processed_path = f"{dataset_path}-processed"
    processed_dir = Path(dataset_path).parent / processed_path
    if not processed_dir.exists():
        # will create dir called maestro-v2.0.0-midi-processed
        prep_maestro_midi(dataset_path, processed_dir)

    train_dataset, val_dataset, test_dataset = create_epiano_datasets(
        processed_path,
        MAX_SEQUENCE,
        full_version=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=N_WORKERS,
        sampler=SubsetRandomSampler(range(train_size)),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=N_WORKERS,
        sampler=SubsetRandomSampler(range(val_size)),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=N_WORKERS,
        sampler=SubsetRandomSampler(range(test_size)),
    )

    return train_loader, val_loader, test_loader
