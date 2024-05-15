"""This module contains functions for data processing on MAESTRO dataset."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple

    from torch.utils.data import Dataset


from dattri.benchmark.models.MusicTransformer.dataset.e_piano import (
    create_epiano_datasets,
)
from dattri.benchmark.models.MusicTransformer.dataset.preprocess_midi import (
    prep_maestro_midi,
)

# Data Processing Hyper-parameters ############
MAX_SEQUENCE = 256  # Maximum midi sequence to consider
FULL_VERSION = True  # State if the whole dataset will be transversed.
# Data Processing Hyper-parameters ############


def create_maestro_datasets(
    dataset_path: str,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Create MAESTRO dataset.

    Args:
        dataset_path (str): Root directory of the MAESTRO Dataset. Should be
            downloaded from https://magenta.tensorflow.org/datasets/maestro#v200.
            The data is called "maestro-v2.0.0-midi.zip". Please unzip the file.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: The training, validate and testing
            MAESTRO datasets.
    """
    # read unzipped files and preprocess the files
    processed_path = f"{dataset_path}-processed"
    processed_dir = Path(dataset_path).parent / processed_path
    if not processed_dir.exists():
        # will create dir called maestro-v2.0.0-midi-processed
        prep_maestro_midi(dataset_path, processed_dir)

    train_dataset, val_dataset, test_dataset = create_epiano_datasets(
        processed_path,
        MAX_SEQUENCE,
        full_version=FULL_VERSION,
    )

    return train_dataset, val_dataset, test_dataset
