"""This module contains functions for data processing on MAESTRO dataset."""

# ruff: noqa: PLR1702
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import requests

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
        dataset_path (str): Root directory of the MAESTRO Dataset. The data source
            is https://magenta.tensorflow.org/datasets/maestro#v200. If the processed
            dataset is not found, this funtion will download the dataset automatically
            to this path, unzip and pre-process it. The processed data will be at
            "dataset_path/maestro-v2.0.0-processed".

    Returns:
        Tuple[Dataset, Dataset, Dataset]: The training, validate and testing
            MAESTRO datasets.
    """
    # define the dir name
    processed_dataset_dir = Path(dataset_path) / "maestro-v2.0.0-processed"
    unzip_dataset_dir = Path(dataset_path) / "maestro-v2.0.0"

    # if no processed dir
    if not processed_dataset_dir.exists():
        # first get the zip file
        zip_dataset_file = Path(dataset_path) / "maestro-v2.0.0-midi.zip"
        dataset_urls = "https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip"
        response = requests.get(dataset_urls, stream=True, timeout=20)
        with zip_dataset_file.open("wb") as midi:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    midi.write(chunk)

        # then unzip the file
        with zipfile.ZipFile(zip_dataset_file, "r") as zip_ref:
            for file_info in zip_ref.infolist():
                zip_ref.extract(file_info, dataset_path)

        # then pre-process the unzipped file
        prep_maestro_midi(unzip_dataset_dir, processed_dataset_dir)

    # create the train datasets from pre-processed data
    train_dataset, val_dataset, _ = create_epiano_datasets(
        processed_dataset_dir,
        MAX_SEQUENCE,
        full_version=FULL_VERSION,
    )

    return train_dataset, val_dataset
