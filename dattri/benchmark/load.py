"""This module provides functions to load predefined benchmark settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Dict, Tuple

import pathlib
import warnings
import zipfile
from functools import partial
from io import BytesIO

import requests
import torch

from dattri.benchmark.datasets.cifar import (
    create_cifar2_dataset,
    create_resnet9_model,
    loss_cifar_resnet9,
    train_cifar_resnet9,
)
from dattri.benchmark.datasets.maestro import (
    create_maestro_datasets,
    create_musictransformer_model,
    loss_maestro_musictransformer,
    train_maestro_musictransformer,
)
from dattri.benchmark.datasets.mnist import (
    create_lr_model,
    create_mlp_model,
    create_mnist_dataset,
    loss_mnist_lr,
    loss_mnist_mlp,
    train_mnist_lr,
    train_mnist_mlp,
)
from dattri.benchmark.datasets.shakespeare_char import create_shakespeare_dataset
from dattri.benchmark.utils import SubsetSampler

REPO_URL = "https://huggingface.co/datasets/trais-lab/dattri-benchmark/resolve/main/"


def generate_url_map(identifier: str) -> Dict[str, Any]:
    """Generate the URL map for the benchmark setting.

    Args:
        identifier (str): The identifier for the benchmark setting.

    Returns:
        Dict[str, Any]: The URL map for the benchmark setting.
    """
    return {
        "models_full": REPO_URL + f"{identifier}/{identifier}_full.zip?download=true",
        "models_half": REPO_URL + f"{identifier}/{identifier}_half.zip?download=true",
        "groundtruth": {
            "lds": [
                REPO_URL + f"{identifier}/lds/indices_lds.pt?download=true",
                REPO_URL + f"{identifier}/lds/target_values_lds.pt?download=true",
            ],
            "loo": [
                REPO_URL + f"{identifier}/loo/indices_loo.pt?download=true",
                REPO_URL + f"{identifier}/loo/target_values_loo.pt?download=true",
            ],
        },
    }


SUPPORTED_DATASETS = {
    "mnist": create_mnist_dataset,
    "cifar2": create_cifar2_dataset,
    "shakespeare": create_shakespeare_dataset,
    "maestro": partial(create_maestro_datasets, generated_music=True),
}

LOSS_MAP = {
    "mnist_mlp": loss_mnist_mlp,
    "mnist_lr": loss_mnist_lr,
    "cifar2_resnet9": loss_cifar_resnet9,
    "shakespeare_nanogpt": None,
    "maestro_musictransformer": loss_maestro_musictransformer,
}

TRAIN_FUNC_MAP = {
    "mnist_mlp": train_mnist_mlp,
    "mnist_lr": train_mnist_lr,
    "cifar2_resnet9": train_cifar_resnet9,
    "shakespeare_nanogpt": None,
    "maestro_musictransformer": train_maestro_musictransformer,
}

MODEL_MAP = {
    "mnist_mlp": partial(create_mlp_model, "mnist"),
    "mnist_lr": partial(create_lr_model, "mnist"),
    "cifar2_resnet9": create_resnet9_model,
    "shakespeare_nanogpt": None,
    "maestro_musictransformer": create_musictransformer_model,
}


def _count_folders(directory_path: str) -> int:
    """Count the number of folders in a directory.

    Args:
        directory_path (str): The path to the directory.

    Returns:
        int: The number of folders in the directory.
    """
    path = pathlib.Path(directory_path)
    folders = [item for item in path.iterdir() if item.is_dir()]
    return len(folders)


def _download(
    url: str,
    destination_path: str = ".",
    unzip: bool = False,
    file_name: str = "file.zip",
) -> None:
    """Download helper function.

    Args:
        url (str): The URL of the file to download.
        destination_path (str): The path to save the downloaded file.
        unzip (bool): Whether to unzip the downloaded file.
        file_name (str): The name of the downloaded file.

    Raises:
        ValueError: If the download fails.
    """
    # Send a GET request to the URL
    response = requests.get(url)  # noqa: S113

    # Check if the request was successful
    destination_path.mkdir(parents=True, exist_ok=True)
    if response.status_code == 200:  # noqa: PLR2004
        if unzip:
            with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(destination_path)
        else:
            with (destination_path / file_name).open(mode="wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
    elif response.status_code == 404:  # noqa: PLR2004
        warn_msg = f"{url} does not exist, which could be a\
                     reasonable case if the benchmark is too large."
        warnings.warn(warn_msg, stacklevel=2)
    else:
        error_msg = f"Failed to download file. Status code: {response.status_code}"
        raise ValueError(error_msg)


def load_benchmark(  # noqa:PLR0914
    model: str,
    dataset: str,
    metric: str,
    download_path: str = "~/.dattri",
    redownload: bool = False,
) -> Tuple[Dict[str, Any], Tuple[torch.Tensor, torch.Tensor]]:
    """Load benchmark settings for a given model, dataset, and metric.

    Please check https://huggingface.co/datasets/trais-lab/dattri-benchmark
    to see the supported benchmark settings (model, dataset).

    Args:
        model (str): The model name for the benchmark setting.
        dataset (str): The dataset name for the benchmark setting.
        metric (str): The matrics name for the benchmark setting, which would
            affected the ground truth. Currently only "lds" and "loo" are
            supported.
        download_path (str): The path to download the benchmark files.
        redownload (bool): Whether to redownload the benchmark files.

    Returns:
        Tuple[Dict[str, Any], Tuple[torch.Tensor, torch.Tensor]]:

            The first dictionary contains the attribution inputs,
            the items are listed as following.
            - "model": The model instance for the benchmark setting.
            - "models_full": The pre-trained model checkpoints' path with
                full train dataset, presented as a list of path(str). The
                models are trained with same hyperparameters and dataset while
                the only difference is the seed for random initialization.
            - "models_half": The pre-trained model checkpoints' path with
                half train dataset, presented as a list of path(str). The
                models are trained with same hyperparameters while
                the difference is the dataset sampling (half sampling) for
                each model checkpoint.
            - "train_dataset": The path to the training dataset with the
                same order as the ground-truth's indices.
            - "test_dataset": The path to the testing dataset with the
                same order as the ground-truth's indices.
            - "loss_func": The loss function for the model training. Normally
                speaking, this should be the same as the target function.
            - "target_func": The target function for the data attribution. Normally
                speaking, this should be the same as the loss function.
            - "train_func": The training function for the model. Normally it's
                not required if the pre-trained model checkpoints is enough
                for the algorithm you want to benchmark.

            The second tuple contains the ground truth for the benchmark,
            the items are subjected to change for each benchmark settings.
            It can be directly sent to the
            metrics function defined in `dattri.metric`. Notably, the ground-truth
            depends on the `metric` parameter user stated.

    Raises:
        ValueError: If the model or dataset is not supported.
    """
    identifier = f"{dataset}_{model}"
    if identifier not in MODEL_MAP:
        error_msg = f"The combination of {identifier} is not supported."
        raise ValueError(error_msg)

    url_map = generate_url_map(identifier)
    download_path = pathlib.Path(download_path).expanduser()

    if not (download_path / "benchmark" / identifier).exists() or redownload:
        for key in ["models_full", "models_half"]:
            _download(
                url_map[key],
                download_path / "benchmark" / identifier / key,
                unzip=True,
            )
        for path in url_map["groundtruth"]["lds"]:
            _download(
                path,
                download_path / "benchmark" / identifier / "lds",
                unzip=False,
                file_name=path.split("?")[0].split("/")[-1],
            )
        for path in url_map["groundtruth"]["loo"]:
            _download(
                path,
                download_path / "benchmark" / identifier / "loo",
                unzip=False,
                file_name=path.split("?")[0].split("/")[-1],
            )

    models_full_count = _count_folders(
        download_path / "benchmark" / identifier / "models_full",
    )
    models_full_list = [
        download_path
        / "benchmark"
        / identifier
        / "models_full"
        / f"{i}"
        / "model_weights_0.pt"
        for i in range(models_full_count)
    ]
    models_half_count = _count_folders(
        download_path / "benchmark" / identifier / "models_half",
    )
    models_half_list = [
        download_path
        / "benchmark"
        / identifier
        / "models_half"
        / f"{i}"
        / "model_weights_0.pt"
        for i in range(models_half_count)
    ]
    if SUPPORTED_DATASETS[dataset] is not None:
        train_dataset, test_dataset = SUPPORTED_DATASETS[dataset](
            download_path / "dataset",
        )
    else:
        train_dataset = test_dataset = None

    if LOSS_MAP[identifier] is not None:
        loss_func = target_func = LOSS_MAP[identifier]
    else:
        loss_func = target_func = None

    if TRAIN_FUNC_MAP[identifier] is not None:
        train_func = TRAIN_FUNC_MAP[identifier]
    else:
        train_func = None

    target_values = torch.load(
        download_path
        / "benchmark"
        / identifier
        / metric
        / f"target_values_{metric}.pt",
    )
    indices = torch.load(
        download_path / "benchmark" / identifier / metric / f"indices_{metric}.pt",
    )

    if MODEL_MAP[identifier] is not None:
        model = MODEL_MAP[identifier]().eval()

    return {
        "model": model,
        "models_full": models_full_list,
        "models_half": models_half_list,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "train_sampler": SubsetSampler(range(5000)),
        "test_sampler": SubsetSampler(range(500)),
        "loss_func": loss_func,
        "target_func": target_func,
        "train_func": train_func,
    }, (target_values, indices)
