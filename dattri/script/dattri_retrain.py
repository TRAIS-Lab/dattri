"""This script is used in terminal to retrain models on various dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Dict, List

import argparse
from pathlib import Path

import torch
from torch.utils.data import SubsetRandomSampler

from dattri.benchmark.datasets.cifar import (
    create_cifar2_dataset,
    train_cifar_resnet9,
)
from dattri.benchmark.datasets.imagenet import (
    create_imagenet_dataset,
    train_imagenet_resnet18,
)
from dattri.benchmark.datasets.maestro import (
    create_maestro_datasets,
    train_maestro_musictransformer,
)
from dattri.benchmark.datasets.mnist import (
    create_mnist_dataset,
    train_mnist_lr,
    train_mnist_mlp,
)
from dattri.model_util.retrain import retrain_lds, retrain_loo

SUPPORTED_MODELS = ["lr", "resnet18", "resnet9", "musictransformer"]

SUPPORTED_SETTINGS = {
    "mnist_lr": train_mnist_lr,
    "mnist_mlp": train_mnist_mlp,
    "imagenet_resnet18": train_imagenet_resnet18,
    "cifar2_resnet9": train_cifar_resnet9,
    "maestro_musictransformer": train_maestro_musictransformer,
}
SUPPORTED_RETRAINING_MODE = {"loo": retrain_loo, "lds": retrain_lds}
SUPPORTED_DATASETS = {
    "mnist": create_mnist_dataset,
    "imagenet": create_imagenet_dataset,
    "cifar2": create_cifar2_dataset,
    "maestro": create_maestro_datasets,
}
DEFAULT_BATCH_SIZE = {
    "mnist_lr": 32,
    "mnist_mlp": 64,
    "imagenet_resnet18": 256,
    "cifar2_resnet9": 64,
    "maestro_musictransformer": 64,
}


def partition_type(arg: str) -> List[int]:
    """Custom type function for parsing partition arguments.

    Args:
        arg (str): The partition argument in the format [start, end, total].

    Returns:
        List[int]: The parsed partition.

    Raises:
        ValueError: If the partition is not in the correct format.
    """
    res = [int(x) if x.lower() != "none" else None for x in arg.split(",")]
    if len(res) != 3:  # noqa: PLR2004
        message = "--partition should be in the format [start, end, total]."
        raise ValueError(message)
    return res


def key_value_pair(arg: Dict[str, Any]) -> tuple[str, Any]:
    """Convert a string in key=value format to a tuple (key, value).

    Args:
        arg (Dict[str, Any]): The argument in key=value format.

    Returns:
        tuple[str, Any]: The parsed key-value pair.
    """
    key, value = arg.split("=")
    if value.isdigit():
        value = int(value)
    return key, value


def main() -> None:
    """This function is used to retrain models on various datasets.

    :return: None
    """
    parser = argparse.ArgumentParser(description="Retrain models on various datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=SUPPORTED_DATASETS.keys(),
        help=f"The dataset to use for retraining.\
               It should be one of {list(SUPPORTED_DATASETS.keys())}.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=SUPPORTED_MODELS,
        help=f"The dataset to use for retraining.\
               It should be one of {SUPPORTED_MODELS}.",
    )
    parser.add_argument(
        "--train_subset",
        type=int,
        default=5000,
        help="The number of training samples to use,\
              for retraining. Default to 5000, set to\
              -1 to use all the data.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=SUPPORTED_RETRAINING_MODE.keys(),
        help=f"The retraining mode to use.\
               It should be one of {list(SUPPORTED_RETRAINING_MODE.keys())}.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=".",
        help="The path to save the retrained model.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=".",
        help="The path to the dataset.",
    )
    parser.add_argument("--seed", type=int, default=0, help="The seed for retraining.")
    parser.add_argument(
        "--partition",
        type=partition_type,
        default=[0, None, None],
        help="The partition for retraining, with the format [start, end, total].\
              This is used for\
              parallel retraining. If the mode is 'lds', the partition should be\
              [`start_id`, `start_id+subset_num`, `total_num_subsets`].\
              If the mode is 'loo', the partition\
              should be [`start_id`, `end_id`, None], the third element is not used.\
              The `indices` will be\
              stated as range(`start_id`, `end_id`). Default value means\
              the script will run all the data, that\
              is 100 subsets for 'lds' and all the data for 'loo'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="The device to train the model on.",
    )
    parser.add_argument(
        "--extra_param",
        type=key_value_pair,
        action="append",
        help="extra parameters to be passed to the retrain function.\
              Must be in key=value format.",
    )
    args = parser.parse_args()

    if args.dataset is None or args.model is None or args.mode is None:
        parser.print_help()
        return

    setting = f"{args.dataset}_{args.model}"

    train_func = SUPPORTED_SETTINGS[setting]
    retrain_helper = SUPPORTED_RETRAINING_MODE[args.mode]
    dataset_func = SUPPORTED_DATASETS[args.dataset]

    path = Path(args.data_path)
    if not path.exists():
        path.mkdir(parents=True)

    dataset_train, _ = dataset_func(args.data_path)

    if args.train_subset > 0:
        sampler = SubsetRandomSampler(list(range(args.train_subset)))
    else:
        sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=DEFAULT_BATCH_SIZE[setting],
        shuffle=sampler is None,
        sampler=sampler,
    )

    if args.partition[1] is None:
        args.partition[1] = len(dataset_train) if args.mode == "loo" else 100
    if args.partition[2] is None:
        args.partition[2] = len(dataset_train) if args.mode == "loo" else 100

    if args.mode == "lds":
        kwargs = {}
        kwargs["num_subsets"] = int(args.partition[1]) - int(args.partition[0])
        kwargs["total_num_subsets"] = int(args.partition[2])
        kwargs["start_id"] = int(args.partition[0])
    if args.mode == "loo":
        kwargs = {}
        kwargs["indices"] = list(range(int(args.partition[0]), int(args.partition[1])))
    kwargs["device"] = args.device
    kwargs.update(args.extra_param or {})

    retrain_helper(
        train_func,
        dataloader=train_loader,
        path=args.save_path,
        seed=args.seed,
        **kwargs,
    )


if __name__ == "__main__":
    main()
