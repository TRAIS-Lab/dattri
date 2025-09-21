"""This script automates the retraining of nanoGPT with different configurations."""

# ruff: noqa: S306, S602, EXE002, S404, PLW1510

import argparse
import logging
import os
import subprocess
import tempfile
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


def retrain(  # noqa:PLR0912
    seed: int,
    subset_ratio: float,
    config_path: str,
    dataset_path: str,
    dataset_file: str,
    save_path: str,
    partition: list,
    only_download: bool = False,
) -> None:
    """Retrains the nanoGPT model multiple times.

    Args:
        seed (int): The initial random seed for training, incrementing with each run.
        subset_ratio (float): Fraction of the dataset to use for training.
        config_path (str): Path to the model's configuration file.
        dataset_path (str): Path where the training data is stored.
        dataset_file (str): Name of the training data file.
        save_path (str): Directory where each model output is saved.
        partition (list): Range of data subsets, [start_id, end_id, total_subsets].
        only_download (bool): Whether to only download the dataset.

    Raises:
        ValueError: If subset_ratio is negative or greater than 1.
    """
    if not Path.exists(Path(dataset_path) / "meta.pkl"):
        if dataset_file is not None:
            command = f"python {Path(dataset_path) / 'prepare.py'} \
                        --data_file {dataset_file}"
        else:
            command = f"python {Path(dataset_path) / 'prepare.py'}"
        subprocess.run(command, shell=True)

    if only_download:
        info_msg = f"Dataset is downloaded to {dataset_path}. Exiting."
        logger.info(info_msg)
        return

    import dattri

    os.chdir(Path(dattri.__file__).parent / Path("benchmark/models/nanoGPT"))
    config = Path(config_path).read_text(encoding="utf-8").splitlines()

    start_id, end_id, total_num = partition
    end_id = min(end_id, total_num)
    # Check subset ratio if satisfy the requirement
    if subset_ratio <= 0:
        error_message = "subset_ratio must be non-negative"
        raise ValueError(error_message)
    if subset_ratio > 1:
        error_message = "subset_ratio must smaller or equal to 1"
        raise ValueError(error_message)
    seed = seed + start_id - 1
    for i in range(start_id, end_id):
        modified_config = []
        for line in config:
            if line.startswith("out_dir"):
                modified_config.append(f"out_dir = '{save_path}/model_{i + 1}'\n")
            elif line.startswith("seed"):
                modified_config.append(f"seed = {seed}\n")
            elif line.startswith("subset_ratio"):
                modified_config.append(f"subset_ratio = {subset_ratio}\n")
            elif line.startswith("dataset_path"):
                modified_config.append(f"dataset_path = '{dataset_path}'\n")
            else:
                modified_config.append(line)
        modified_config.append(f"dataset_path = '{dataset_path!s}'\n")

        temp_config_path = Path(tempfile.mktemp())
        temp_config_path.write_text("\n".join(modified_config), encoding="utf-8")
        command = f"python train.py {temp_config_path}"
        subprocess.run(command, shell=True)

        temp_config_path.unlink()
        seed += 1


def main() -> None:
    """Main function to parse arguments and call the retraining function."""
    parser = argparse.ArgumentParser(
        description="Retrain models with different configurations.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="output_models",
        help="The path to save the retrained model.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="shakespeare_char",
        help="The dataset to be retrained. Choose from\
                        ['shakespeare_char', 'tinystories'].",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="TinyStoriesV2-GPT4-train.txt",
        help="(optional) Path to the tinystories dataset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for retraining.")
    parser.add_argument(
        "--partition",
        type=int,
        nargs=3,
        default=[0, 5, 5],
        help="Partition for retraining, format in [start, end, total].",
    )
    parser.add_argument(
        "--subset_ratio",
        type=float,
        default=0.5,
        help="Subset ratio of the training data.",
    )
    parser.add_argument(
        "--only_download_dataset",
        action="store_true",
        help="Only download the dataset.",
    )

    args = parser.parse_args()

    import dattri

    args.config_path = Path(dattri.__file__).parent / Path(
        f"benchmark/models/nanoGPT/config/train_{args.dataset}.py",
    )
    args.data_path = Path(dattri.__file__).parent / Path(
        f"benchmark/datasets/{args.dataset}",
    )

    retrain(
        args.seed,
        args.subset_ratio,
        args.config_path,
        args.data_path,
        args.data_file,
        args.save_path,
        args.partition,
        args.only_download_dataset,
    )


if __name__ == "__main__":
    main()
