"""This script automates the retraining of nanoGPT with different configurations."""

# ruff: noqa: S306, S602, EXE002, S404, PLW1510

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


def retrain(num_runs: int,
            seed: int,
            subset_ratio: float,
            config_path: str,
            dataset_path: str,
            base_out_dir: str) -> None:
    """Retrain the model multiple times with varying configurations.

    Args:
        num_runs: Number of training runs to perform.
        seed: Initial random seed for training.
        subset_ratio: Subset ratio of the training data.
        config_path: Path to the training configuration file.
        dataset_path: Path to the dataset folder.
        base_out_dir: Base directory for output models.
    """
    os.chdir("./models/nanoGPT")
    config = Path(config_path).read_text(encoding="utf-8").splitlines()
    for i in range(num_runs):
        modified_config = []
        for line in config:
            if line.startswith("out_dir"):
                modified_config.append(f"out_dir = '{base_out_dir}/model_{i + 1}'\n")
            elif line.startswith("seed"):
                modified_config.append(f"seed = {seed}\n")
            elif line.startswith("subset_ratio"):
                modified_config.append(f"subset_ratio = {subset_ratio}\n")
            elif line.startswith("dataset_path"):
                modified_config.append(f"dataset_path = {dataset_path}\n")
            else:
                modified_config.append(line)

        temp_config_path = Path(tempfile.mktemp())
        temp_config_path.write_text("\n".join(modified_config), encoding="utf-8")
        command = f"python train.py {temp_config_path}"
        subprocess.run(command, shell=True)

        temp_config_path.unlink()
        seed += 1


def main() -> None:
    """Main function to parse arguments and call the retraining function."""
    parser = argparse.ArgumentParser(
        description="Retrain models with different seeds and output directories.")
    parser.add_argument("--num_runs", type=int,
                        default=5,
                        help="Number of training runs to perform.")
    parser.add_argument("--seed", type=int,
                        default=42,
                        help="Initial random seed for training.")
    parser.add_argument("--subset_ratio", type=float,
                        default=1.0,
                        help="Subset ratio of the training data.")
    parser.add_argument("--dataset_path", type=str,
                        default="./dataset/shakespeare_char",
                        help="Path to the dataset.")
    parser.add_argument("--config_path", type=str,
                        default="./models/nanoGPT/config/train_shakespeare_char.py",
                        help="Path to the training configuration file.")
    parser.add_argument("--base_out_dir", type=str,
                        default="out-shakespeare",
                        help="Base directory for output models.")

    args = parser.parse_args()

    retrain(args.num_runs,
            args.seed,
            args.subset_ratio,
            args.config_path,
            args.dataset_path,
            args.base_out_dir)


if __name__ == "__main__":
    main()
