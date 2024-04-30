"""This module contains functions for model training on MAESTRO dataset."""
# reference: https://github.com/gwinndr/MusicTransformer-Pytorch
# author: Damon Gwinn

from __future__ import annotations

import csv
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dattri.benchmark.models.MusicTransformer.dataset.e_piano import (
    create_epiano_datasets,
)
from dattri.benchmark.models.MusicTransformer.music_transformer import MusicTransformer
from dattri.benchmark.models.MusicTransformer.utilities.constants import (
    ADAM_BETA_1,
    ADAM_BETA_2,
    ADAM_EPSILON,
    LR_DEFAULT_START,
    PREPEND_ZEROS_WIDTH,
    SCHEDULER_WARMUP_STEPS,
    TOKEN_PAD,
)
from dattri.benchmark.models.MusicTransformer.utilities.device import (
    get_device,
    use_cuda,
)
from dattri.benchmark.models.MusicTransformer.utilities.lr_scheduling import (
    LrStepTracker,
    get_lr,
)
from dattri.benchmark.models.MusicTransformer.utilities.run_model import (
    eval_model,
    train_epoch,
)

CSV_HEADER = [
    "Epoch",
    "Learn rate",
    "Avg Train loss",
    "Train Accuracy",
    "Avg Eval loss",
    "Eval accuracy",
]

# Baseline is an untrained epoch that we evaluate as a baseline loss and accuracy
BASELINE_EPOCH = -1

# Training Hyper-parameters ############
INPUT_DIR = "./dataset/e_piano"  # Folder of preprocessed and pickled midi files
OUTPUT_DIR = "./saved_models"  # Folder to save model weights
WEIGHT_MODULUS = 5  # Frequency to save epoch weights
PRINT_MODOLUS = 1  # Frequency to print train results for a batch
N_WORKERS = 1  # Number of threads for the dataloader
FORCE_CPU = False  # Forces model to run on a cpu even when gpu is available
NO_TENSORBOARD = True  # Turns off tensorboard result reporting
CONTINUE_WEIGHTS = None  # Model weights to continue training (str: file location)
CONTINUE_EPOCH = None  # Epoch the CONTINUE_WEIGHTS model was at (int: epoch number)
LR = None  # Set constant learn rate
CE_SMOOTHING = None  # Smoothing parameter for smoothed cross entropy loss
BATCH_SIZE = 64  # Batch size to use
EPOCHS = 50  # Number of epochs to use
RPR = True  # Use a modified Transformer for Relative Position Representations
MAX_SEQUENCE = 2048  # Maximum midi sequence to consider
N_LAYERS = 6  # Number of decoder layers to use
NUM_HEADS = 8  # Number of heads to use for multi-head attention
D_MODEL = 512  # Dimension of the model (output dim of embedding layers)
DIM_FEEDFORWARD = 1024  # Dimension of the feedforward layer
DROPOUT = 0.1  # Dropout rate
# Training Hyper-parameters ############


def create_loaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing.

    Returns:
        A tuple of three DataLoader objects for training, validation, and testing.
    """
    train_dataset, val_dataset, test_dataset = create_epiano_datasets(
        INPUT_DIR,
        MAX_SEQUENCE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=N_WORKERS,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=N_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=N_WORKERS,
    )

    return train_loader, val_loader, test_loader


def create_model() -> MusicTransformer:
    """Create a MusicTransformer model.

    # noqa: DAR201
    # noqa: DAR101
    """
    return MusicTransformer(
        n_layers=N_LAYERS,
        num_heads=NUM_HEADS,
        d_model=D_MODEL,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_sequence=MAX_SEQUENCE,
        rpr=RPR,
    ).to(get_device())


def create_optimizer_and_scheduler(
    model: nn.Module,
    train_loader: DataLoader,
) -> Tuple[Adam, LambdaLR]:
    """Create an optimizer and a learning rate scheduler.

    # noqa: DAR201
    # noqa: DAR101
    """
    if LR is None:
        init_step = 0 if CONTINUE_EPOCH is None else CONTINUE_EPOCH * len(train_loader)
        lr = LR_DEFAULT_START
        lr_stepper = LrStepTracker(D_MODEL, SCHEDULER_WARMUP_STEPS, init_step)
    else:
        lr = LR

    opt = Adam(
        model.parameters(),
        lr=lr,
        betas=(ADAM_BETA_1, ADAM_BETA_2),
        eps=ADAM_EPSILON,
    )

    lr_scheduler = LambdaLR(opt, lr_stepper.step) if LR is None else None

    return opt, lr_scheduler


# main
def train_maestro() -> MusicTransformer:  # noqa: PLR0914, PLR0915
    """Trains a model specified by default arguments.

    ----------
    Author: Damon Gwinn
    ----------
    # noqa: DAR201
    # noqa: DAR101
    """
    if FORCE_CPU:
        use_cuda(cuda_bool=False)
    Path(OUTPUT_DIR).mkdir(parents=True)

    # Output prep #####
    weights_folder = Path(OUTPUT_DIR) / "weights"
    Path(weights_folder).mkdir(parents=True)

    results_folder = Path(OUTPUT_DIR) / "results"
    Path(results_folder).mkdir(parents=True)

    results_file = Path(results_folder) / "results.csv"
    best_loss_file = Path(results_folder) / "best_loss_weights.pickle"
    best_acc_file = Path(results_folder) / "best_acc_weights.pickle"
    best_text = Path(results_folder) / "best_epochs.txt"

    # Tensorboard #####
    # Removed this session as we assume users will not launch TensorBoard

    # Datasets #####
    train_loader, _, test_loader = create_loaders()
    # Model #####
    model = create_model()
    # Lr Scheduler and Optimizer #####
    opt, lr_scheduler = create_optimizer_and_scheduler(model, train_loader)

    # Continuing from previous training session #####
    # Removed this session as we assume users will not load any
    # pretrained weights
    start_epoch = BASELINE_EPOCH

    # Not smoothing evaluation loss #####
    eval_loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)

    # SmoothCrossEntropyLoss or CrossEntropyLoss for training #####
    # Removed this session as we assume users will not use SmoothCrossEntropyLoss
    train_loss_func = eval_loss_func

    # Tracking best evaluation accuracy #####
    best_eval_acc = 0.0
    best_eval_acc_epoch = -1
    best_eval_loss = float("inf")
    best_eval_loss_epoch = -1

    # Results reporting #####
    if not Path.isfile(results_file):
        with Path.open(results_file, "w", newline="") as o_stream:
            writer = csv.writer(o_stream)
            writer.writerow(CSV_HEADER)

    # TRAIN LOOP #####
    # Removed the print statements!
    for epoch in range(start_epoch, EPOCHS):
        # Baseline has no training and acts as a base loss and accuracy
        if epoch > BASELINE_EPOCH:
            # Train
            train_epoch(
                epoch + 1,
                model,
                train_loader,
                train_loss_func,
                opt,
                lr_scheduler,
                PRINT_MODOLUS,
            )

        # Eval
        train_loss, train_acc = eval_model(model, train_loader, train_loss_func)
        eval_loss, eval_acc = eval_model(model, test_loader, eval_loss_func)

        # Learn rate
        lr = get_lr(opt)

        # Record the best model parameters
        new_best = False

        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            best_eval_acc_epoch = epoch + 1
            torch.save(model.state_dict(), best_acc_file)
            new_best = True

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_eval_loss_epoch = epoch + 1
            torch.save(model.state_dict(), best_loss_file)
            new_best = True

        # Writing out new bests
        if new_best:
            with Path.open(best_text, "w") as o_stream:
                o_stream.write(f"Best eval acc epoch: {best_eval_acc_epoch}\n")
                o_stream.write(f"Best eval acc: {best_eval_acc}\n\n")
                o_stream.write(f"Best eval loss epoch: {best_eval_loss_epoch}\n")
                o_stream.write(f"Best eval loss: {best_eval_loss}\n")

        if (epoch + 1) % WEIGHT_MODULUS == 0:
            epoch_str = str(epoch + 1).zfill(PREPEND_ZEROS_WIDTH)
            path = Path(weights_folder) / "epoch_" + epoch_str + ".pickle"
            torch.save(model.state_dict(), path)

        with Path.open(results_file, "a", newline="", encoding="utf-8") as o_stream:
            writer = csv.writer(o_stream)
            writer.writerow([epoch + 1, lr, train_loss, train_acc, eval_loss, eval_acc])

    # Sanity check just to make sure everything is gone
    # Removed this session since users will not open TensorBoard

    return model
