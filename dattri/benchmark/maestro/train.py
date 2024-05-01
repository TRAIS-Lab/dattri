"""This module contains functions for model training on MAESTRO dataset."""
# reference: https://github.com/gwinndr/MusicTransformer-Pytorch
# author: Damon Gwinn

from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SubsetRandomSampler

from dattri.benchmark.models.MusicTransformer.dataset.e_piano import (
    create_epiano_datasets,
)
from dattri.benchmark.models.MusicTransformer.music_transformer import MusicTransformer
from dattri.benchmark.models.MusicTransformer.utilities.constants import (
    ADAM_BETA_1,
    ADAM_BETA_2,
    ADAM_EPSILON,
    LR_DEFAULT_START,
    SCHEDULER_WARMUP_STEPS,
    TOKEN_PAD,
)
from dattri.benchmark.models.MusicTransformer.utilities.device import (
    get_device,
)
from dattri.benchmark.models.MusicTransformer.utilities.lr_scheduling import (
    LrStepTracker,
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
INPUT_DIR = "./maestro-v2.0.0-processed"  # Folder of preprocessed midi files
OUTPUT_DIR = "./maestro_model"  # Folder to save model weights
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
EPOCHS = 20  # Number of epochs to use
RPR = True  # Use a modified Transformer for Relative Position Representations
MAX_SEQUENCE = 256  # Maximum midi sequence to consider
N_LAYERS = 6  # Number of decoder layers to use
NUM_HEADS = 8  # Number of heads to use for multi-head attention
D_MODEL = 512  # Dimension of the model (output dim of embedding layers)
DIM_FEEDFORWARD = 1024  # Dimension of the feedforward layer
DROPOUT = 0.1  # Dropout rate
# Training Hyper-parameters ############


def create_loaders(
    train_size: int = 5000,
    test_size: int = 500,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing.

    Args:
        train_size (int): the train loader size.
        test_size (int): the val/test loader size.
        seed (int): the random seed used to sample training examples.

    Returns:
        A tuple of three DataLoader objects for training.
    """
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = create_epiano_datasets(
        INPUT_DIR,
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
        sampler=SubsetRandomSampler(range(test_size)),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=N_WORKERS,
        sampler=SubsetRandomSampler(range(test_size)),
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
def train_maestro(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    seed: int = 0,
) -> MusicTransformer:
    """Train a MusicTransformer on the MAESTRO dataset.

    Args:
        train_dataloader (DataLoader): The dataloader to train
            on MAESTRO dataset.
        val_dataloader (DataLoader): The dataloader to validate
            on MAESTRO dataset.
        seed (int): The seed for training the model.

    Returns:
        The trained MusicTransformer model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)

    # # Output prep #####
    # Removed weight folder creation
    # Removed results folder creation
    # Removed best_loss_file, best_acc_file, best_text write out

    # Tensorboard #####
    # Removed this session as we assume users will not launch TensorBoard

    # Datasets #####
    # Removed this session as we make dataset as input to the function

    # Model #####
    model = create_model()
    # Lr Scheduler and Optimizer #####
    opt, lr_scheduler = create_optimizer_and_scheduler(model, train_dataloader)

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
    best_eval_loss = float("inf")

    # Results reporting #####
    # Removed results_file writeout section

    # TRAIN LOOP #####
    for epoch in range(start_epoch, EPOCHS):
        # Baseline has no training and acts as a base loss and accuracy
        if epoch > BASELINE_EPOCH:
            # Train
            train_epoch(
                epoch + 1,
                model,
                train_dataloader,
                train_loss_func,
                opt,
                lr_scheduler,
                PRINT_MODOLUS,
            )

        # Eval
        # Removed train_loss, train_acc
        eval_loss, eval_acc = eval_model(model, val_dataloader, eval_loss_func)

        # Learn rate
        # Removed lr = get_lr(opt)

        # Record the best model parameters
        new_best = False

        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            new_best = True

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            new_best = True

        # Writing out new bests
        if new_best:
            best_model = model

        # Removed best model write out section

        # Removed WEIGHT_MODULUS store model section
        # Removed results_file write out section

    # Sanity check just to make sure everything is gone
    # Removed this session since users will not open TensorBoard

    return best_model


def eval_maestro(
    model_path: str,
    dataloader: DataLoader,
) -> float:
    """Calculate the loss of the MusicTransformer on the MAESTRO dataset.

    Args:
        model_path: The path to the saved model weights.
        dataloader: The dataloader for the MAESTRO dataset.

    Returns:
        The sum of loss of the model on the loader.
    """
    model = create_model()
    model.load_state_dict(torch.load(Path(model_path), get_device()))
    # Not smoothing evaluation loss #####
    eval_loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)
    eval_loss, _ = eval_model(model, dataloader, eval_loss_func)

    return eval_loss
