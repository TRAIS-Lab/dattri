"""This module contains functions for model training on MAESTRO dataset."""
# reference: https://github.com/gwinndr/MusicTransformer-Pytorch
# author: Damon Gwinn

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple

    from torch.utils.data import DataLoader
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from dattri.benchmark.models.MusicTransformer.music_transformer import MusicTransformer
from dattri.benchmark.models.MusicTransformer.utilities.constants import (
    ADAM_BETA_1,
    ADAM_BETA_2,
    ADAM_EPSILON,
    LR_DEFAULT_START,
    SCHEDULER_WARMUP_STEPS,
    TOKEN_PAD,
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
WEIGHT_MODULUS = 5  # Frequency to save epoch weights
PRINT_MODOLUS = 1  # Frequency to print train results for a batch
FORCE_CPU = False  # Forces model to run on a cpu even when gpu is available
NO_TENSORBOARD = True  # Turns off tensorboard result reporting
CONTINUE_WEIGHTS = None  # Model weights to continue training (str: file location)
CONTINUE_EPOCH = None  # Epoch the CONTINUE_WEIGHTS model was at (int: epoch number)
LR = 1e-4  # Set constant learn rate
CE_SMOOTHING = None  # Smoothing parameter for smoothed cross entropy loss
EPOCHS = 20  # Number of epochs to use
RPR = True  # Use a modified Transformer for Relative Position Representations
MAX_SEQUENCE = 256  # Maximum midi sequence to consider
N_LAYERS = 6  # Number of decoder layers to use
NUM_HEADS = 8  # Number of heads to use for multi-head attention
D_MODEL = 512  # Dimension of the model (output dim of embedding layers)
DIM_FEEDFORWARD = 1024  # Dimension of the feedforward layer
DROPOUT = 0.1  # Dropout rate
# Training Hyper-parameters ############


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
def train_maestro_musictransformer(
    train_dataloader: DataLoader,
    seed: int = 0,
    num_epoch: int = EPOCHS,
    device: str = "cpu",
) -> MusicTransformer:
    """Train a MusicTransformer on the MAESTRO dataset.

    Args:
        train_dataloader (DataLoader): The dataloader to train
            on MAESTRO dataset.
        seed (int): The seed to train the model.
        num_epoch (int): The number of epochs to train the model.
        device: The device to evaluate the model on.

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
    model = create_musictransformer_model(device)
    model.train()
    model.to(device)
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
    for epoch in range(start_epoch, num_epoch):
        # Baseline has no training and acts as a base loss and accuracy
        if epoch > BASELINE_EPOCH:
            # Train
            train_epoch(
                epoch + 1,
                model,
                train_dataloader,
                train_loss_func,
                opt,
                device,
                lr_scheduler,
                PRINT_MODOLUS,
            )

        # Eval
        # removed train_loss, train_acc
        # evaluate on train_dataloader instead
        eval_loss, eval_acc = eval_model(
            model,
            train_dataloader,
            eval_loss_func,
            device,
        )

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


def loss_maestro_musictransformer(
    model_path: str,
    dataloader: DataLoader,
    device: str = "cpu",
) -> float:
    """Calculate the loss of the MusicTransformer on the MAESTRO dataset.

    Args:
        model_path (str): The path to the saved model weights.
        dataloader (DataLoader): The dataloader for the MAESTRO dataset.
        device: The device to evaluate the model on.

    Returns:
        The sum of loss of the model on the loader.
    """
    model = create_musictransformer_model()
    model.load_state_dict(torch.load(Path(model_path)))
    model.eval()
    model.to(device)
    loss_list = []
    # Not smoothing evaluation loss #####
    eval_loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD, reduction="none")

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            tgt = batch[1].to(device)

            y = model(x)

            tgt = tgt[:, -1]
            y = y[:, -1:, :]
            loss = -eval_loss_func(y.squeeze(1), tgt)
            loss_list.append(loss.clone().detach().cpu())

    return torch.cat(loss_list)


def create_musictransformer_model(device: str = "cpu") -> MusicTransformer:
    """Create a MusicTransformer model.

    Args:
        device: The device to evaluate the model on.

    Returns:
        The MusicTransformer Model.
    """
    return MusicTransformer(
        n_layers=N_LAYERS,
        num_heads=NUM_HEADS,
        d_model=D_MODEL,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_sequence=MAX_SEQUENCE,
        rpr=RPR,
        device=device,
    )
