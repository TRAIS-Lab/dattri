"""This module contains functions for GPT2 training/evaluation on the MNIST dataset."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dattri.benchmark.models.gpt2 import (
    create_gpt2_model,
)


def train_wikitext2_gpt2(
    dataloader: torch.utils.data.DataLoader,
    seed: int = 0,
    device: str = "cpu",
    epoch_num: int = 3,
) -> nn.Module:
    """Train a gpt2  model on the wikitext dataset.

    Args:
        dataloader: The dataloader for the wikitext dataset.
        seed: The seed for training the model.
        device: The device to train the model on.
        epoch_num: The number of epochs to train the model.

    Returns:
        The trained gpt2 model.
    """
    torch.manual_seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)

    # model, optimizer
    model = create_gpt2_model()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
    )

    model.train()
    model.to(device)
    for _ in range(epoch_num):
        for batch in dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return model


def loss_wikitext2_gpt2(
    model_path: str,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> float:
    """Calculate the loss of the gpt2 model on the wikitext dataset.

    Args:
        model_path: The path to the saved model weights.
        dataloader: The dataloader for the wikitext dataset.
        device: The device to evaluate the model on.

    Returns:
        float: The per-example loss of the model on the loader.
    """
    model = create_gpt2_model()
    model.load_state_dict(torch.load(Path(model_path)))
    model.eval()
    model.to(device)
    loss_list = []

    # loss
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            loss_list.append(loss.clone().detach().cpu().unsqueeze(0))
    return torch.cat(loss_list)
