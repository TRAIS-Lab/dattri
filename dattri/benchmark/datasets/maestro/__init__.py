"""This module contains functions for model training/evaluation on MAESTRO dataset."""

__all__ = [
    "create_maestro_datasets",
    "create_musictransformer_model",
    "loss_maestro_musictransformer",
    "train_maestro_musictransformer",
]


from .data import create_maestro_datasets
from .maestro_musictransformer import (
    create_musictransformer_model,
    loss_maestro_musictransformer,
    train_maestro_musictransformer,
)
