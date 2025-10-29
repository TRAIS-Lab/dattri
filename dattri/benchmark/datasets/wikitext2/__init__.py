"""This module contains functions for model training/evaluation on wikitext2."""

__all__ = [
    "create_wikitext2_dataset",
    "loss_wikitext2_gpt2",
    "train_wikitext2_gpt2",
]

from .data import create_wikitext2_dataset
from .wikitext2_gpt2 import loss_wikitext2_gpt2, train_wikitext2_gpt2
