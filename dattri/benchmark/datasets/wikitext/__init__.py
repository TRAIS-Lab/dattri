"""This module contains functions for model training/evaluation on the wikitext dataset."""

__all__ = ["create_wikitext_dataset" "train_wikitext_gpt2" "loss_wikitext_gpt2"]

from .data import create_wikitext_dataset
from .wikitext_gpt2 import train_wikitext_gpt2, loss_wikitext_gpt2
