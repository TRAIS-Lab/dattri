"""This module contains functions for model training/evaluation on the wikitext dataset."""

__all__ = ["create_wikitext2_dataset", "create_wikitext_datasettrain_wikitext_gpt2loss_wikitext_gpt2", "loss_wikitext2_gpt2", "train_wikitext2_gpt2"]

from .data import create_wikitext2_dataset
from .wikitext2_gpt2 import loss_wikitext2_gpt2, train_wikitext2_gpt2
