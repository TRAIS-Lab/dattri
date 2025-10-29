# Code is adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py .

from __future__ import annotations

import logging
import random
from itertools import chain
from typing import Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def create_wikitext2_dataset(
    block_size: int = 512,
    subset_ratio: float = 1,  # default half dataset
    seed: int = 0,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Create tokenized WikiText datasets for GPT-style language modeling.

    Args:
        block_size: Length of each sequence block after tokenization.
        subset_ratio: Fraction of the training set to use (for quick experiments).
        seed: Random seed for reproducibility.

    Returns:
        (train_dataset, eval_dataset): tokenized torch datasets.
    """
    # dataset and tokenizer
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # tokenize all text samples
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    # check block size
    if block_size > tokenizer.model_max_length:
        logging.Logger.warning(
            f"The block_size passed ({block_size}) is larger than the maximum length for the model "
            f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}.",
        )
    block_size = min(block_size, tokenizer.model_max_length)

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=4,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # in default cut to half.
    if subset_ratio < 1.0:
        random.seed(seed)
        subset_size = int(subset_ratio * len(train_dataset))
        indices = random.sample(range(len(train_dataset)), subset_size)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    return train_dataset, eval_dataset
