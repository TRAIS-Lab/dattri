"""This module contains function for creating wikitext2 dataset."""

# Code adapted from
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py.

from __future__ import annotations

import logging
from itertools import chain
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    import torch


def create_wikitext2_dataset(
    block_size: int = 512,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Create tokenized WikiText datasets for GPT-style language modeling.

    Args:
        block_size: Length of each sequence block after tokenization.

    Returns:
        (train_dataset, eval_dataset): tokenized torch datasets.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # dataset and tokenizer
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
    if "validation" not in raw_datasets:
        raw_datasets["validation"] = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train[:5%]",
        )
        raw_datasets["train"] = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train[5%:]",
        )
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", use_fast=True)

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # tokenize all text samples
    def tokenize_function(examples: dict) -> dict:
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    # check block size
    if block_size > tokenizer.model_max_length:
        logging.Logger.warning(
            f"Block size ({block_size}) is larger than the model's maximum length."
            f"(Maximum: ({tokenizer.model_max_length}))"
            f"Using block_size={tokenizer.model_max_length}.",
        )
    block_size = min(block_size, tokenizer.model_max_length)

    def group_texts(examples: dict) -> dict:
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
        total_length = len(concatenated_examples[next(iter(examples.keys()))])
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=None,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    return train_dataset, eval_dataset
