"""TF-IDF Subset Sampler Module.

This module provides function called tfidf_subset_sampler
to sample subsets based on TF-IDF similarity and save the filterd train data
and return the indices of orignial train set.
The indices in return value represent the indices of blocks.
"""

# ruff: noqa: S301, S403

import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(data_path: str) -> tuple:
    """Load metadata, training, and validation data."""
    data_path = Path(data_path)
    meta_path = data_path / "meta.pkl"
    train_bin_path = data_path / "train.bin"
    val_bin_path = data_path / "val.bin"

    with meta_path.open("rb") as f:
        meta = pickle.load(f)
        if not isinstance(meta, dict) or "itos" not in meta or "stoi" not in meta:
            error = "Invalid metadata format"
            raise ValueError(error)

    encoded_train_data = np.fromfile(train_bin_path, dtype=np.uint16)
    encoded_val_data = np.fromfile(val_bin_path, dtype=np.uint16)

    return meta, encoded_train_data, encoded_val_data


def decode_data(encoded_data: np.ndarray, itos: dict) -> str:
    """Decode the encoded data using the provided itos mapping."""
    return "".join(itos[i] for i in encoded_data)


def split_blocks(data: str, block_size: int) -> list:
    """Split data into blocks of specified size."""
    return [data[i:i + block_size]
        for i in range(0, len(data) - block_size, block_size)]


def compute_similarity(train_blocks: list, val_block: str, subset_num: int) -> tuple:
    """Compute cosine similarity between train blocks and a validation block."""
    vectorizer = TfidfVectorizer()
    train_tfidf_matrix = vectorizer.fit_transform(train_blocks)
    val_tfidf = vectorizer.transform([val_block])
    cos_sim = cosine_similarity(val_tfidf, train_tfidf_matrix)
    selected_indices = np.argsort(-cos_sim[0])[:subset_num]
    return selected_indices, train_blocks


def tfidf_subset_sampler(data_path: str,
                         val_block_position: int,
                         block_size: int,
                         subset_num: int) -> list:
    """Sample a subset of TF-IDF blocks based on similarity and save to given path.

    Args:
        data_path (str): path to directory containing and to save the data files.
        val_block_position (int): Position index of the validation block.
        block_size (int): Size of each text block.
        subset_num (int): Number of top similar blocks to return.
    """
    meta, encoded_train_data, encoded_val_data = load_data(data_path)
    itos = meta["itos"]
    stoi = meta["stoi"]

    decoded_train_data = decode_data(encoded_train_data, itos)
    decoded_val_data = decode_data(encoded_val_data, itos)

    train_blocks = split_blocks(decoded_train_data, block_size)
    val_blocks = split_blocks(decoded_val_data, block_size)

    val_block = val_blocks[val_block_position]

    selected_indices, train_blocks = compute_similarity(train_blocks,
                                                        val_block,
                                                        subset_num)
    similar_train_blocks = [train_blocks[i] for i in selected_indices]

    output_bin_path = Path(data_path) / "filtered_train.bin"
    reencoded_chars = [
        stoi[char]
        for block in similar_train_blocks
        for char in block
    ]
    filtered_train_data = np.array(reencoded_chars, dtype=np.uint16)
    filtered_train_data.tofile(output_bin_path)

    return selected_indices
