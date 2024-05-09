# reference: https://github.com/gwinndr/MusicTransformer-Pytorch
# author: Damon Gwinn
import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler
import math

from dattri.benchmark.models.MusicTransformer.utilities.constants import (
    TORCH_LABEL_TYPE,
    TOKEN_PAD,
    TOKEN_END,
    TORCH_FLOAT
)
from dattri.benchmark.models.MusicTransformer.utilities.device import cpu_device

SEQUENCE_START = 0

# EPianoDataset
class EPianoDataset(Dataset):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Pytorch Dataset for the Maestro e-piano dataset (https://magenta.tensorflow.org/datasets/maestro).
    Recommended to use with Dataloader (https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

    Uses all files found in the given root directory of pre-processed (preprocess_midi.py)
    Maestro midi files.
    # noqa: DAR201
    # noqa: DAR101
    ----------
    """

    def __init__(self, root, max_seq=2048, random_seq=True, full_version=False, sliding_windows_size=None):
        self.root       = root
        self.max_seq    = max_seq
        self.random_seq = random_seq

        # attributes for full version
        self.full_version = full_version
        self.dataset_length = 0
        self.length_dict = {}  # {idx: [file, offset]}
        if sliding_windows_size is None:
            sliding_windows_size = max_seq
        self.sliding_windows_size = sliding_windows_size

        fs = [os.path.join(root, f) for f in os.listdir(self.root)]
        self.data_files = [f for f in fs if os.path.isfile(f)]
        self.data_files = sorted(self.data_files)  # to make the order of index fixed.

        # full_version pre-process
        if self.full_version:
            idx = 0
            for file in self.data_files:
                # read the file
                i_stream    = open(file, "rb")
                raw_mid     = torch.tensor(pickle.load(i_stream), dtype=TORCH_LABEL_TYPE, device=cpu_device())
                i_stream.close()

                # calculate the fragment of each file
                length_item = raw_mid.shape[0]
                # fragment_num = max((length_item-1) // self.max_seq, 0)
                fragment_num = max(math.ceil((length_item-self.max_seq) / self.sliding_windows_size), 0)
                self.dataset_length += fragment_num
                for offset in range(0, length_item-self.max_seq, self.sliding_windows_size):
                    self.length_dict[idx] = [file, offset]
                    idx += 1
            assert self.dataset_length == len(self.length_dict)

    # __len__
    def __len__(self):
        """
        ----------
        Author: Damon Gwinn
        ----------
        How many data files exist in the given directory
        ----------
        # noqa: DAR201
        # noqa: DAR101
        """
        if self.full_version:
            return self.dataset_length
        return len(self.data_files)

    # __getitem__
    def __getitem__(self, idx):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Gets the indexed midi batch. Gets random sequence or from start depending on random_seq.

        Returns the input and the target.
        # noqa: DAR201
        # noqa: DAR101
        ----------
        """

        if self.full_version:
            filename, offset = self.length_dict[idx]
            x, tgt = self._get_x_tgt(filename, offset=offset)
        else:
            x, tgt = self._get_x_tgt(self.data_files[idx])

        return x, tgt

    def _get_x_tgt(self, filename, offset=0):
        # All data on cpu to allow for the Dataloader to multithread
        i_stream    = open(filename, "rb")
        # return pickle.load(i_stream), None
        raw_mid     = torch.tensor(pickle.load(i_stream), dtype=TORCH_LABEL_TYPE, device=cpu_device())
        i_stream.close()

        x, tgt = process_midi(raw_mid, self.max_seq, self.random_seq, offset=offset)

        return x, tgt


# process_midi
def process_midi(raw_mid, max_seq, random_seq, offset=None):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Takes in pre-processed raw midi and returns the input and target. Can use a random sequence or
    go from the start based on random_seq.
    ----------
    # noqa: DAR201
    # noqa: DAR101
    """

    x   = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=cpu_device())
    tgt = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=cpu_device())

    raw_len     = len(raw_mid)
    full_seq    = max_seq + 1 # Performing seq2seq

    if(raw_len == 0):
        return x, tgt

    if(raw_len < full_seq):
        x[:raw_len]         = raw_mid
        tgt[:raw_len-1]     = raw_mid[1:]
        tgt[raw_len-1]      = TOKEN_END
    else:
        # have a customized offset
        if offset:
            start = offset
        else:
            # Randomly selecting a range
            if(random_seq):
                end_range = raw_len - full_seq
                start = random.randint(SEQUENCE_START, end_range)

            # Always taking from the start to as far as we can
            else:
                start = SEQUENCE_START

        end = start + full_seq

        data = raw_mid[start:end]

        x = data[:max_seq]
        tgt = data[1:full_seq]

    return x, tgt


# create_epiano_datasets
def create_epiano_datasets(dataset_root, max_seq, random_seq=True, full_version=False, split=True, sliding_windows_size=None):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Creates train, evaluation, and test EPianoDataset objects for a pre-processed (preprocess_midi.py)
    root containing train, val, and test folders.
    ----------

    :param full_version: State if the whole dataset will be transversed. If set to True, random_seq` will
           be ignored and each training music will be cut to serveral fragment in order. Default to False.
    :param split: If false, it will not append "train"/"val"/"test" to the dataset_root and only return one
           dataset as return value.
    :param sliding_windows_size: sliding_windows_size
    # noqa: DAR201
    # noqa: DAR101
    """

    if not split:
        return EPianoDataset(dataset_root, max_seq, random_seq, full_version=full_version, sliding_windows_size=sliding_windows_size)

    train_root = os.path.join(dataset_root, "train")
    val_root = os.path.join(dataset_root, "val")
    test_root = os.path.join(dataset_root, "test")

    train_dataset = EPianoDataset(train_root, max_seq, random_seq, full_version=full_version, sliding_windows_size=sliding_windows_size)
    val_dataset = EPianoDataset(val_root, max_seq, random_seq, full_version=full_version, sliding_windows_size=sliding_windows_size)
    test_dataset = EPianoDataset(test_root, max_seq, random_seq, full_version=full_version, sliding_windows_size=sliding_windows_size)

    return train_dataset, val_dataset, test_dataset

# compute_epiano_accuracy
def compute_epiano_accuracy(out, tgt):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Computes the average accuracy for the given input and output batches. Accuracy uses softmax
    of the output.
    # noqa: DAR201
    # noqa: DAR101
    ----------
    """

    softmax = nn.Softmax(dim=-1)
    out = torch.argmax(softmax(out), dim=-1)

    out = out.flatten()
    tgt = tgt.flatten()

    mask = (tgt != TOKEN_PAD)

    out = out[mask]
    tgt = tgt[mask]

    # Empty
    if(len(tgt) == 0):
        return 1.0

    num_right = (out == tgt)
    num_right = torch.sum(num_right).type(TORCH_FLOAT)

    acc = num_right / len(tgt)

    return acc
