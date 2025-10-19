import torch
import torch.nn as nn
import os

from tqdm import tqdm
import numpy as np

from scipy.stats import spearmanr
import csv
import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--score_path",
        type=str,
        default=None,
        help="The path of the score file to be evaluated",
    )
    args = parser.parse_args()
    if args.score_path is None:
        raise ValueError("Need the path of the score file.")
    return args


def read_nodes(file_path):
    int_list = []
    with open(file_path, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            for item in row:
                try:
                    int_list.append(int(item))
                except ValueError:
                    print(
                        f"Warning: '{item}' could not be converted to an integer and was skipped."
                    )
    return int_list


def calculate_one(path):

    # score = torch.load(path, map_location=torch.device('cpu'))  # _test_0225_regroup
    score = torch.load(path, map_location=torch.device("cpu"))
    # score = torch.rand(5000, 500)
    print("score shape:", score.shape)

    nodes_str = []
    for i in range(50):
        nodes_str.append(f"./checkpoints/{i}/train_index.csv")

    full_nodes = [i for i in range(4656)]

    node_list = []
    for node_str in nodes_str:
        numbers = read_nodes(node_str)
        index = []
        for number in numbers:
            index.append(full_nodes.index(number))
        node_list.append(index)

    loss_list = torch.load("gt.pt", map_location=torch.device("cpu")).detach()

    approx_output = []
    for i in range(len(nodes_str)):
        score_approx_0 = score[node_list[i], :]
        sum_0 = torch.sum(score_approx_0, axis=0)
        approx_output.append(sum_0)

    print(len(loss_list), loss_list[0].shape)
    print(len(approx_output), approx_output[0].shape)

    res = 0
    counter = 0
    for i in range(481):
        tmp = spearmanr(
            np.array([approx_output[k][i] for k in range(len(approx_output))]),
            np.array([loss_list[k][i].numpy() for k in range(len(loss_list))]),
        ).statistic
        if np.isnan(tmp):
            print("Numerical issue")
            continue
        res += tmp
        counter += 1

    print(counter)

    return res / counter, loss_list, approx_output


if __name__ == "__main__":
    args = parse_args()
    if args.score_path:
        path = args.score_path
    print(calculate_one(path)[0])
