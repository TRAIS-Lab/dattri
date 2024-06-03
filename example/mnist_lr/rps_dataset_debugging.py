"""This example shows how to use the IF to detect noisy labels in the MNIST."""

import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Sampler
from torchvision import datasets, transforms

from dattri.algorithm.rps import RPSAttributor
from dattri.benchmark.datasets.mnist import train_mnist_lr
from dattri.benchmark.utils import flip_label, SubsetSampler



def get_mnist_indices_and_adjust_labels(dataset, subset_indice, p=0.1):
    dataset.targets, flip_index = flip_label(torch.tensor(dataset.targets)[subset_indice], p=p)
    return flip_index


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ],
    )
    dataset = datasets.MNIST("../data", train=True, download=True, transform=transform)


    subset_size = 1000
    train_loader_full = torch.utils.data.DataLoader(
        dataset,
        batch_size=subset_size,
        sampler=SubsetSampler(range(subset_size)),
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        sampler=SubsetSampler(range(subset_size)),
    )

    flip_index = get_mnist_indices_and_adjust_labels(dataset, range(subset_size))


    model = train_mnist_lr(train_loader_full)
    model.cuda()
    model.eval()


    # define the loss function
    def f(pre_activation_list, label_list):
        return torch.nn.functional.cross_entropy(pre_activation_list, label_list)

    # experiment hyper-param
    for l2 in [1]:
        model_params = {k: p for k, p in model.named_parameters() if p.requires_grad}
        attributor = RPSAttributor(
            loss_func=f,
            model=model,
            final_linear_layer_name="linear",
            nomralize_preactivate=False,
            l2_strength=l2,
            device=torch.device("cuda"),
        )

        attributor.cache(train_loader_full)
        start_attribute = time.time()
        torch.cuda.reset_peak_memory_stats("cuda")
        score = attributor.attribute(train_loader, train_loader).diag().abs()
        peak_memory = torch.cuda.max_memory_allocated("cuda") / 1e6  # Convert to MB
        print(f"Peak memory usage: {peak_memory} MB")
        end_attribute = time.time()
        print("Attribution time: ", end_attribute-start_attribute)
        print(score.shape)
        _, indices = torch.sort(-score)
        cr = 0
        cr_list = []
        for idx, index in enumerate(indices):
            if (idx+1) % 100 == 0:
                cr_list.append((idx+1, cr))
            if int(index) in set(flip_index):
                cr += 1
        print(cr_list)
        print(f"{'Checked Data Sample':<25}{'Found flipped Sample':25}")
        print("-" * 50)

        for row in cr_list:
            print(f"{row[0]:<25}{row[1]:<25}")
