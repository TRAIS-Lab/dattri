"""This example shows how to use the IF to detect noisy labels in the MNIST."""

import argparse
import time
from functools import partial

import torch
from torch import nn

from dattri.algorithm.influence_function import (
    IFAttributorArnoldi,
    IFAttributorCG,
    IFAttributorDataInf,
    IFAttributorExplicit,
    IFAttributorLiSSA,
)
from dattri.benchmark.datasets.mnist import create_mnist_dataset, train_mnist_lr
from dattri.benchmark.utils import SubsetSampler, flip_label
from dattri.task import AttributionTask

ATTRIBUTOR_MAP = {
    "explicit": partial(IFAttributorExplicit, regularization=0.01),
    "cg": partial(IFAttributorCG, regularization=0.01),
    "lissa": partial(IFAttributorLiSSA, recursion_depth=100),
    "datainf": partial(IFAttributorDataInf, regularization=0.01),
    "arnoldi": partial(IFAttributorArnoldi, regularization=0.01, max_iter=10),
}


def get_mnist_indices_and_adjust_labels(dataset):
    dataset.targets, flip_index = flip_label(dataset.targets, p=0.1)
    return flip_index


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="explicit")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    dataset, _ = create_mnist_dataset("./data")

    flip_index = get_mnist_indices_and_adjust_labels(dataset)

    # for model training
    train_loader_full = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        sampler=SubsetSampler(range(1000)),
    )
    # training samples for attribution
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
        sampler=SubsetSampler(range(1000)),
    )
    # test samples for attribution
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
        sampler=SubsetSampler(range(1000)),
    )

    model = train_mnist_lr(train_loader_full)
    model.cuda()
    model.eval()

    def f(params, data_target_pair):
        image, label = data_target_pair
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model, params, image)
        return loss(yhat, label.long())

    task = AttributionTask(loss_func=f,
                           model=model,
                           checkpoints=model.state_dict())
    attributor = ATTRIBUTOR_MAP[args.method](
        task=task,
        device=args.device,
    )

    attributor.cache(train_loader_full)
    start_attribute = time.time()
    torch.cuda.reset_peak_memory_stats("cuda")
    with torch.no_grad():
        score = attributor.attribute(train_loader, test_loader).diag()
    peak_memory = torch.cuda.max_memory_allocated("cuda") / 1e6  # Convert to MB
    print(f"Peak memory usage: {peak_memory} MB")
    end_attribute = time.time()
    print("Attribution time: ", end_attribute - start_attribute)
    print(score.shape)
    _, indices = torch.sort(-score)

    cr = 0
    cr_list = []
    for idx, index in enumerate(indices):
        if idx % 100 == 0:
            cr_list.append((idx, cr))
        if int(index) in set(flip_index):
            cr += 1
    print(cr_list)
    print(f"{'Checked Data Sample':<25}{'Found flipped Sample':25}")
    print("-" * 50)

    for row in cr_list:
        print(f"{row[0]:<25}{row[1]:<25}")
