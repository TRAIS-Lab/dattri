"""This example shows how to use the IF to detect low-quality data in the MNIST."""

import argparse
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
from dattri.benchmark.utils import SubsetSampler
from dattri.metric import mislabel_detection_auc
from dattri.task import AttributionTask

ATTRIBUTOR_MAP = {
    "explicit": partial(IFAttributorExplicit, regularization=0.01),
    "cg": partial(IFAttributorCG, regularization=0.01),
    "lissa": partial(IFAttributorLiSSA, recursion_depth=100),
    "datainf": partial(IFAttributorDataInf, regularization=0.01),
    "arnoldi": partial(IFAttributorArnoldi, regularization=0.01, max_iter=10),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="explicit")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_size", type=int, default=10000)
    parser.add_argument("--val_size", type=int, default=10000)
    parser.add_argument("--test_size", type=int, default=5000)
    parser.add_argument("--remove_number", type=int, default=100)
    args = parser.parse_args()

    print(args)

    # load the training dataset
    dataset, dataset_test = create_mnist_dataset("./data")

    # for model training, batch size is 64
    train_loader_full = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        sampler=SubsetSampler(range(args.train_size)),
    )

    # training samples for attribution
    # batch size is 1000 to speed up the process
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1000,
        sampler=SubsetSampler(range(args.train_size)),
    )
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1000,
        sampler=SubsetSampler(range(args.train_size, args.train_size + args.val_size)),
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1000,
        sampler=SubsetSampler(range(args.test_size)),
    )

    model = train_mnist_lr(train_loader_full, seed=args.seed)
    model.to(args.device)
    model.eval()

    def f(params, data_target_pair):
        image, label = data_target_pair
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model, params, image)
        return loss(yhat, label.long())

    task = AttributionTask(loss_func=f, model=model, checkpoints=model.state_dict())

    attributor = ATTRIBUTOR_MAP[args.method](
        task=task,
        device=args.device,
    )

    attributor.cache(train_loader)
    with torch.no_grad():
        score = attributor.attribute(train_loader, val_loader)

    score_overall = score.mean(dim=1)
    value, indices = torch.sort(score_overall, descending=True)

    # calculate the test loss
    loss = nn.CrossEntropyLoss(reduction="none")
    loss_value = []
    for image, label in test_loader:
        image, label = image.to(args.device), label.to(args.device)
        yhat = model(image)
        loss_value.append(loss(yhat, label.long()))
    loss_value = torch.mean(torch.cat(loss_value))

    print(f"Test loss: {loss_value}")

    # calculate the accuracy
    correct = 0
    total = 0
    for image, label in test_loader:
        image, label = image.to(args.device), label.to(args.device)
        yhat = model(image)
        _, predicted = torch.max(yhat.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    print(f"Accuracy: {100 * correct / total}")

    REMOVE_NUMBER = args.remove_number
    new_training_set = list(range(args.train_size))
    new_training_set = [i for i in new_training_set if i not in indices[-REMOVE_NUMBER:].tolist()]
    train_loader_full = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        sampler=SubsetSampler(new_training_set),
    )
    model = train_mnist_lr(train_loader_full, seed=args.seed)
    model.to(args.device)
    model.eval()

    # calculate the test loss
    loss = nn.CrossEntropyLoss(reduction="none")
    loss_value = []
    for image, label in test_loader:
        image, label = image.to(args.device), label.to(args.device)
        yhat = model(image)
        loss_value.append(loss(yhat, label.long()))
    loss_value = torch.mean(torch.cat(loss_value))

    print(f"Test loss (after): {loss_value}")

    # calculate the accuracy
    correct = 0
    total = 0
    for image, label in test_loader:
        image, label = image.to(args.device), label.to(args.device)
        yhat = model(image)
        _, predicted = torch.max(yhat.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    print(f"Accuracy  (after): {100 * correct / total}")
