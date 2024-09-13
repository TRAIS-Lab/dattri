"""This experiment brittleness TDA methods on the MNIST-10 dataset."""

# ruff: noqa
import time
import argparse
from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dattri.algorithm.influence_function import (
    IFAttributorExplicit,
    IFAttributorCG,
    IFAttributorLiSSA,
    IFAttributorDataInf,
    IFAttributorArnoldi,
)
from dattri.benchmark.datasets.mnist import train_mnist_lr, create_mnist_dataset
from dattri.benchmark.utils import SubsetSampler
from dattri.metric import brittleness
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
    args = parser.parse_args()

    dataset, _ = create_mnist_dataset("./data")

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

    attributor.cache(train_loader_full)
    with torch.no_grad():
        score = attributor.attribute(train_loader, test_loader)

    model = train_mnist_lr(train_loader)
    model.to(args.device)
    model.eval()
    correct_x = None
    correct_label = None
    correct_index = None

    # Find a correct predicted data
    for test_batch in test_loader:
        for i in range(len(test_batch[0])):
            i = i
            x = test_batch[0][i].unsqueeze(0).to(args.device)
            label = test_batch[1][i].to(args.device)
            output = model(x)
            pred = output.argmax(dim=1, keepdim=True)

            if pred.item() == label.item():
                print(
                    f"Found a correctly predicted test sample with correct index is: {i}"
                )
                print(f"The label: {label.item()}, the prediction: {pred.item()}")
                correct_x = x.unsqueeze(0)
                correct_label = label.unsqueeze(0)
                correct_index = i
                break

        if correct_x is not None:
            break

    # Example test sample
    test_dataset = TensorDataset(correct_x, correct_label)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    score = score[:, correct_index]

    # Compute brittleness
    start_time = time.time()

    # Evaluation
    def eval_func(model, test_loader, device="cpu"):
        model.to(device)
        model.eval()
        all_predictions = []

        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                model_output = model(data)
                probabilities = torch.softmax(model_output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
                all_predictions.append(predicted_class.cpu())

        return torch.cat(all_predictions)

    smallest_k = brittleness(
        train_loader=train_loader,
        test_loader=test_loader,
        scores=score,
        train_func=lambda loader: train_mnist_lr(loader),
        eval_func=eval_func,
        device=args.device,
        search_space=[20, 40, 60, 80, 100, 120, 140, 180],
    )
    if smallest_k is not None:
        print(f"The number of removal that can make it flip: {smallest_k}")
    else:
        print("No k found that causes a flip.")
    end_time = time.time()
    print("Total time for brittleness test: ", end_time - start_time)
