"""This example shows how to use the IF to detect noisy labels in the MNIST."""

import argparse

import torch
from torch import nn

from dattri.algorithm.tracin import TracInAttributor
from dattri.benchmark.datasets.mnist import create_mnist_dataset, train_mnist_mlp
from dattri.benchmark.utils import SubsetSampler, flip_label
from dattri.metric import mislabel_detection_auc
from dattri.task import AttributionTask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # load the training dataset
    dataset, _ = create_mnist_dataset("./data")

    # flip 10% of the first 1000 data points
    dataset.targets[:1000], flip_index = flip_label(dataset.targets[:1000], p=0.1)

    # for model training, batch size is 64
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        sampler=SubsetSampler(range(1000)),
    )

    # training samples for attribution
    # batch size is 1000 to speed up the process
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1000,
        sampler=SubsetSampler(range(1000)),
    )

    model = train_mnist_mlp(train_loader, epoch_num=20)
    model.to(args.device)
    model.eval()

    def f(params, image_label_pair):
        image, label = image_label_pair
        image_t = image.unsqueeze(0)
        label_t = label.unsqueeze(0)
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model, params, image_t)
        return loss(yhat, label_t)

    task = AttributionTask(
        loss_func=f,
        model=model,
        checkpoints=model.state_dict(),
    )

    # simulate checkpoints
    attributor = TracInAttributor(
        task=task,
        weight_list=torch.tensor([0.01]),
        normalized_grad=False,
        device=args.device,
    )

    with torch.no_grad():
        score = attributor.attribute(train_loader, test_loader).diag()

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
    print("-" * 50)

    # calculate the AUC
    ground_truth = torch.zeros(1000)
    ground_truth[flip_index] = 1
    print("AUC: ", float(mislabel_detection_auc(score, ground_truth)[0]))
