"""This example shows how to use the IF to detect noisy labels in the MNIST."""

import argparse

import torch
from torch import nn

from dattri.algorithm.trak import TRAKAttributor
from dattri.benchmark.datasets.cifar import train_cifar_resnet9, create_cifar_dataset
from dattri.benchmark.utils import SubsetSampler, flip_label
from dattri.metric import mislabel_detection_auc
from dattri.task import AttributionTask


def get_cifar_indices_and_adjust_labels(dataset, subset_indice):
    dataset.targets, flip_index = flip_label(
        torch.tensor(dataset.targets)[subset_indice], p=0.1
    )
    return flip_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # create cifar 10 dataset
    dataset, _ = create_cifar_dataset("./data")

    flip_index = get_cifar_indices_and_adjust_labels(dataset, range(1000))

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        sampler=SubsetSampler(range(1000)),
    )
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        sampler=SubsetSampler(range(1000)),
    )

    model = train_cifar_resnet9(train_loader, num_epochs=3, num_classes=10)
    model.to(args.device)
    model.eval()

    def f(params, data_target_pair):
        image, label = data_target_pair
        image_t = image.unsqueeze(0)
        label_t = label.unsqueeze(0)
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model, params, image_t)
        logp = -loss(yhat, label_t)
        return logp - torch.log(1 - torch.exp(logp))

    def m(params, image_label_pair):
        image, label = image_label_pair
        image_t = image.unsqueeze(0)
        label_t = label.unsqueeze(0)
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model, params, image_t)
        p = torch.exp(-loss(yhat, label_t))
        return p

    task = AttributionTask(loss_func=f, model=model, checkpoints=model.state_dict())

    projector_kwargs = {
        "device": args.device,
    }

    attributor = TRAKAttributor(
        task=task,
        correct_probability_func=m,
        device=args.device,
        projector_kwargs=projector_kwargs,
    )

    attributor.cache(train_loader)
    with torch.no_grad():
        score = attributor.attribute(test_loader).diag()
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
