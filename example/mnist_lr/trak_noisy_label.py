"""This example shows how to use the IF to detect noisy labels in the MNIST."""

import torch
from torch import nn
from torchvision import datasets, transforms

from dattri.algorithm.trak import TRAKAttributor
from dattri.benchmark.datasets.mnist import train_mnist_lr
from dattri.benchmark.utils import SubsetSampler, flip_label
from dattri.task import AttributionTask


def get_mnist_indices_and_adjust_labels(dataset):
    dataset.targets, flip_index = flip_label(dataset.targets, p=0.1)
    return flip_index


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ],
    )
    dataset = datasets.MNIST("../data", train=True, download=True, transform=transform)

    flip_index = get_mnist_indices_and_adjust_labels(dataset)

    train_loader_full = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        sampler=SubsetSampler(range(1000)),
    )
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

    model = train_mnist_lr(train_loader_full)
    model.cuda()
    model.eval()

    def f(params, image_label_pair):
        image, label = image_label_pair
        image_t = image.unsqueeze(0)
        label_t = label.unsqueeze(0)
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model, params, image_t)
        return loss(yhat, label_t)

    def m(params, image_label_pair):
        image, label = image_label_pair
        image_t = image.unsqueeze(0)
        label_t = label.unsqueeze(0)
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model, params, image_t)
        p = torch.exp(-loss(yhat, label_t))
        return p

    task = AttributionTask(loss_func=f,
                           model=model,
                           checkpoints=model.state_dict())

    projector_kwargs = {
        "device": "cuda",
        "use_half_precision": False,
    }

    attributor = TRAKAttributor(
        task=task,
        correct_probability_func=m,
        device="cuda",
        projector_kwargs=projector_kwargs,
    )

    attributor.cache(train_loader)
    torch.cuda.reset_peak_memory_stats("cuda")
    with torch.no_grad():
        score = attributor.attribute(test_loader).diag()
    peak_memory = torch.cuda.max_memory_allocated("cuda") / 1e6  # Convert to MB
    print(f"Peak memory usage: {peak_memory} MB")

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
