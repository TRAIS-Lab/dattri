"""This example shows how to use the IF to detect noisy labels in the MNIST."""

import torch
from torch import nn
from torch.utils.data import Sampler
from torchvision import datasets, transforms

from dattri.algorithm.influence_function import IFAttributor
from dattri.benchmark.datasets.mnist import train_mnist_lr
from dattri.benchmark.utils import flip_label
from dattri.func.utils import flatten_func


def get_mnist_indices_and_adjust_labels(dataset):
    dataset.targets, flip_index = flip_label(dataset.targets, p=0.1)
    return flip_index


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


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
        batch_size=1000,
        sampler=SubsetSampler(range(1000)),
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=SubsetSampler(range(1000)),
    )
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=SubsetSampler(range(1000)),
    )

    model = train_mnist_lr(train_loader_full)
    model.cuda()
    model.eval()

    @flatten_func(model, param_num=0)
    def f(params, dataloader):
        loss = nn.CrossEntropyLoss()
        loss_val = 0
        for image, label in dataloader:
            yhat = torch.func.functional_call(model, params, image)
            loss_val += loss(yhat, label)
        return loss_val

    model_params = {k: p for k, p in model.named_parameters() if p.requires_grad}
    attributor = IFAttributor(
        target_func=f,
        params=model_params,
        ihvp_solver="cg",
        ihvp_kwargs={"regularization": 1e-3},
        device=torch.device("cuda"),
    )

    attributor.cache(train_loader_full)
    torch.cuda.reset_peak_memory_stats("cuda")
    with torch.no_grad():
        score = attributor.attribute(train_loader, test_loader).diag()
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
