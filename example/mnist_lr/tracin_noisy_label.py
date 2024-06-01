"""This example shows how to use the IF to detect noisy labels in the MNIST."""

import torch
from torch import nn
from torch.utils.data import Sampler
from torchvision import datasets, transforms

from dattri.algorithm.tracin import TracInAttributor
from dattri.benchmark.datasets.mnist import train_mnist_lr
from dattri.benchmark.utils import flip_label
from dattri.func.utils import flatten_func


def get_mnist_indices_and_adjust_labels(dataset, subset_indice):
    dataset.targets, flip_index = flip_label(dataset.targets[subset_indice], p=0.1)
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

    flip_index = get_mnist_indices_and_adjust_labels(dataset, range(1000))

    # simulate checkpoints
    model_1 = train_mnist_lr(train_loader, epoch_num=20)

    # only need one model definition
    model_1.cuda()
    model_1.eval()


    @flatten_func(model_1)
    def f(params, image_label_pair):
        image, label = image_label_pair
        image_t = image.unsqueeze(0)
        label_t = label.unsqueeze(0)
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model_1, params, image_t)
        return loss(yhat, label_t)

    # simulate checkpoints
    model_params_1 = {k: p for k, p in model_1.named_parameters() if p.requires_grad}
    attributor = TracInAttributor(
        target_func=f,
        params_list=[model_params_1],
        weight_list=torch.tensor([0.01]),
        normalized_grad=False,
        device=torch.device("cuda"),
    )

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
