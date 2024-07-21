"""This example shows how to use the IF to detect noisy labels in the MNIST."""

import torch
from torch import nn

from dattri.algorithm.tracin import TracInAttributor
from dattri.benchmark.datasets.mnist import create_mnist_dataset, train_mnist_lr
from dattri.benchmark.utils import SubsetSampler, flip_label
from dattri.task import AttributionTask


def get_mnist_indices_and_adjust_labels(dataset, subset_indice):
    dataset.targets, flip_index = flip_label(dataset.targets[subset_indice], p=0.1)
    return flip_index


if __name__ == "__main__":
    dataset, _ = create_mnist_dataset("./data")

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

    def f(params, image_label_pair):
        image, label = image_label_pair
        image_t = image.unsqueeze(0)
        label_t = label.unsqueeze(0)
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model_1, params, image_t)
        return loss(yhat, label_t)

    task = AttributionTask(
        loss_func=f,
        model=model_1,
        checkpoints=model_1.state_dict(),
    )

    # simulate checkpoints
    attributor = TracInAttributor(
        task=task,
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
