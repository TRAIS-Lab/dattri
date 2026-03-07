"""This example shows how to use TracInAttributor with DataloaderGroup and
group_target_func=True so target_func is used for group attribution.
Uses MNIST + MLP.
"""

import argparse
from typing import Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader

from dattri.algorithm.tracin import TracInAttributor
from dattri.benchmark.datasets.mnist import create_mnist_dataset, train_mnist_mlp
from dattri.benchmark.utils import SubsetSampler
from dattri.task import AttributionTask


class DataloaderGroup(DataLoader):
    """Helper class to wrap a DataLoader for group attribution.

    This wrapper presents the dataloader as a single item (length 1).
    When iterated, it yields the original dataloader itself, allowing the
    consumer to treat the entire dataset as one attribution target.
    """

    def __init__(self, original_test_dataloader: DataLoader) -> None:
        """Initialize the DataloaderGroup.

        Args:
            original_test_dataloader (DataLoader):
                The PyTorch dataloader for individual test data samples
        """
        super().__init__(torch.utils.data.TensorDataset(torch.zeros(1)))
        self.original_test_dataloader = original_test_dataloader

    def __iter__(self) -> Iterator[DataLoader]:
        """Iterate over the group.

        Yields:
            DataLoader: Yields the original dataloader as a single object.
        """
        yield self.original_test_dataloader

    def __len__(self) -> int:
        """Return the length of the group wrapper.

        Returns:
            int: Always 1, as the whole dataset is treated as one group.
        """
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_size", type=int, default=10000)
    parser.add_argument("--test_size", type=int, default=5000)
    args = parser.parse_args()

    print(args)

    # load the training dataset (same as influence_function_data_cleaning.py)
    dataset, dataset_test = create_mnist_dataset("./data")

    # for model training, batch size is 64
    train_loader_full = DataLoader(
        dataset,
        batch_size=64,
        sampler=SubsetSampler(range(args.train_size)),
    )

    # training samples for attribution; batch size 1000 to speed up
    train_loader = DataLoader(
        dataset,
        batch_size=1000,
        sampler=SubsetSampler(range(args.train_size)),
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=1000,
        sampler=SubsetSampler(range(args.test_size)),
    )

    model = train_mnist_mlp(train_loader_full, seed=args.seed)
    model.to(args.device)
    model.eval()

    # loss and target in AttributionTask style; match IF example signature.
    # When group_target_func=True, target_func is also called with (params_dict, list_of_batches).
    def f(params, data_target_pair):
        image, label = data_target_pair
        label = label.view(-1).long()
        yhat = torch.func.functional_call(model, params, (image,))
        return nn.CrossEntropyLoss()(yhat, label)

    def target_func(params, data):
        if isinstance(data, list):
            # group mode: data is list of (image, label) batches
            device = next(iter(params.values())).device
            total = None
            for image, label in data:
                image, label = image.to(device), label.to(device)
                loss = f(params, (image, label))
                n = image.shape[0]
                total = loss * n if total is None else total + loss * n
            return total
        return f(params, data)

    task = AttributionTask(
        loss_func=f,
        model=model,
        checkpoints=model.state_dict(),
        target_func=target_func,
        group_target_func=True,
    )

    attributor = TracInAttributor(
        task=task,
        weight_list=torch.tensor([1.0]),
        normalized_grad=False,
        device=args.device,
    )

    test_group = DataloaderGroup(test_loader)
    with torch.no_grad():
        scores = attributor.attribute(train_loader, test_group)
        scores_temp = attributor.attribute(train_loader, test_loader)

    print("Test Dataloader Group (AttributionTask + group_target_func=True) — MNIST + MLP.")
    print(f"Score Shape: {scores.shape}")
    print(f"Calculated Scores (first 10):\n{scores.flatten()[:10]}")
    print(f"Calculated Scores Temp sum over test (first 10):\n{scores_temp.sum(dim=1)[:10]}")
    diff = (scores.flatten() - scores_temp.sum(dim=1)).abs()
    print(f"Max |group - sum(per-test)|: {diff.max().item():.6f}")
