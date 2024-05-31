"""This example shows how to use the IF to detect noisy labels in the MNIST."""

import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Sampler
from torchvision import datasets, transforms

from dattri.algorithm.rps import RPSAttributor
from dattri.benchmark.datasets.mnist import train_mnist_lr
from dattri.benchmark.utils import flip_label

class SimpleDNN(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # Flatten the 28x28 input image
        self.dropout1 = nn.Dropout(dropout_rate) # 50% dropout
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_rate) # 50% dropout
        self.fc3 = nn.Linear(64, 10)      # 10 classes for MNIST digits

    def forward(self, x):
        x = x.view(-1, 28*28)            # Flatten the image
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def train_mnist_lr(
    dataloader: torch.utils.data.DataLoader,
    seed: int = 0,
    device: str = "cpu",
) -> nn.Module:
    """Train a logistic regression model on the MNIST dataset.

    Args:
        dataloader: The dataloader for the MNIST dataset.
        seed: The seed for training the model.
        device: The device to train the model on.

    Returns:
        The trained logistic regression model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)

    # get a more complext MLP model for MNIST
    model = SimpleDNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model.train()
    model.to(device)
    epoch_num = 20
    for _ in range(epoch_num):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

    return model


def get_mnist_indices_and_adjust_labels(dataset, subset_indice, p=0.1):
    dataset.targets, flip_index = flip_label(torch.tensor(dataset.targets)[subset_indice], p=p)
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


    subset_size = 1000
    train_loader_full = torch.utils.data.DataLoader(
        dataset,
        batch_size=subset_size,
        sampler=SubsetSampler(range(subset_size)),
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        sampler=SubsetSampler(range(subset_size)),
    )

    flip_index = get_mnist_indices_and_adjust_labels(dataset, range(subset_size))


    model = train_mnist_lr(train_loader_full)
    model.cuda()
    model.eval()


    # define the loss function
    def f(pre_activation_list, label_list):
        return torch.nn.functional.cross_entropy(pre_activation_list, label_list)


    model_params = {k: p for k, p in model.named_parameters() if p.requires_grad}
    attributor = RPSAttributor(
        loss_func=f,
        model=model,
        final_linear_layer_name="fc3",
        nomralize_preactivate=True,
        device=torch.device("cuda"),
    )

    attributor.cache(train_loader_full)
    start_attribute = time.time()
    torch.cuda.reset_peak_memory_stats("cuda")
    score = attributor.attribute(train_loader, train_loader).diag().abs()
    peak_memory = torch.cuda.max_memory_allocated("cuda") / 1e6  # Convert to MB
    print(f"Peak memory usage: {peak_memory} MB")
    end_attribute = time.time()
    print("Attribution time: ", end_attribute-start_attribute)
    print(score.shape)
    _, indices = torch.sort(-score)

    cr = 0
    cr_list = []
    for idx, index in enumerate(indices):
        if (idx+1) % 100 == 0:
            cr_list.append((idx+1, cr))
        if int(index) in set(flip_index):
            cr += 1
    print(cr_list)
    print(f"{'Checked Data Sample':<25}{'Found flipped Sample':25}")
    print("-" * 50)

    for row in cr_list:
        print(f"{row[0]:<25}{row[1]:<25}")
