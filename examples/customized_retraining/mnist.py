import argparse
from functools import partial
import random

import torch
from torch import nn
import numpy as np

from dattri.model_util.retrain import retrain_lds
from dattri.metric.ground_truth import calculate_lds_ground_truth
from dattri.benchmark.utils import SubsetSampler
from dattri.task import AttributionTask
from dattri.metric import lds
from dattri.algorithm.influence_function import IFAttributorCG
from dattri.benchmark.datasets.mnist import create_mnist_dataset

# Here we define a simple MLP model to be trained on MNIST
# As a user, you can define any model you like
class MLPMnist(nn.Module):
    def __init__(self, dropout_rate: float = 0.1) -> None:
        super(MLPMnist, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 28*28)
        # x = self.fc1(x) # TODO: delete
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# This function initializes an MLP model and trains it on MNIST
# As a user, you can define the training process
def train_mnist_mlp(
    dataloader: torch.utils.data.DataLoader,
    seed: int = 0,
    device: str = "cpu",
    epoch_num: int = 50,
) -> MLPMnist:
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)

    model = MLPMnist().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    model.train()
    for _ in range(epoch_num):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

    return model

# This function calculates the loss of the trained MLP model on MNIST
# as the target function
# As a user, you can define your own target function
def target_mnist_mlp(
    ckpt_path: str,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
):
    criterion = nn.CrossEntropyLoss(reduction="none")
    params = torch.load(ckpt_path)
    model = MLPMnist().to(device)
    model.load_state_dict(params)  # assuming model is defined somewhere
    model.eval()

    loss_list = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss_list.append(loss.clone().detach().cpu())

    return torch.cat(loss_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    
    ##############################
    # Data and model preparation
    ##############################
    # load the MNIST dataset
    dataset, dataset_test = create_mnist_dataset("./data")

    # dataloaders
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        sampler=SubsetSampler(range(1000)),
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=500,
        sampler=SubsetSampler(range(500)),
    )
    train_loader_attr = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
        sampler=SubsetSampler(range(1000)),
    )

    ##############################
    # Influence function score
    ##############################
    # train the full model
    print("========== Training full model for IF ==========")
    model = train_mnist_mlp(train_loader)
    model.to(args.device)
    model.eval()

    # define loss function
    # As a user, you will define it corresponding to
    # the model and target function you have defined
    def f(params, data_target_pair):
        image, label = data_target_pair
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model, params, image)
        return loss(yhat, label.long())

    # get influence function score
    task = AttributionTask(loss_func=f, model=model, checkpoints=model.state_dict())
    attributor = IFAttributorCG(
        task=task, device=args.device, regularization=0.01, max_iter=10
    )
    attributor.cache(train_loader_attr)
    with torch.no_grad():
        score = attributor.attribute(train_loader_attr, test_loader)

    ##############################
    # Ground truth
    ##############################
    # retrain the model for the Linear Datamodeling Score (LDS) metric calculation
    print("========== Retraining LDS ==========")
    retrain_lds(
        train_mnist_mlp,
        train_loader,
        args.path,
        num_subsets=30,
        num_runs_per_subset=2,
        seed=1,
        device=args.device
    )

    # calculate the ground-truth values for the Linear Datamodeling Score (LDS) metric
    print("========== Calculating ground truth ==========")
    ground_truth = calculate_lds_ground_truth(
        target_mnist_mlp,
        args.path,
        test_loader
    )

    ##############################
    # Calculate and print LDS score
    ##############################
    lds_score = lds(score, ground_truth)[0]
    print("lds:", torch.mean(lds_score[~torch.isnan(lds_score)]))