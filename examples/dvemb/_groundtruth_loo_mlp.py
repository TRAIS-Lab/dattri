import torch
import random
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import os
from tqdm import tqdm

from dattri.benchmark.load import load_benchmark

"""This file define functions to calculate the trajectory-specific leave-one-out values for MLP on MNIST dataset."""
from torch import nn


class MLPMnist(nn.Module):
    """A simple MLP model for MNIST dataset."""

    def __init__(self, dropout_rate: float = 0.1) -> None:
        """Initialize the MLP model.

        Args:
            dropout_rate: The dropout rate to use.
        """
        super(MLPMnist, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP model.

        Args:
            x: The input image tensor.

        Returns:
            (torch.Tensor): The output tensor.
        """
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class LOOGroundTruthGenerator:
    def __init__(self, num_epochs=3, batch_size=64, learning_rate=1e-3, train_samples=5000, test_samples=500, all_seed=0):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = all_seed

    def _load_data(self, download_path: str = "./benchmark_data"):
        model_details, _ = load_benchmark(
            model="mlp",
            dataset="mnist",
            metric="loo",
            download_path=download_path
        )

        train_dataset_full = model_details["train_dataset"]
        train_sampler = model_details["train_sampler"]
        test_dataset_full = model_details["test_dataset"]
        test_sampler = model_details["test_sampler"]

        train_sampler.indices = range(0, self.train_samples)
        test_sampler.indices = range(0, self.test_samples)

        train_subset = Subset(train_dataset_full, train_sampler.indices)
        train_inputs = torch.stack([train_subset[i][0] for i in range(len(train_subset))]).double()
        train_labels = torch.tensor([train_subset[i][1] for i in range(len(train_subset))])

        test_subset = Subset(test_dataset_full, test_sampler.indices)
        test_inputs = torch.stack([test_subset[i][0] for i in range(len(test_subset))]).double()
        test_labels = torch.tensor([test_subset[i][1] for i in range(len(test_subset))])

        return train_inputs, train_labels, test_inputs, test_labels

    def _train_model(self, train_inputs, train_labels, excluded_idx=None):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        model = MLPMnist(dropout_rate=0).to(self.device).double()

        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        indices = torch.arange(len(train_inputs))
        train_dataset = TensorDataset(train_inputs, train_labels, indices)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, generator=generator)

        model.train()
        for epoch in range(self.num_epochs):
            for inputs, labels, batch_indices in train_loader:
                if excluded_idx is not None:
                    mask = batch_indices != excluded_idx
                    if not mask.any():
                        continue
                    inputs = inputs[mask]
                    labels = labels[mask]
                    if len(inputs) == 0:
                        continue
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return model

    def _compute_loss_on_test_set(self, model, test_inputs, test_labels):
        """Compute loss on test set."""
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction='none')  # Get per-sample losses

        test_inputs = test_inputs.to(self.device)
        test_labels = test_labels.to(self.device)

        with torch.no_grad():
            outputs = model(test_inputs)
            losses = criterion(outputs, test_labels)

        return losses.cpu()

    def generate_loo_groundtruth(self, download_path="./benchmark_data", output_dir=".", num_train_samples=100):
        train_inputs, train_labels, test_inputs, test_labels = self._load_data(download_path)

        # num_train_samples = len(train_inputs)
        num_test_samples = len(test_inputs)

        # add seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # keep the sampling to be balanced among labels
        label_to_indices = {}
        for idx, label in enumerate(train_labels):
            label = label.item()
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        selected_indices = []
        num_classes = len(label_to_indices)
        samples_per_class = num_train_samples // num_classes
        for label, indices in label_to_indices.items():
            selected_indices.extend(random.sample(indices, samples_per_class))
        while len(selected_indices) < num_train_samples:
            label = random.choice(list(label_to_indices.keys()))
            indices = label_to_indices[label]
            selected_indices.append(random.choice(indices))

        # create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # save it to a text file
        with open(os.path.join(output_dir, "selected_train_indices.txt"), "w") as f:
            for idx in selected_indices:
                f.write(f"{idx}\n")

        full_model = self._train_model(train_inputs, train_labels)
        full_losses = self._compute_loss_on_test_set(full_model, test_inputs, test_labels)

        loo_losses = torch.zeros(num_train_samples, num_test_samples)

        for i in tqdm(range(num_train_samples), desc="LOO iterations"):
            loo_model = self._train_model(train_inputs, train_labels, excluded_idx=selected_indices[i])
            loo_test_losses = self._compute_loss_on_test_set(loo_model, test_inputs, test_labels)
            loo_losses[i] = loo_test_losses - full_losses

        torch.save({
            'loo_losses': loo_losses,
            'full_losses': full_losses,
            'train_samples': num_train_samples,
            'test_samples': num_test_samples,
            'hyperparameters': {
                'num_epochs': self.num_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate
            }
        }, os.path.join(output_dir, "loo_groundtruth.pt"))

        return loo_losses
