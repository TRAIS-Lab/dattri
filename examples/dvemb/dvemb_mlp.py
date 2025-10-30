"""
Example code to compute Leave-One-Out (LOO) scores on MLP trained on MNIST dataset.
"""

import torch
import random
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm
import unittest
import matplotlib.pyplot as plt

from dattri.benchmark.load import load_benchmark
from dattri.benchmark.models.mlp import MLPMnist
from dattri.algorithm.dvemb import DVEmbAttributor
from dattri.task import AttributionTask
from torch.func import functional_call

class DVEmbExample(unittest.TestCase):
    def setUp(self):
        self.num_epochs = 10
        self.batch_size = 64
        self.learning_rate = 1e-4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Fix random seeds for reproducibility
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

    def load_mnist_data(self):
        model_details, _ = load_benchmark(
            model="mlp",
            dataset="mnist",
            download_path="./benchmark_data",
            metric="loo"
        )
        train_dataset_full = model_details["train_dataset"]
        train_sampler = model_details["train_sampler"]
        test_dataset_full = model_details["test_dataset"]
        test_sampler = model_details["test_sampler"]

        train_indices = range(0, 5000)
        test_indices = range(0, 500)

        train_subset = Subset(train_dataset_full, train_indices)
        train_inputs = torch.stack([train_subset[i][0] for i in range(len(train_subset))])
        train_labels = torch.tensor([train_subset[i][1] for i in range(len(train_subset))])

        test_subset = Subset(test_dataset_full, test_indices)
        test_inputs = torch.stack([test_subset[i][0] for i in range(len(test_subset))])
        test_labels = torch.tensor([test_subset[i][1] for i in range(len(test_subset))])

        return train_inputs, train_labels, torch.tensor(train_indices), test_inputs, test_labels

    def test_dvemb_attribution(self):
        """Run the full DVEmb pipeline: training, attribution, and visualization."""
        # Load MNIST data
        train_inputs, train_labels, train_indices, test_inputs, test_labels = self.load_mnist_data()

        # Initialize model, optimizer, and loss function
        model = MLPMnist(dropout_rate=0).to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        train_dataset = TensorDataset(train_inputs, train_labels, train_indices)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

        # Define loss function for AttributionTask
        def loss_func_for_task(params, data):
            inputs, labels = data
            inputs = inputs.unsqueeze(0)
            labels = labels.unsqueeze(0)
            outputs = torch.func.functional_call(model, params, (inputs,))
            return criterion(outputs, labels)

        # Initialize AttributionTask
        task = AttributionTask(
            loss_func=loss_func_for_task,
            model=model,
            checkpoints=[model.state_dict()],
        )

        # DVEmb initialization
        attributor = DVEmbAttributor(
            task=task,
            criterion=criterion,
            proj_dim=4096,
            factorization_type="kronecker",
            # "kronecker" is the same as used in the original DVEmb paper
            # To get better performance, consider using "elementwise"
        )

        # Train the model and cache gradients using DVEmbAttributor
        model.train()
        print("Starting training and caching gradients...")
        for epoch in range(self.num_epochs):
            for inputs, labels, indices in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # --- DVEmb Usage: Caching Gradients ---
                # In each training step (before the optimizer step), call `cache_gradients`.
                # This computes and stores the per-sample gradients for the current batch,
                # which are essential for calculating the data value embeddings.
                # It requires the current epoch, batch data, indices, and learning rate.
                attributor.cache_gradients(epoch, (inputs, labels), indices, self.learning_rate)

                # Perform the standard training step
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Compute DVEmb influence
        test_dataset = TensorDataset(test_inputs, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        # --- DVEmb Usage: Computing Embeddings and Influence ---
        # After training, first call `compute_embeddings`.
        # This method uses all cached gradients to compute the data value embedding
        # vector for each training sample.
        print("\nComputing Data Value Embeddings...")
        attributor.compute_embeddings()

        # Next, call `attribute` to calculate the influence.
        # It computes the gradients for the test set and multiplies them with the
        # training samples' embeddings to produce a matrix representing the influence
        # of each training point on each test point.
        print("Calculating influence...")
        dvemb_influence = attributor.attribute(test_loader, epoch=None)

        # Visualize the results (LOO scores trend)
        all_epochs_influence = []
        num_samples_per_epoch = len(train_inputs)
        # Calculate influence per epoch for trend visualization
        for epoch in range(self.num_epochs):
            dvemb_influence_epoch = attributor.attribute(test_loader, epoch=epoch)
            total_influence_epoch = dvemb_influence_epoch.sum(dim=1).cpu().detach().numpy()
            all_epochs_influence.append(total_influence_epoch)
        continuous_influence = np.concatenate(all_epochs_influence)
        plt.figure(figsize=(15, 7))
        plt.plot(continuous_influence, alpha=0.8)
        for epoch in range(1, self.num_epochs):
            plt.axvline(x=epoch * num_samples_per_epoch -1, color='r', linestyle='--', linewidth=1)
        plt.title("Continuous Influence of Training Samples Across All Epochs")
        plt.xlabel("Training Sample Index (Processed Sequentially Across Epochs)")
        plt.ylabel("Summed Influence on Test Set")
        plt.grid(True)
        plt.savefig("./dvemb_continuous_influence_trend.png")

if __name__ == "__main__":
    unittest.main()
