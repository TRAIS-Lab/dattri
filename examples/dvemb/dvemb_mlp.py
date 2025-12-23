"""Example to compute trajectory-specific leave-one-out correlation on MLP + MNIST.
"""

import os
import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from _groundtruth_loo_mlp import LOOGroundTruthGenerator, MLPMnist
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

from dattri.algorithm.dvemb import DVEmbAttributor
from dattri.benchmark.load import load_benchmark
from dattri.task import AttributionTask

num_epochs = 3  # number of training epochs
batch_size = 64  # batch size for training
learning_rate = 1e-3  # learning rate
training_size = 5000  # number of training samples
test_size = 500  # number of test samples


def load_mnist_data(device):
    """Load MNIST data for training and testing."""
    model_details, _ = load_benchmark(
        model="mlp",
        dataset="mnist",
        download_path="./benchmark_data",
        metric="loo",
    )
    train_dataset_full = model_details["train_dataset"]
    train_sampler = model_details["train_sampler"]
    test_dataset_full = model_details["test_dataset"]
    test_sampler = model_details["test_sampler"]

    train_indices = range(training_size)
    test_indices = range(test_size)

    train_subset = Subset(train_dataset_full, train_indices)
    train_inputs = torch.stack([train_subset[i][0] for i in range(len(train_subset))])
    train_labels = torch.tensor([train_subset[i][1] for i in range(len(train_subset))])

    test_subset = Subset(test_dataset_full, test_indices)
    test_inputs = torch.stack([test_subset[i][0] for i in range(len(test_subset))])
    test_labels = torch.tensor([test_subset[i][1] for i in range(len(test_subset))])

    return train_inputs, train_labels, torch.tensor(train_indices), test_inputs, test_labels


def calculate_dvemb_score():
    """Run the full DVEmb pipeline: training, attribution."""
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load MNIST data
    train_inputs, train_labels, train_indices, test_inputs, test_labels = load_mnist_data(device)

    # Fix random seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Initialize model, optimizer, and loss function
    model = MLPMnist(dropout_rate=0).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(train_inputs, train_labels, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    def loss_func_for_task(model, batch, device):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        return nn.functional.cross_entropy(outputs, targets)

    # Initialize AttributionTask
    task = AttributionTask(
        loss_func=loss_func_for_task,
        model=model,
        checkpoints=[model.state_dict()],
    )

    # DVEmb initialization
    attributor = DVEmbAttributor(
        task=task,
        proj_dim=4096,
        factorization_type="elementwise",
    )

    # Train the model and cache gradients using DVEmbAttributor
    model.train()
    print("Starting training and caching gradients...")
    for epoch in range(num_epochs):
        for inputs, labels, indices in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            # --- DVEmb Usage: Caching Gradients ---
            # In each training step (before the optimizer step), call `cache_gradients`.
            # This computes and stores the per-sample gradients for the current batch,
            # which are essential for calculating the data value embeddings.
            # It requires the current epoch, batch data, indices, and learning rate.
            attributor.cache_gradients(epoch=epoch,
                                       batch_data=(inputs, labels),
                                       indices=indices,
                                       learning_rate=learning_rate)

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
    # After training, first call `cache`.
    # This method uses all cached gradients to compute the data value embedding
    # vector for each training sample.
    print("\nComputing Data Value Embeddings...")
    attributor.cache()

    # Next, call `attribute` to calculate the influence.
    # It computes the gradients for the test set and multiplies them with the
    # training samples' embeddings to produce a matrix representing the influence
    # of each training point on each test point.
    print("Calculating influence...")
    dvemb_scores = attributor.attribute(test_loader, epoch=None)

    return dvemb_scores


def calculate_loo_groundtruth():
    """Calculate LOO ground truth scores using LOOGroundTruthGenerator."""
    # plot LOO correlation (spearman ranking correlation)
    loo_generator = LOOGroundTruthGenerator(num_epochs=num_epochs,
                                            batch_size=batch_size,
                                            learning_rate=learning_rate,
                                            train_samples=training_size,
                                            test_samples=test_size,
                                            all_seed=0)
    groundtruth_scores = loo_generator.generate_loo_groundtruth(
        download_path="./benchmark_data",
        output_dir="./mlp_gt",
    )
    print("LOO ground truth generation completed!")

    return groundtruth_scores


def evaluate(dvemb_scores, groundtruth_scores):
    """Evaluate DVEmb scores against groundtruth LOO scores."""
    selected_indices = []
    with pathlib.Path(os.path.join("./mlp_gt", "selected_train_indices.txt")).open("r") as f:
        for line in f:
            selected_indices.append(int(line.strip()))
    dvemb_scores = dvemb_scores[selected_indices, :]

    dvemb_np = dvemb_scores.detach().cpu().numpy()
    groundtruth_scores = (groundtruth_scores, torch.Tensor(list(range(dvemb_scores.shape[0]))))

    # calculate mean spearmanr for scores and groundtruth_scores
    from scipy.stats import spearmanr
    spearman_corrs = []
    for i in range(dvemb_np.shape[1]):
        spearman_corr = spearmanr(dvemb_np[:, i], groundtruth_scores[0][:, i].numpy()).correlation
        if not np.isnan(spearman_corr):
            spearman_corrs.append(spearman_corr)
    mean_spearman_corr = np.mean(spearman_corrs)
    print(f"Mean Spearmanr: {mean_spearman_corr:.4f}")

    # plot scatter plots for each of the 10 classes
    for i in range(10):
        plt.figure()
        plt.scatter(torch.tensor(dvemb_np)[:, i].numpy(), groundtruth_scores[0][:, i].numpy())
        # calculate the spearmanr
        from scipy.stats import spearmanr
        spearman_corr = spearmanr(torch.tensor(dvemb_np)[:, i].numpy(), groundtruth_scores[0][:, i].numpy()).correlation
        plt.xlabel("DVEmb Scores")
        plt.ylabel("Ground Truth LOO Scores")
        plt.title("DVEmb vs Ground Truth LOO Scores, Spearmanr: {:.4f}".format(spearman_corr))
        plt.savefig(f"dvemb_vs_groundtruth_loo_{i}.png")


if __name__ == "__main__":
    calculated_dvemb_scores = calculate_dvemb_score()
    groundtruth_scores = calculate_loo_groundtruth()
    evaluate(calculated_dvemb_scores, groundtruth_scores)
