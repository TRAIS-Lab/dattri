"""
Example code to compute DVEmb attributions on MLP trained on MNIST dataset.
Code using DVEmb is in TestDVEmbMLP class at the bottom.
Also includes code to generate LOO ground truth scores for comparison since DVEmb is
very sensitive to the order of training samples, so we provide this for validation.
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
from dattri.func.utils import flatten_func

class LOOGroundTruthGenerator:
    def __init__(self):
        self.num_epochs = 3
        self.batch_size = 64
        self.learning_rate = 1e-4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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

        train_sampler.indices = range(0, 5000)
        test_sampler.indices = range(0, 500)
        print(train_sampler.indices)
        print(test_sampler.indices)

        train_subset = Subset(train_dataset_full, train_sampler.indices)
        train_inputs = torch.stack([train_subset[i][0] for i in range(len(train_subset))])
        train_labels = torch.tensor([train_subset[i][1] for i in range(len(train_subset))])

        test_subset = Subset(test_dataset_full, test_sampler.indices)
        test_inputs = torch.stack([test_subset[i][0] for i in range(len(test_subset))])
        test_labels = torch.tensor([test_subset[i][1] for i in range(len(test_subset))])

        return train_inputs, train_labels, test_inputs, test_labels

    def _train_model(self, train_inputs, train_labels, excluded_idx=None):
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        model = MLPMnist(dropout_rate=0).to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        indices = torch.arange(len(train_inputs))
        train_dataset = TensorDataset(train_inputs, train_labels, indices)
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        generator = torch.Generator()
        generator.manual_seed(0)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, generator=generator)

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

    def generate_loo_groundtruth(self, download_path="./benchmark_data", output_path="./loo_groundtruth.pt"):
        train_inputs, train_labels, test_inputs, test_labels = self._load_data(download_path)

        num_train_samples = len(train_inputs)
        num_test_samples = len(test_inputs)

        full_model = self._train_model(train_inputs, train_labels)
        full_losses = self._compute_loss_on_test_set(full_model, test_inputs, test_labels)

        loo_losses = torch.zeros(100, num_test_samples)

        for i in tqdm(range(100), desc="LOO iterations"):
            loo_model = self._train_model(train_inputs, train_labels, excluded_idx=i)
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
        }, output_path)

        print(f"LOO ground truth shape: {loo_losses.shape}")
        print(f"Mean absolute LOO effect: {torch.abs(loo_losses).mean():.6f}")
        print(f"Max absolute LOO effect: {torch.abs(loo_losses).max():.6f}")

        return loo_losses

class TestDVEmbMLP(unittest.TestCase):
    def setUp(self):
        self.num_epochs = 3
        self.batch_size = 64
        self.learning_rate = 1e-4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.correlation_threshold = 0.6

    def _load_data(self, download_path: str = "./benchmark_data"):
        """Load MNIST benchmark data and ground truth LOO scores."""
        model_details, old_gt = load_benchmark(
            model="mlp",
            dataset="mnist",
            metric="loo",
            download_path=download_path
        )
        print(old_gt[0].shape, old_gt[1].shape)

        train_dataset_full = model_details["train_dataset"]
        train_sampler = model_details["train_sampler"]
        test_dataset_full = model_details["test_dataset"]
        test_sampler = model_details["test_sampler"]

        train_sampler.indices = range(0, 5000)
        test_sampler.indices = range(0, 500)

        train_subset = Subset(train_dataset_full, train_sampler.indices)
        train_inputs = torch.stack([train_subset[i][0] for i in range(len(train_subset))])
        train_labels = torch.tensor([train_subset[i][1] for i in range(len(train_subset))])

        test_subset = Subset(test_dataset_full, test_sampler.indices)
        test_inputs = torch.stack([test_subset[i][0] for i in range(len(test_subset))])
        test_labels = torch.tensor([test_subset[i][1] for i in range(len(test_subset))])

        groundtruth = torch.load("./benchmark_data/loo_groundtruth_mlp_mnist.pt")['loo_losses']

        # groundtruth_loo_scores, _ = groundtruth

        return train_inputs, train_labels, train_sampler.indices, test_inputs, test_labels, groundtruth

    def _setup_model_and_loss(self):
        """Setup logistic regression model and loss function."""
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        model = MLPMnist(dropout_rate=0).to(self.device)
        print(list(model.parameters()))
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        @flatten_func(model)
        def per_sample_loss_fn(params, data_tensors):
            image, label = data_tensors
            image = image.unsqueeze(0)
            label = label.unsqueeze(0)
            yhat = torch.func.functional_call(model, params, image)
            return criterion(yhat, label)

        return model, optimizer, criterion, per_sample_loss_fn

    def _train_with_dvemb_caching(
        self, model, optimizer, criterion, per_sample_loss_fn,
        train_inputs, train_labels, train_indices
    ):
        """Train model while caching gradients for DVEmb."""
        train_indices_tensor = torch.tensor(train_indices)
        train_dataset_with_indices = TensorDataset(train_inputs, train_labels, train_indices_tensor)
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        generator = torch.Generator()
        generator.manual_seed(0)
        train_loader = DataLoader(train_dataset_with_indices, batch_size=self.batch_size, shuffle=False, generator=generator)

        """Initialize DVEmb attributor with projection."""
        attributor = DVEmbAttributor(
            model=model,
            loss_func=per_sample_loss_fn,
            device=self.device,
            use_projection=True,
            projection_dim=4096
        )

        """Train the model while caching gradients for DVEmb."""
        model.train()
        for epoch in range(self.num_epochs):
            for inputs, labels, indices in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                attributor.cache_gradients(epoch, (inputs, labels), indices, self.learning_rate)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return attributor

    def _compute_dvemb_scores(
        self, attributor, train_inputs, test_inputs, test_labels
    ):
        """Compute and return the final DVEmb attribution scores."""
        test_dataset = TensorDataset(test_inputs, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        """Compute DVEmb scores."""
        attributor.compute_embeddings()
        """Get DVEmb scores for the test set."""
        dvemb_scores = attributor.attribute(test_loader, epoch=None)

        return dvemb_scores

    def _compare_with_groundtruth(self, dvemb_scores, groundtruth_scores):
        """Compare DVEmb scores with ground truth using Pearson correlation."""
        dvemb_np = dvemb_scores.detach().cpu().numpy()

        from dattri.metric import loo_corr

        groundtruth_scores = (groundtruth_scores, torch.Tensor(list(range(0, 500))))

        print(dvemb_np.shape, groundtruth_scores[0].shape)

        dvemb_np = dvemb_np[:100]
        groundtruth_scores = (groundtruth_scores[0][:100], groundtruth_scores[1])

        corr = loo_corr(torch.tensor(dvemb_np), groundtruth_scores)[0]
        corr = torch.mean(corr[~torch.isnan(corr)]).item()

        plt.figure()
        plt.scatter(torch.tensor(dvemb_np).sum(dim=1).numpy(), groundtruth_scores[0].sum(dim=1).numpy())
        from scipy.stats import spearmanr
        spearman_corr = spearmanr(torch.tensor(dvemb_np).sum(dim=1).numpy(), groundtruth_scores[0].sum(dim=1).numpy()).correlation
        plt.xlabel("DVEmb Scores")
        plt.ylabel("Ground Truth LOO Scores")
        plt.title("DVEmb vs Ground Truth LOO Scores, Spearmanr: {:.4f}".format(spearman_corr))
        plt.savefig(f"./dvemb_vs_groundtruth_loo_all.png")

        print(f"Mean LOO correlation: {corr:.4f}")

        self.assertGreater(
            corr,
            self.correlation_threshold,
            f"LOO correlation {corr:.4f} is below threshold {self.correlation_threshold}"
        )

    def test_correlation_with_ground_truth(self):
        train_inputs, train_labels, train_indices, test_inputs, test_labels, groundtruth_scores = self._load_data()
        model, optimizer, criterion, per_sample_loss_fn = self._setup_model_and_loss()
        attributor = self._train_with_dvemb_caching(
            model, optimizer, criterion, per_sample_loss_fn,
            train_inputs, train_labels, train_indices
        )
        dvemb_scores = self._compute_dvemb_scores(
            attributor, train_inputs, test_inputs, test_labels
        )
        self._compare_with_groundtruth(dvemb_scores, groundtruth_scores)

if __name__ == "__main__":
    generator = LOOGroundTruthGenerator()
    generator.generate_loo_groundtruth(
        download_path="./benchmark_data",
        output_path="./benchmark_data/loo_groundtruth_mlp_mnist.pt"
    )
    unittest.main()
