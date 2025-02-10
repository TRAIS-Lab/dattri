import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import argparse

from dattri.algorithm import IFAttributorLiSSA
from dattri.task import AttributionTask
from dattri.benchmark.utils import SubsetSampler


def create_synthetic_2d_data(num_per_class=30, outlier_coord=(1.5, 1.5), seed=0):
    """
    Generate two Gaussian clusters for binary classification.
    One point is replaced with an outlier_coord and intentionally mislabeled.
    Returns x_train, y_train, outlier_idx.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    class0 = torch.randn(num_per_class, 2) * 0.5 + torch.tensor([-0.5, -1.0])
    class1 = torch.randn(num_per_class, 2) * 0.5 + torch.tensor([0.5, 1.0])

    x_train = torch.cat([class0, class1], dim=0)
    y_train = torch.cat([
        torch.zeros(num_per_class),
        torch.ones(num_per_class)
    ], dim=0).long()

    outlier_idx = num_per_class
    x_train[outlier_idx] = torch.tensor(outlier_coord)
    y_train[outlier_idx] = 0

    return x_train, y_train, outlier_idx


class SimpleLinearModel(nn.Module):
    def __init__(self, in_features=2, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)


def train_linear_model(x_train, y_train, lr=0.1, epochs=2000, device="cuda"):
    model = SimpleLinearModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    x_train_device = x_train.to(device)
    y_train_device = y_train.to(device)

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_train_device)
        loss = loss_fn(logits, y_train_device)
        loss.backward()
        optimizer.step()

    return model


def loss_func(params, data_target_pair):
    x, y = data_target_pair
    y = y.to(torch.int64)
    logits = torch.func.functional_call(model, params, x)
    return nn.CrossEntropyLoss()(logits, y)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--no_output", action="store_true")
    args = parser.parse_args()

    # Generate training data with outlier
    outlier_coord = (0.0, 2.8)
    x_train, y_train, outlier_idx = create_synthetic_2d_data(
        num_per_class=50,
        outlier_coord=outlier_coord,
        seed=0
    )

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        sampler=SubsetSampler(range(len(train_dataset)))
    )

    # Train model
    model = train_linear_model(x_train, y_train, lr=0.1, epochs=2000, device=device)

    task = AttributionTask(
        loss_func=loss_func,
        model=model,
        checkpoints=model.state_dict()
    )

    attributor = IFAttributorLiSSA(
        task=task,
        device=device,
        num_repeat=1,
        recursion_depth=200,
        scaling=100
    )
    attributor.cache(train_loader)

    # Generate grid test set
    x_min, x_max = -3.0, 3.0
    y_min, y_max = -3.0, 3.0
    grid_res = 80
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res)
    )
    grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float()
    grid_labels = torch.zeros(len(grid_points), dtype=torch.long)
    test_dataset = TensorDataset(grid_points, grid_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        sampler=SubsetSampler(range(len(test_dataset)))
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    methods = [None, "l", "theta"]
    titles = [
        "IF: None",
        "IF: relative='l'",
        "IF: relative='theta'"
    ]

    for ax, method, title in zip(axes, methods, titles):
        with torch.no_grad():
            influence_scores = attributor.attribute(
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                relatif_method=method
            )
        influence_scores = influence_scores.cpu()

        # Indices of training samples with max absolute influence on each test point
        dominant_indices = torch.abs(influence_scores).argmax(dim=0)

        # Model predictions on grid
        model.eval()
        with torch.no_grad():
            logits_test = model(grid_points.to(device))
            preds_test = torch.argmax(logits_test, dim=1).cpu()

        # Decision boundary
        x_boundary = np.linspace(x_min, x_max, 200)
        w_diff = (model.linear.weight[0] - model.linear.weight[1]).detach().cpu().numpy()
        b_diff = (model.linear.bias[0] - model.linear.bias[1]).detach().cpu().item()
        if abs(w_diff[1]) > 1e-7:
            y_boundary = - (w_diff[0] * x_boundary + b_diff) / w_diff[1]
            ax.plot(x_boundary, y_boundary, "k--", label="Boundary")
        else:
            x_const = - b_diff / w_diff[0]
            ax.axvline(x_const, color="k", linestyle="--", label="Boundary")

        # Plot training points
        x_np = x_train.numpy()
        y_np = y_train.numpy()
        ax.scatter(x_np[y_np == 0, 0], x_np[y_np == 0, 1], c="red", edgecolors="k", label="C0")
        ax.scatter(x_np[y_np == 1, 0], x_np[y_np == 1, 1], c="green", edgecolors="k", label="C1")

        # Highlight outlier
        outlier_pt = x_train[outlier_idx].numpy()
        ax.scatter(
            outlier_pt[0],
            outlier_pt[1],
            s=200,
            facecolors="none",
            edgecolors="blue",
            linewidths=2,
            label="Outlier"
        )

        # Mark region dominated by outlier with a contour
        dominated_mask = (dominant_indices == outlier_idx).numpy().reshape(xx.shape)
        ax.contourf(xx, yy, dominated_mask.astype(float),
                    levels=[-0.5, 0.5, 1.5], colors=["white", "yellow"], alpha=0.3)

        ax.set_title(title)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend(loc="best")

    plt.tight_layout()

    if not args.no_output:
        plt.show()
