"""Test for influence function."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dattri.algorithm.influence_function import (
    IFAttributorArnoldi,
    IFAttributorCG,
    IFAttributorDataInf,
    IFAttributorEKFAC,
    IFAttributorExplicit,
    IFAttributorLiSSA,
)
from dattri.benchmark.datasets.mnist import train_mnist_lr
from dattri.task import AttributionTask


class TestInfluenceFunction:
    """Test for influence function."""

    def test_influence_function(self):
        """Test for influence function."""
        train_dataset = TensorDataset(
            torch.randn(20, 1, 28, 28),
            torch.randint(0, 10, (20,)),
        )
        test_dataset = TensorDataset(
            torch.randn(10, 1, 28, 28),
            torch.randint(0, 10, (10,)),
        )
        train_loader = DataLoader(train_dataset, batch_size=4)
        test_loader = DataLoader(test_dataset, batch_size=2)

        model = train_mnist_lr(train_loader)

        def f(params, data_target_pair):
            image, label = data_target_pair
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image)
            return loss(yhat, label.long())

        task = AttributionTask(
            loss_func=f,
            model=model,
            checkpoints=model.state_dict(),
        )
        # Explicit
        attributor = IFAttributorExplicit(
            task=task,
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor.cache(train_loader)
        attributor.attribute(train_loader, test_loader)

        # CG
        attributor = IFAttributorCG(
            task=task,
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor.cache(train_loader)
        attributor.attribute(train_loader, test_loader)

        # arnoldi
        attributor = IFAttributorArnoldi(
            task=task,
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor.cache(train_loader)
        attributor.attribute(train_loader, test_loader)

        # lissa
        attributor = IFAttributorLiSSA(
            task=task,
            device=torch.device("cpu"),
            recursion_depth=5,
            batch_size=2,
        )
        attributor.cache(train_loader)
        attributor.attribute(train_loader, test_loader)
        attributor.attribute(train_loader, test_loader, "l")
        attributor.attribute(train_loader, test_loader, "theta")

        # DataInf
        attributor = IFAttributorDataInf(
            task=task,
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor.cache(train_loader)
        attributor.attribute(train_loader, test_loader)

        # EK-FAC
        attributor = IFAttributorEKFAC(
            task=task,
            device=torch.device("cpu"),
            damping=0.1,
        )
        attributor.cache(train_loader)
        attributor.attribute(train_loader, test_loader)

    # pylint: disable=too-many-statements,PLR0915
    # flake8: noqa: PLR0915
    def test_influence_function_self_attribute(self):
        """Test for self_attribute function in influence function."""
        # Test the self attribute
        train_dataset = TensorDataset(
            torch.randn(20, 1, 28, 28),
            torch.randint(0, 10, (20,)),
        )
        train_loader = DataLoader(train_dataset, batch_size=20)

        model = train_mnist_lr(train_loader)

        def f(params, data_target_pair):
            image, label = data_target_pair
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image)
            return loss(yhat, label.long())

        task = AttributionTask(
            loss_func=f,
            model=model,
            checkpoints=model.state_dict(),
        )

        # Explicit
        attributor = IFAttributorExplicit(
            task=task,
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor.cache(train_loader)
        tensor1 = attributor.attribute(train_loader, train_loader).diag()
        tensor2 = attributor.self_attribute(train_loader)
        assert torch.allclose(tensor1, tensor2)

        # CG
        attributor = IFAttributorCG(
            task=task,
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor.cache(train_loader)
        tensor1 = attributor.attribute(train_loader, train_loader).diag()
        tensor2 = attributor.self_attribute(train_loader)
        assert torch.allclose(tensor1, tensor2)

        # Arnoldi
        attributor = IFAttributorArnoldi(
            task=task,
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor.cache(train_loader)
        tensor1 = attributor.attribute(train_loader, train_loader).diag()
        tensor2 = attributor.self_attribute(train_loader)

        # LiSSA
        attributor = IFAttributorLiSSA(
            task=task,
            device=torch.device("cpu"),
            recursion_depth=5,
            batch_size=20,
        )
        attributor.cache(train_loader)
        tensor1 = attributor.attribute(train_loader, train_loader).diag()
        tensor2 = attributor.self_attribute(train_loader)
        assert torch.allclose(tensor1, tensor2)

        # DataInf
        attributor = IFAttributorDataInf(
            task=task,
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor.cache(train_loader)
        tensor1 = attributor.attribute(train_loader, train_loader).diag()
        tensor2 = attributor.self_attribute(train_loader)
        assert torch.allclose(tensor1, tensor2)

        # EK-FAC
        attributor = IFAttributorEKFAC(
            task=task,
            device=torch.device("cpu"),
            damping=0.1,
        )
        attributor.cache(train_loader)
        tensor1 = attributor.attribute(train_loader, train_loader).diag()
        tensor2 = attributor.self_attribute(train_loader)
        assert torch.allclose(tensor1, tensor2)

    def test_influence_function_partial_param(self):
        """Test for influence function."""
        train_dataset = TensorDataset(
            torch.randn(20, 1, 28, 28),
            torch.randint(0, 10, (20,)),
        )
        test_dataset = TensorDataset(
            torch.randn(10, 1, 28, 28),
            torch.randint(0, 10, (10,)),
        )
        train_loader = DataLoader(train_dataset, batch_size=4)
        test_loader = DataLoader(test_dataset, batch_size=2)

        model = train_mnist_lr(train_loader)

        def f(params, data_target_pair):
            image, label = data_target_pair
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image)
            return loss(yhat, label.long())

        task = AttributionTask(
            loss_func=f,
            model=model,
            checkpoints=model.state_dict(),
        )

        # Explicit w/ layer_name
        attributor = IFAttributorExplicit(
            task=task,
            layer_name=["linear.weight"],
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor.cache(train_loader)
        attributor.attribute(train_loader, test_loader)

        # CG
        attributor = IFAttributorCG(
            task=task,
            layer_name=["linear.weight"],
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor.cache(train_loader)
        attributor.attribute(train_loader, test_loader)

        # arnoldi
        attributor = IFAttributorArnoldi(
            task=task,
            layer_name=["linear.weight"],
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor.cache(train_loader)
        attributor.attribute(train_loader, test_loader)

        # lissa
        attributor = IFAttributorLiSSA(
            task=task,
            layer_name=["linear.weight"],
            device=torch.device("cpu"),
            recursion_depth=5,
            batch_size=2,
        )
        attributor.cache(train_loader)
        attributor.attribute(train_loader, test_loader)

    def test_misusage_influence_function(self):
        """Test for some misusage of influence function."""
        train_dataset = TensorDataset(
            torch.randn(20, 1, 28, 28),
            torch.randint(0, 10, (20,)),
        )
        test_dataset = TensorDataset(
            torch.randn(10, 1, 28, 28),
            torch.randint(0, 10, (10,)),
        )
        train_loader = DataLoader(train_dataset, batch_size=4)
        test_loader = DataLoader(test_dataset, batch_size=2)

        model = train_mnist_lr(train_loader)

        def f(params, data_target_pair):
            image, label = data_target_pair
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image)
            return loss(yhat, label.long())

        task = AttributionTask(
            loss_func=f,
            model=model,
            checkpoints=model.state_dict(),
        )

        # Arnoldi w/o cache calling
        attributor = IFAttributorArnoldi(
            task=task,
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        error_msg = "The Arnoldi projector has not been cached"
        with pytest.raises(ValueError, match=error_msg):
            attributor.attribute(train_loader, test_loader)

    def test_influence_function_ensemble(self):
        """Test for influence function with ensembling."""
        train_dataset = TensorDataset(
            torch.randn(20, 1, 28, 28),
            torch.randint(0, 10, (20,)),
        )
        test_dataset = TensorDataset(
            torch.randn(10, 1, 28, 28),
            torch.randint(0, 10, (10,)),
        )
        train_loader = DataLoader(train_dataset, batch_size=4)
        test_loader = DataLoader(test_dataset, batch_size=2)

        model = train_mnist_lr(train_loader)

        def f(params, data_target_pair):
            image, label = data_target_pair
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image)
            return loss(yhat, label.long())

        task = AttributionTask(
            loss_func=f,
            model=model,
            checkpoints=model.state_dict(),
        )
        task_m = AttributionTask(
            loss_func=f,
            model=model,
            checkpoints=[model.state_dict(), model.state_dict()],
        )
        # 1 checkpoint
        attributor = IFAttributorExplicit(
            task=task,
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor.cache(train_loader)
        score = attributor.attribute(train_loader, test_loader)

        # multi checkpoints
        attributor_m = IFAttributorExplicit(
            task=task_m,
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor_m.cache(train_loader)
        score_m = attributor_m.attribute(train_loader, test_loader)

        assert torch.allclose(score, score_m)

    def test_datainf_transform_test_rep(self):
        """Test for datainf test representation transformation."""
        train_dataset = TensorDataset(
            torch.randn(20, 1, 28, 28),
            torch.randint(0, 10, (20,)),
        )
        train_loader = DataLoader(train_dataset, batch_size=4)

        model = train_mnist_lr(train_loader)

        def f(params, data_target_pair):
            image, label = data_target_pair
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image)
            return loss(yhat, label.long())

        task = AttributionTask(
            loss_func=f,
            model=model,
            checkpoints=model.state_dict(),
        )

        # DataInf
        attributor = IFAttributorDataInf(
            task=task,
            device=torch.device("cpu"),
            regularization=1e-3,
            fim_estimate_data_ratio=0.5,
        )
        attributor.cache(train_loader)
        test_rep = torch.randn((30, 7850), device=torch.device("cpu"))
        transformed_test_rep = attributor.transform_test_rep(0, test_rep)
        # Calculating ground truth transformed query
        cached_train_reps = attributor._cached_train_reps[0]  # noqa: SLF001
        cached_train_reps_layers = attributor._get_layer_wise_reps(0, cached_train_reps)  # noqa: SLF001
        test_rep_layers = attributor._get_layer_wise_reps(0, test_rep)  # noqa: SLF001

        test_rep_layer = test_rep_layers[0]
        grad_layer = cached_train_reps_layers[0]
        running_transformation = 0
        # One layer model
        for grad_ in grad_layer:
            grad = grad_.unsqueeze(-1)
            running_transformation += (
                (test_rep_layer @ grad) @ grad.T / (1e-3 + torch.norm(grad) ** 2)
            )
        running_transformation /= len(grad_layer)
        running_transformation = test_rep_layer - running_transformation
        running_transformation /= 1e-3
        ground_truth_test_rep = running_transformation

        import math

        # Check for estimate data ratio
        assert cached_train_reps.shape[0] == 4 * math.ceil(0.5 * len(train_loader))
        # Check for transformed test rep
        assert torch.allclose(ground_truth_test_rep, transformed_test_rep, atol=1e-4)

    def test_ekfac_transform_test_rep(self):
        """Test for EK-FAC test representation transformation."""

        def average_pairwise_correlation(tensor1, tensor2):
            stacked = torch.stack([tensor1, tensor2], dim=0)
            reshaped = stacked.view(2, -1)

            corr_matrix = torch.corrcoef(reshaped)
            pairwise_corr = corr_matrix[0, 1]

            return pairwise_corr.item()

        train_dataset = TensorDataset(
            torch.randn(20, 1, 28, 28),
            torch.randint(0, 10, (20,)),
        )
        train_loader = DataLoader(train_dataset, batch_size=4)

        model = train_mnist_lr(train_loader)

        def f(params, data_target_pair):
            image, label = data_target_pair
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image)
            return loss(yhat, label.long())

        task = AttributionTask(
            loss_func=f,
            model=model,
            checkpoints=model.state_dict(),
        )

        # EK-FAC
        attributor = IFAttributorEKFAC(
            task=task,
            device=torch.device("cpu"),
            damping=0.1,
        )
        attributor.cache(train_loader)

        attributor_gt = IFAttributorExplicit(
            task=task,
            layer_name=attributor.layer_name,
            device=torch.device("cpu"),
            regularization=1e-3,
        )
        attributor_gt.cache(train_loader)

        test_rep = torch.randn((30, 7850), device=torch.device("cpu"))
        transformed_test_rep = attributor.transform_test_rep(0, test_rep)
        gt_test_rep = attributor_gt.transform_test_rep(0, test_rep)

        # Check pair-wise correlation
        threshold = 0.98
        corr = average_pairwise_correlation(gt_test_rep, transformed_test_rep)
        assert corr > threshold
