"""Test for TracIn."""

import copy
import shutil
from pathlib import Path

import torch
from torch import nn
from torch.func import grad, vmap
from torch.utils.data import DataLoader, TensorDataset

from dattri.algorithm.tracin import TracInAttributor
from dattri.benchmark.datasets.cifar import train_cifar_resnet9
from dattri.benchmark.datasets.mnist import train_mnist_lr, train_mnist_mlp
from dattri.func.utils import flatten_func, flatten_params
from dattri.task import AttributionTask


class TestTracInAttributor:
    """Test for TracIn."""

    def test_tracin_proj(self):
        """Test for TracIn with projectors."""
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

        # the function directly operate on batches of images/labels
        # not on dataloader anymore to allow vmap usage
        # need to prepare a batch dim in the begining
        def f(params, image_label_pair):
            image, label = image_label_pair
            image_t = image.unsqueeze(0)
            label_t = label.unsqueeze(0)
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image_t)
            return loss(yhat, label_t)

        # to simlulate multiple checkpoints
        model_1 = train_mnist_lr(train_loader, epoch_num=1)
        model_2 = train_mnist_lr(train_loader, epoch_num=2)
        path = Path("./ckpts")
        if not path.exists():
            path.mkdir(parents=True)
        torch.save(model_1.state_dict(), path / "model_1.pt")
        torch.save(model_2.state_dict(), path / "model_2.pt")

        checkpoint_list = ["ckpts/model_1.pt", "ckpts/model_2.pt"]

        task = AttributionTask(
            loss_func=f,
            model=model,
            checkpoints=checkpoint_list,
        )

        # train and test always share the same projector
        # checkpoints need to have differnt projectors
        pytest_device = "cpu"
        projector_kwargs = {
            "proj_dim": 512,
            "proj_max_batch_size": 32,
            "proj_seed": 42,
            "device": pytest_device,
            "use_half_precision": True,
        }

        # test with projector list
        attributor = TracInAttributor(
            task=task,
            weight_list=torch.ones(len(checkpoint_list)),
            normalized_grad=True,
            projector_kwargs=projector_kwargs,
            device=torch.device(pytest_device),
        )
        score = attributor.attribute(train_loader, test_loader)
        assert score.shape == (len(train_loader.dataset), len(test_loader.dataset))
        assert torch.count_nonzero(score) == len(train_loader.dataset) * len(
            test_loader.dataset,
        )

        shutil.rmtree(path)

    def test_tracin(self):
        """Test for TracIn without projectors."""
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

        # the function directly operate on batches of images/labels
        # not on dataloader anymore to allow vmap usage
        # need to prepare a batch dim in the begining
        def f(params, image_label_pair):
            image, label = image_label_pair
            image_t = image.unsqueeze(0)
            label_t = label.unsqueeze(0)
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image_t)
            return loss(yhat, label_t)

        # to simlulate multiple checkpoints
        model_1 = train_mnist_lr(train_loader, epoch_num=1)
        model_2 = train_mnist_lr(train_loader, epoch_num=2)
        path = Path("./ckpts")
        if not path.exists():
            path.mkdir(parents=True)
        torch.save(model_1.state_dict(), path / "model_1.pt")
        torch.save(model_2.state_dict(), path / "model_2.pt")

        checkpoint_list = ["ckpts/model_1.pt", "ckpts/model_2.pt"]

        task = AttributionTask(
            loss_func=f,
            model=model,
            checkpoints=checkpoint_list,
        )

        pytest_device = "cpu"
        # test with no projector list
        attributor = TracInAttributor(
            task=task,
            weight_list=torch.ones(len(checkpoint_list)),
            normalized_grad=True,
            device=torch.device(pytest_device),
        )
        score = attributor.attribute(train_loader, test_loader)
        assert score.shape == (len(train_loader.dataset), len(test_loader.dataset))
        assert torch.count_nonzero(score) == len(train_loader.dataset) * len(
            test_loader.dataset,
        )

        shutil.rmtree(path)

    def test_mnist_lr_multi_ckpts_grad(self):
        """Test for gradient computation correctness for mnist lr."""
        train_dataset = TensorDataset(
            torch.randn(20, 1, 28, 28),
            torch.randint(0, 10, (20,)),
        )

        train_loader = DataLoader(train_dataset, batch_size=4)

        model = train_mnist_lr(train_loader)

        @flatten_func(model)
        def f(params, image_label_pair):
            image, label = image_label_pair
            image_t = image.unsqueeze(0)
            label_t = label.unsqueeze(0)
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image_t)
            return loss(yhat, label_t)

        grad_func = vmap(grad(f), in_dims=(None, 0))

        # to simlulate multiple checkpoints
        model_1 = train_mnist_lr(train_loader, epoch_num=1)
        model_2 = train_mnist_lr(train_loader, epoch_num=2)
        path = Path("./ckpts")
        if not path.exists():
            path.mkdir(parents=True)
        torch.save(model_1.state_dict(), path / "model_1.pt")
        torch.save(model_2.state_dict(), path / "model_2.pt")

        checkpoint_list = ["ckpts/model_1.pt", "ckpts/model_2.pt"]

        # correct version: no param storage
        ckpt_grad_1 = []
        for checkpoint in checkpoint_list:
            # load checkpoint to the model
            model.load_state_dict(torch.load(checkpoint))
            model.eval()
            # get the model parameter
            parameters = {k: v.detach() for k, v in model.named_parameters()}

            grad_cache = []
            for train_batch_data in train_loader:
                # get gradient of train
                grad_t = grad_func(flatten_params(parameters), train_batch_data)
                grad_cache.append(grad_t)
            ckpt_grad_1.append(torch.cat(grad_cache, dim=0))

        # incorrect version: param storage
        ckpt_grad_2 = []
        param_list = []
        for checkpoint in checkpoint_list:
            # load checkpoint to the model
            model.load_state_dict(torch.load(checkpoint))
            model.eval()
            # get the model parameter
            parameters = {k: copy.deepcopy(v) for k, v in model.named_parameters()}

            param_list.append(parameters)
        for param in param_list:
            grad_cache = []
            for train_batch_data in train_loader:
                # get gradient of train
                grad_t = grad_func(flatten_params(param), train_batch_data)
                grad_cache.append(grad_t)
            ckpt_grad_2.append(torch.cat(grad_cache, dim=0))

        # check closeness of gradient result
        # match for mnist exp
        for idx in range(len(checkpoint_list)):
            assert torch.allclose(ckpt_grad_1[idx], ckpt_grad_2[idx])

        shutil.rmtree(path)

    def test_mnist_mlp_multi_ckpts_grad(self):
        """Test for gradient computation correctness for mnist mlp."""
        train_dataset = TensorDataset(
            torch.randn(20, 1, 28, 28),
            torch.randint(0, 10, (20,)),
        )

        train_loader = DataLoader(train_dataset, batch_size=4)

        model = train_mnist_mlp(train_loader)

        @flatten_func(model)
        def f(params, image_label_pair):
            image, label = image_label_pair
            image_t = image.unsqueeze(0)
            label_t = label.unsqueeze(0)
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image_t)
            return loss(yhat, label_t)

        grad_func = vmap(grad(f), in_dims=(None, 0))

        # to simlulate multiple checkpoints
        model_1 = train_mnist_mlp(train_loader, epoch_num=1)
        model_2 = train_mnist_mlp(train_loader, epoch_num=2)
        path = Path("./ckpts")
        if not path.exists():
            path.mkdir(parents=True)
        torch.save(model_1.state_dict(), path / "model_1.pt")
        torch.save(model_2.state_dict(), path / "model_2.pt")

        checkpoint_list = ["ckpts/model_1.pt", "ckpts/model_2.pt"]

        # correct version: no param storage
        ckpt_grad_1 = []
        for checkpoint in checkpoint_list:
            # load checkpoint to the model
            model.load_state_dict(torch.load(checkpoint))
            model.eval()
            # get the model parameter
            parameters = {k: v.detach() for k, v in model.named_parameters()}

            grad_cache = []
            for train_batch_data in train_loader:
                # get gradient of train
                grad_t = grad_func(flatten_params(parameters), train_batch_data)
                grad_cache.append(grad_t)
            ckpt_grad_1.append(torch.cat(grad_cache, dim=0))

        # incorrect version: param storage
        ckpt_grad_2 = []
        param_list = []
        for checkpoint in checkpoint_list:
            # load checkpoint to the model
            model.load_state_dict(torch.load(checkpoint))
            model.eval()
            # get the model parameter
            parameters = {k: copy.deepcopy(v) for k, v in model.named_parameters()}

            param_list.append(parameters)
        for param in param_list:
            grad_cache = []
            for train_batch_data in train_loader:
                # get gradient of train
                grad_t = grad_func(flatten_params(param), train_batch_data)
                grad_cache.append(grad_t)
            ckpt_grad_2.append(torch.cat(grad_cache, dim=0))

        # check closeness of gradient result
        # match for mnist exp
        for idx in range(len(checkpoint_list)):
            assert torch.allclose(ckpt_grad_1[idx], ckpt_grad_2[idx])

        shutil.rmtree(path)

    def test_cifar2_multi_ckpts_grad(self):
        """Test for gradient computation correctness for cifar2."""
        train_dataset = TensorDataset(
            torch.randn(20, 3, 32, 32),
            torch.randint(0, 2, (20,)),
        )

        train_loader = DataLoader(train_dataset, batch_size=4)

        model = train_cifar_resnet9(train_loader)

        @flatten_func(model)
        def f(params, image_label_pair):
            image, label = image_label_pair
            image_t = image.unsqueeze(0)
            label_t = label.unsqueeze(0)
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model, params, image_t)
            return loss(yhat, label_t)

        grad_func = vmap(grad(f), in_dims=(None, 0))

        # to simlulate multiple checkpoints
        model_1 = train_cifar_resnet9(train_loader, num_epochs=1)
        model_2 = train_cifar_resnet9(train_loader, num_epochs=2)
        path = Path("./ckpts")
        if not path.exists():
            path.mkdir(parents=True)
        torch.save(model_1.state_dict(), path / "model_1.pt")
        torch.save(model_2.state_dict(), path / "model_2.pt")

        checkpoint_list = ["ckpts/model_1.pt", "ckpts/model_2.pt"]

        # correct version: no param storage
        ckpt_grad_1 = []
        for checkpoint in checkpoint_list:
            # load checkpoint to the model
            model.load_state_dict(torch.load(checkpoint))
            model.eval()
            # get the model parameter
            parameters = {k: v.detach() for k, v in model.named_parameters()}

            grad_cache = []
            for train_batch_data in train_loader:
                # get gradient of train
                grad_t = grad_func(flatten_params(parameters), train_batch_data)
                grad_cache.append(grad_t)
            ckpt_grad_1.append(torch.cat(grad_cache, dim=0))

        # incorrect version: param storage
        ckpt_grad_2 = []
        param_list = []
        for checkpoint in checkpoint_list:
            # load checkpoint to the model
            model.load_state_dict(torch.load(checkpoint))
            model.eval()
            # get the model parameter
            # need deepcopy to
            parameters = {k: copy.deepcopy(v) for k, v in model.named_parameters()}
            param_list.append(parameters)
        for param in param_list:
            grad_cache = []
            for train_batch_data in train_loader:
                # get gradient of train
                grad_t = grad_func(flatten_params(param), train_batch_data)
                grad_cache.append(grad_t)
            ckpt_grad_2.append(torch.cat(grad_cache, dim=0))

        # check closeness of gradient result
        # DOES NOT match for cifar2 exp (only match for the last ckpt no matter
        # how many chpts will have)

        # first checkpoint: not match
        assert not torch.allclose(ckpt_grad_1[0], ckpt_grad_2[0])

        # second (last) checkpoint: match
        assert torch.allclose(ckpt_grad_1[1], ckpt_grad_2[1])

        shutil.rmtree(path)
