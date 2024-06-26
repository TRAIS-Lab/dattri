"""This example shows how to use the IF to detect noisy labels in the MNIST."""

# ruff: noqa
import argparse
from functools import partial
import time


import numpy as np
import torch
from torch import nn


from dattri.algorithm.influence_function import IFAttributor
from dattri.algorithm.tracin import TracInAttributor
from dattri.algorithm.trak import TRAKAttributor
from dattri.benchmark.datasets.cifar2 import (
    create_cifar2_dataset,
    train_cifar2_resnet9,
)
from dattri.benchmark.datasets.cifar2.cifar2_resnet9 import create_resnet9_model, loss_cifar2_resnet9
from dattri.benchmark.utils import SubsetSampler
from dattri.func.utils import flatten_func
from dattri.metrics.metrics import lds
from dattri.metrics.ground_truth import calculate_lds_ground_truth


IHVP_CONFIG = {
    "explicit": [
        {"regularization": r} for r in [1e0, 1e-1, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5]
    ],
    "cg": [
        {"regularization": r, "max_iter": 10}
        for r in [1e0, 1e-1, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5]
    ],
    "lissa": [
        {"recursion_depth": 100, "batch_size": 100},
        {"recursion_depth": 100, "batch_size": 50},
        {"recursion_depth": 100, "batch_size": 10},
    ],
    "arnoldi": [
        {"regularization": r, "max_iter": 50}
        for r in [1e0, 1e-1, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5]
    ],
}

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="cifar2")
    argparser.add_argument("--model", type=str, default="resnet9")
    argparser.add_argument("--method", type=str, default="explicit")
    args = argparser.parse_args()

    print(args)
    # create dataset
    dataset_train, dataset_test = create_cifar2_dataset("./data")

    # the exp size
    train_size = 5000
    test_size = 500
    # train/test dataloader
    train_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64,
        sampler=SubsetSampler(range(train_size)),
    )
    train_loader_full = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64,
        sampler=SubsetSampler(range(train_size)),
    )
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64,
        sampler=SubsetSampler(range(train_size)),
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=64,
        sampler=SubsetSampler(range(test_size)),
    )

    model = create_resnet9_model()
    model.cuda()
    model.eval()

    @flatten_func(model)
    def f(params, data_target_pair):
        image, label = data_target_pair
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model, params, image)
        return loss(yhat, label.long())

    @flatten_func(model)
    def f_tracin(params, data_target_pair):
        image, label = data_target_pair
        image_t = image.unsqueeze(0)
        label_t = label.unsqueeze(0)
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model, params, image_t)
        return loss(yhat, label_t.long())

    @flatten_func(model)
    def f_0(params, data_target_pair):
        image, label = data_target_pair
        image_t = image.unsqueeze(0)
        label_t = label.unsqueeze(0)
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model, params, image_t)
        logp = -loss(yhat, label_t)
        return logp - torch.log(1 - torch.exp(logp))

    @flatten_func(model)
    def m(params, image_label_pair):
        image, label = image_label_pair
        image_t = image.unsqueeze(0)
        label_t = label.unsqueeze(0)
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model, params, image_t)
        p = torch.exp(-loss(yhat, label_t.long()))
        return p

    def f_rps(pre_activation_list, label_list):
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(pre_activation_list, label_list)

    # the eval size
    lds_eval_size = 50

    model_params = {k: p for k, p in model.named_parameters() if p.requires_grad}
    best_result = 0
    best_config = None
    if args.method in ["explicit", "cg", "lissa", "arnoldi"]:
        for ihvp_config in IHVP_CONFIG[args.method]:
            print(ihvp_config)
            attributor = IFAttributor(
                target_func=f,
                params=model_params,
                ihvp_solver=args.method,
                ihvp_kwargs=ihvp_config,
                device=torch.device("cuda"),
            )
            attributor.cache(train_loader)
            torch.cuda.reset_peak_memory_stats("cuda")
            with torch.no_grad():
                score = attributor.attribute(train_loader, test_loader)
            peak_memory = torch.cuda.max_memory_allocated("cuda") / 1e6  # Convert to MB
            print(f"Peak memory usage: {peak_memory} MB")

            target_values = torch.load(
                "/scratch/bbyo/tli3/cifar2_lds/groundtruth/target_values.pt"
            )
            indices = torch.load(
                "/scratch/bbyo/tli3/cifar2_lds/groundtruth/training_indices.pt"
            )
            print(score.shape)
            loo_corr_v = lds(
                score.cpu().T, (-target_values[lds_eval_size:], indices[lds_eval_size:])
            )[0]

            sum_val = 0
            counter = 0
            for i in range(test_size):
                if np.isnan(loo_corr_v[i]):
                    continue
                sum_val += loo_corr_v[i]
                counter += 1
            print(sum_val, counter)
            if counter == 0:
                continue
            print(sum_val / counter)
            if sum_val / counter > best_result:
                best_result = sum_val / counter
                best_config = ihvp_config
            print("complete\n")
        print(args.method, "RESULT:", best_config, "lds:", best_result)

    if args.method == "TRAK":
        # create model
        model = create_resnet9_model()
        model.to("cuda")

        for proj_dim, ensemble in [(512, 1)]:
            print("proj_dim, ensemble", proj_dim, ensemble)
            projector_kwargs = {
                "proj_dim": proj_dim,
                "device": "cuda",
            }
            params = []
            
            # collect retrained models' paths
            for i in range(ensemble):
                params.append(f"/home/shared/dattri-dataset/cifar2_lds_test/models/{i}/model_weights_0.pt")

            attributor = TRAKAttributor(
                f_0,
                m,
                model=model,
                params=params,
                device=torch.device("cuda"),
                projector_kwargs=projector_kwargs,
            )
            attributor.cache(train_loader)
            torch.cuda.reset_peak_memory_stats("cuda")
            with torch.no_grad():
                score = attributor.attribute(test_loader)
            peak_memory = torch.cuda.max_memory_allocated("cuda") / 1e6  # Convert to MB
            print(f"Peak memory usage: {peak_memory} MB")

            # get retrained models' location
            retrain_dir = "/home/shared/dattri-dataset/cifar2_lds_test/models"
            # get model output and indices
            target_values, indices = calculate_lds_ground_truth(partial(loss_cifar2_resnet9, device="cuda"), retrain_dir, test_loader)
            # compute LDS value
            lds_value = lds(score.T.cpu(), (-target_values, indices))[0]
            # compute average LDS
            sum_val = 0
            counter = 0
            for i in range(test_size):
                if np.isnan(lds_value[i]):
                    continue
                sum_val += lds_value[i]
                counter += 1
            print(sum_val, counter)
            if counter == 0:
                continue
            if sum_val / counter > best_result:
                best_result = sum_val / counter
                best_config = (proj_dim, ensemble)
            print(sum_val / counter)

            print("complete\n")
            print(args.method, "RESULT:", best_config, "lds:", best_result)

    if args.method == "TracIn":
        # create model
        model = create_resnet9_model()
        model.to("cuda")

        for ensemble, normalized_grad in [(1, False), (10, False)]:
            print("ensemble, normalize grad: ", ensemble, normalized_grad)
            ensemble = ensemble
            params = []
            for i in range(ensemble):
                params.append(f"/home/shared/dattri-dataset/cifar2_lds_test/models/{i}/model_weights_0.pt")

            proj_kwargs = {
                "proj_dim": 512,
                "proj_max_batch_size": 32,
                "proj_seed": 0,
                "device": "cuda",
                "use_half_precision": False,
            }
            attributor = TracInAttributor(
                f_tracin,
                model,
                params_list=params,
                weight_list=torch.ones(ensemble) * 1e-3,
                normalized_grad=normalized_grad,
                projector_kwargs=proj_kwargs,
                device=torch.device("cuda"),
            )
            torch.cuda.reset_peak_memory_stats("cuda")
            with torch.no_grad():
                score = attributor.attribute(train_loader, test_loader)
            peak_memory = torch.cuda.max_memory_allocated("cuda") / 1e6  # Convert to MB
            print(f"Peak memory usage: {peak_memory} MB")

            # get retrained models' location
            retrain_dir = "/home/shared/dattri-dataset/cifar2_lds_test/models"
            # get model output and indices
            target_values, indices = calculate_lds_ground_truth(partial(loss_cifar2_resnet9, device="cuda"), retrain_dir, test_loader)
            # compute LDS value
            lds_value = lds(score.T.cpu(), (-target_values, indices))[0]

            # compute average LDS
            sum_val = 0
            counter = 0
            for i in range(test_size):
                if np.isnan(lds_value[i]):
                    continue
                sum_val += lds_value[i]
                counter += 1
            print(sum_val, counter)
            if counter == 0:
                continue
            # if sum_val / counter > best_result:
            #     best_result = sum_val / counter
            #     best_config = (proj_dim, ensemble)
            print(sum_val / counter)

            print("complete\n")
            # print(args.method, "RESULT:", best_config, "lds:", best_result)

    if args.method == "RPS":
        model = train_cifar2_resnet9(train_loader, device="cuda")
        model.eval()
        model.to("cuda")
        model.load_state_dict(
            torch.load(
                f"/scratch/bbyo/tli3/cifar2-checkpoints/checkpoint_without_dropout/checkpoint_0.pt"
            )
        )

        for l2 in [1, 1e-1, 1e-2, 1e-3, 1e-4]:
            attributor = RPSAttributor(
                f_rps,
                model=model,
                final_linear_layer_name="linear",
                nomralize_preactivate="True",
                l2_strength=l2,
                device="cuda",
            )
            # attributor.cache(train_loader)
            start_attribute = time.time()
            torch.cuda.reset_peak_memory_stats("cuda")
            score = attributor.attribute(train_loader, test_loader)
            peak_memory = torch.cuda.max_memory_allocated("cuda") / 1e6  # Convert to MB
            print(f"Peak memory usage: {peak_memory} MB")
            end_attribute = time.time()
            print("Attribution time: ", end_attribute - start_attribute)
            # print(score.shape)

            index_to_value_dict = {value: idx for idx, value in enumerate(all_index)}
            indices = []
            # for i in range(50,100):
            for i in range(50):
                file_name = f"/scratch/bbyo/tli3/cifar2-checkpoints/checkpoint_without_dropout/selected_indices_seed_{i}.txt"
                with open(file_name, "r") as f:
                    contents = f.read()
                    numbers = contents.splitlines()
                    indice = torch.tensor(
                        [index_to_value_dict[int(num)] for num in numbers]
                    )
                    # print("indice: ", indice)
                    indices.append(indice)
            indices = torch.stack(indices)
            # print("indices shape: ", indices.shape)
            print("indices: ", indices)

            target_values = []
            # for i in range(50,100):
            for i in range(50):
                file_name = f"/scratch/bbyo/tli3/cifar2-checkpoints/checkpoint_without_dropout/model_output_checkpoint_{i}.pt"
                target_value = torch.load(file_name).detach().cpu()
                # print("target: ", target_value)
                target_values.append(target_value)
            target_values = torch.stack(target_values)
            # print("target value shape: ", target_values.shape)
            print("target value: ", target_values)

            print(score.shape)
            # loo_corr_v = lds(score.cpu().T, (-target_values[50:], indices[50:]))[0]
            loo_corr_v = lds(score.cpu().T, (-target_values, indices))[0]

            sum_val = 0
            counter = 0
            for i in range(500):
                if np.isnan(loo_corr_v[i]):
                    continue
                sum_val += loo_corr_v[i]
                counter += 1
            print(sum_val, counter)
            if counter == 0:
                continue
            if sum_val / counter > best_result:
                best_result = sum_val / counter
                best_config = l2
            print(sum_val / counter)
            print("complete\n")
            print(args.method, "RESULT:", best_config, "lds:", best_result)