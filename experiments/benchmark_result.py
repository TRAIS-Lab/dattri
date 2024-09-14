"""This experiment benchmarks TDA methods on the MNIST-10 dataset."""

# ruff: noqa
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader


from dattri.algorithm.influence_function import (
    IFAttributorCG,
    IFAttributorLiSSA,
    IFAttributorArnoldi,
    IFAttributorExplicit,
)
from dattri.algorithm.tracin import TracInAttributor
from dattri.algorithm.trak import TRAKAttributor
from dattri.algorithm.rps import RPSAttributor
from dattri.metric import lds, loo_corr
from dattri.benchmark.load import load_benchmark
from dattri.task import AttributionTask


IHVP_SEARCH_SPACE = {
    "if-explicit": [
        {"regularization": r} for r in [1e0, 1e-1, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5]
    ],
    "if-cg": [
        {"regularization": r, "max_iter": 10}
        for r in [1e0, 1e-1, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5]
    ],
    "if-lissa": [
        {"recursion_depth": 100, "batch_size": 100},
        {"recursion_depth": 100, "batch_size": 50},
        {"recursion_depth": 100, "batch_size": 10},
    ],
    "if-arnoldi": [
        {"regularization": r, "max_iter": 50}
        for r in [1e0, 1e-1, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5]
    ],
}

ATTRIBUTOR_DICT = {
    "if-explicit": IFAttributorExplicit,
    "if-cg": IFAttributorCG,
    "if-lissa": IFAttributorLiSSA,
    "if-arnoldi": IFAttributorArnoldi,
    "TRAK-1": TRAKAttributor,
    "TRAK-10": TRAKAttributor,
    "TRAK-50": TRAKAttributor,
    "TracIn": TracInAttributor,
    "Grad-Dot": TracInAttributor,
    "Grad-Cos": TracInAttributor,
    "RPS": RPSAttributor,
}

METRICS_DICT = {
    "lds": lds,
    "loo": loo_corr,
}

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar2"],
        help="The dataset of the benchmark.",
    )
    argparser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["lr", "mlp", "resnet9"],
        help="The model of the benchmark.",
    )
    argparser.add_argument(
        "--method",
        type=str,
        default="if-explicit",
        choices=[
            "if-explicit",
            "if-cg",
            "if-lissa",
            "if-arnoldi",
            "TRAK-1",
            "TRAK-10",
            "TRAK-50",
            "TracIn",
            "Grad-Dot",
            "Grad-Cos",
            "RPS",
        ],
        help="The TDA method to benchmark.",
    )
    argparser.add_argument(
        "--metric",
        type=str,
        default="lds",
        choices=["lds", "loo"],
        help="The metric to evaluate the TDA method.",
    )
    argparser.add_argument(
        "--device", type=str, default="cuda", help="The device to run the experiment."
    )
    args = argparser.parse_args()

    print(args)

    # 1. load the benchmark

    model_details, groundtruth = load_benchmark(
        model=args.model, dataset=args.dataset, metric=args.metric
    )

    # 2. prepare task for each method

    train_loader_cache = DataLoader(
        model_details["train_dataset"],
        shuffle=False,
        batch_size=5000 if args.dataset == "mnist" else 64,
        sampler=model_details["train_sampler"],
    )
    train_loader = DataLoader(
        model_details["train_dataset"],
        shuffle=False,
        batch_size=500 if args.dataset == "mnist" else 64,
        sampler=model_details["train_sampler"],
    )
    test_loader = DataLoader(
        model_details["test_dataset"],
        shuffle=False,
        batch_size=500 if args.dataset == "mnist" else 64,
        sampler=model_details["test_sampler"],
    )

    if args.method in ["if-explicit", "if-cg", "if-lissa", "if-arnoldi"]:

        def loss_if(params, data_target_pair):
            image, label = data_target_pair
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model_details["model"], params, image)
            return loss(yhat, label.long())

        task = AttributionTask(
            model=model_details["model"].to(args.device),
            loss_func=loss_if,
            checkpoints=model_details["models_full"][0],
        )

    if args.method in ["TracIn", "Grad-Dot", "Grad-Cos"]:

        def loss_tracin(params, data_target_pair):
            image, label = data_target_pair
            image_t = image.unsqueeze(0)
            label_t = label.unsqueeze(0)
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model_details["model"], params, image_t)
            return loss(yhat, label_t.long())

        task = AttributionTask(
            model=model_details["model"].to(args.device),
            loss_func=loss_tracin,
            checkpoints=model_details["models_full"][0:10]
            if args.method == "TracIn"
            else model_details["models_full"][0],
        )

    if args.method in ["TRAK-1", "TRAK-10", "TRAK-50"]:

        def loss_trak(params, data_target_pair):
            image, label = data_target_pair
            image_t = image.unsqueeze(0)
            label_t = label.unsqueeze(0)
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model_details["model"], params, image_t)
            logp = -loss(yhat, label_t)
            return logp - torch.log(1 - torch.exp(logp))

        task = AttributionTask(
            model=model_details["model"].to(args.device),
            loss_func=loss_trak,
            checkpoints=model_details["models_half"][
                0 : int(args.method.split("-")[1])
            ],
        )

        def m_trak(params, image_label_pair):
            image, label = image_label_pair
            image_t = image.unsqueeze(0)
            label_t = label.unsqueeze(0)
            loss = nn.CrossEntropyLoss()
            yhat = torch.func.functional_call(model_details["model"], params, image_t)
            p = torch.exp(-loss(yhat, label_t.long()))
            return p

    if args.method == "RPS":

        def loss_rps(pre_activation_list, label_list):
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(pre_activation_list, label_list)

        task = AttributionTask(
            model=model_details["model"].to(args.device),
            loss_func=loss_rps,
            checkpoints=model_details["models_full"][0],
        )

    # 3. start running the TDA methods
    best_result = 0
    best_config = None

    if args.method in ["if-explicit", "if-cg", "if-lissa", "if-arnoldi"]:
        for ihvp_config in IHVP_SEARCH_SPACE[args.method]:
            print(ihvp_config)
            attributor = ATTRIBUTOR_DICT[args.method](
                task=task,
                device=args.device,
                **ihvp_config,
            )
            attributor.cache(train_loader_cache)
            with torch.no_grad():
                score = attributor.attribute(train_loader, test_loader)

            # compute metrics
            metric_score = METRICS_DICT[args.metric](score, groundtruth)[0]
            metric_score = torch.mean(metric_score[~torch.isnan(metric_score)])

            print(f"{args.metric}:", metric_score)
            if metric_score > best_result:
                best_result = metric_score
                best_config = ihvp_config
            print("complete\n")
        print(args.method, "RESULT:", best_config, f"{args.metric}:", best_result)

    if args.method in ["TRAK-1", "TRAK-10", "TRAK-50"]:
        projector_kwargs = {
            "proj_dim": 512,
            "device": args.device,
        }

        attributor = ATTRIBUTOR_DICT[args.method](
            task=task,
            correct_probability_func=m_trak,
            device=args.device,
            projector_kwargs=projector_kwargs,
        )
        attributor.cache(train_loader)
        with torch.no_grad():
            score = attributor.attribute(test_loader)

        # compute metrics
        metric_score = METRICS_DICT[args.metric](score, groundtruth)[0]
        metric_score = torch.mean(metric_score[~torch.isnan(metric_score)])

        print(f"{args.metric}:", metric_score)
        if metric_score > best_result:
            best_result = metric_score
            best_config = projector_kwargs
        print("complete\n")

        print(args.method, "RESULT:", best_config, f"{args.metric}:", best_result)

    if args.method in ["TracIn", "Grad-Dot", "Grad-Cos"]:
        normalized_grad = False
        if args.method == "Grad-Cos":
            normalized_grad = True

        ensemble = 1
        if args.method == "TracIn":
            ensemble = 10

        attributor = ATTRIBUTOR_DICT[args.method](
            task=task,
            weight_list=torch.ones(ensemble) * 1e-3,
            normalized_grad=normalized_grad,
            # projector_kwargs=proj_kwargs,
            device=args.device,
        )
        with torch.no_grad():
            score = attributor.attribute(train_loader, test_loader)

        # compute metrics
        metrics_score = METRICS_DICT[args.metric](score, groundtruth)[0]
        metrics_score = torch.mean(metrics_score[~torch.isnan(metrics_score)])

        if metrics_score > best_result:
            best_result = metrics_score
        print("complete\n")

        print(args.method, f"{args.metric}:", best_result)

    if args.method == "RPS":
        for l2 in [1, 1e-1, 1e-2, 1e-3, 1e-4]:
            attributor = ATTRIBUTOR_DICT[args.method](
                task=task,
                final_linear_layer_name="fc3" if args.model == "mlp" else "linear",
                normalize_preactivate=True,
                l2_strength=l2,
                device=args.device,
            )

            train_loader_train = DataLoader(
                model_details["train_dataset"],
                shuffle=False,
                batch_size=64,
                sampler=model_details["train_sampler"],
            )
            attributor.cache(train_loader_train)
            score = attributor.attribute(train_loader, test_loader)

            # compute metrics
            metrics_score = METRICS_DICT[args.metric](score, groundtruth)[0]
            metrics_score = torch.mean(metrics_score[~torch.isnan(metrics_score)])

            print(f"{args.metric}:", metrics_score)
            if metrics_score > best_result:
                best_result = metrics_score
                best_config = l2
            print("complete\n")

        print(args.method, "RESULT:", {"lr": l2}, f"{args.metric}:", best_result)
