import argparse


import torch
from torch.utils.data import DataLoader

from dattri.algorithm.tracin import TracInAttributor
from dattri.algorithm.trak import TRAKAttributor
from dattri.metric import lds
from dattri.benchmark.load import load_benchmark
from dattri.task import AttributionTask
from dattri.benchmark.models.MusicTransformer.utilities.constants import TOKEN_PAD


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--method",
        type=str,
        default="TRAK-10",
        choices=[
            "TRAK-1",
            "TRAK-10",
            "TRAK-50",
            "TracIn",
            "Grad-Dot",
            "Grad-Cos",
        ],
        help="The TDA method to benchmark.",
    )
    argparser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    args = argparser.parse_args()

    print(args)


    # 1. load the benchmark

    model_details, groundtruth = load_benchmark(
        model="musictransformer",
        dataset="maestro",
        metric="lds",
    )

    # 2. prepare task for each method

    train_loader = DataLoader(
        model_details["train_dataset"],
        batch_size=64,
        shuffle=False,
        sampler=model_details["train_sampler"],
    )
    test_loader = DataLoader(
        model_details["test_dataset"],
        batch_size=64,
        shuffle=False,
    )

    if args.method in ["TRAK-1", "TRAK-10", "TRAK-50"]:
        ind = int(args.method.split("-")[1])
        def loss_trak(params, data_target_pair):
            x, y = data_target_pair
            x_t = x.unsqueeze(0)
            y_t = y.unsqueeze(0)
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=TOKEN_PAD, reduction='none')

            output = torch.func.functional_call(model_details["model"], params, x_t)
            output_last = output[:, -1, :]
            y_last = y_t[:, -1]

            logp = -loss_fn(output_last, y_last)
            logit_func = logp - torch.log(1 - torch.exp(logp))
            return logit_func.squeeze(0)


        def correctness_p(params, data_target_pair):
            x, y = data_target_pair
            x_t = x.unsqueeze(0)
            y_t = y.unsqueeze(0)
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=TOKEN_PAD, reduction='none')

            output = torch.func.functional_call(model_details["model"], params, x_t)
            output_last = output[:, -1, :]
            y_last = y_t[:, -1]
            logp = -loss_fn(output_last, y_last)

            return torch.exp(logp)

        task = AttributionTask(
            loss_func=loss_trak,
            model=model_details["model"].to(args.device),
            checkpoints=model_details["models_half"][0:ind]
        )

        projector_kwargs = {
            "proj_dim": 2048,
            "device": args.device,
        }

        attributor = TRAKAttributor(
            task=task,
            correct_probability_func=correctness_p,
            device=args.device,
            projector_kwargs=projector_kwargs,
        )

        with torch.no_grad():
            attributor.cache(train_loader)
            score = attributor.attribute(test_loader)

    if args.method in ["TracIn", "Grad-Dot", "Grad-Cos"]:
        normalized_grad = False
        if args.method == "Grad-Cos":
            normalized_grad = True

        ensemble = 1
        if args.method == "TracIn":
            ensemble = 10

        projector_kwargs = {}

        def loss_tracin(params, data_target_pair):
            x, y = data_target_pair
            x_t = x.unsqueeze(0)
            y_t = y.unsqueeze(0)
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=TOKEN_PAD, reduction='none')

            output = torch.func.functional_call(model_details["model"], params, x_t)
            output_last = output[:, -1, :]
            y_last = y_t[:, -1]

            return loss_fn(output_last, y_last).squeeze(0)

        task = AttributionTask(
            model=model_details["model"].to(args.device),
            loss_func=loss_tracin,
            checkpoints=model_details["models_full"][0:10]
            if args.method == "TracIn"
            else model_details["models_full"][0],
        )

        attributor = TracInAttributor(
            task=task,
            weight_list=torch.ones(ensemble) * 1e-3,
            normalized_grad=normalized_grad,
            device=args.device,
        )
        with torch.no_grad():
            score = attributor.attribute(train_loader, test_loader)


    best_result = 0
    best_config = None
    metric_score = lds(score, groundtruth)[0]
    metric_score = torch.mean(metric_score[~torch.isnan(metric_score)])

    print("lds:", metric_score)
    if metric_score > best_result:
        best_result = metric_score
        best_config = projector_kwargs
    print("complete\n")

    print(args.method, "RESULT:", best_config, "lds:", best_result)
