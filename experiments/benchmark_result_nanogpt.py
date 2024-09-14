import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import dattri
from dattri.algorithm.tracin import TracInAttributor
from dattri.algorithm.trak import TRAKAttributor
from dattri.benchmark.load import load_benchmark
from dattri.benchmark.models.nanoGPT.model import GPT, GPTConfig
from dattri.metric import lds
from dattri.task import AttributionTask

data_dir = Path(dattri.__file__).parent / Path(
    "benchmark/datasets/shakespeare_char",
)


def load_model_from_checkpoint(ckpt_path, device):
    # TODO: This forces some extra memory usage
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod.") :]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


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

    model_details, groundtruth = load_benchmark(
        model="nanogpt",
        dataset="shakespeare",
        metric="lds",
    )

    checkpoint = torch.load(model_details["models_half"][0], map_location=args.device)
    model_args = checkpoint["model_args"]
    print(model_args)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(args.device)
    model.eval()

    train_loader = DataLoader(
        model_details["train_dataset"], batch_size=32, shuffle=False
    )
    val_loader = DataLoader(model_details["test_dataset"], batch_size=32, shuffle=False)

    if args.method in ["TRAK-1", "TRAK-10", "TRAK-50"]:
        ind = int(args.method.split("-")[1])

        def loss_func(params, data_target_pair):
            x, y = data_target_pair
            x_t = x.unsqueeze(0)
            y_t = y.unsqueeze(0)
            _, loss = torch.func.functional_call(model, params, (x_t, y_t))
            logp = -loss
            return logp - torch.log(1 - torch.exp(logp))

        def correctness_p(params, image_label_pair):
            x, y = image_label_pair
            x_t = x.unsqueeze(0)
            y_t = y.unsqueeze(0)
            _, loss = torch.func.functional_call(model, params, (x_t, y_t))
            p = torch.exp(-loss)
            return p

        checkpoints_list = []
        for checkpoint in model_details["models_half"][0:ind]:
            checkpoints_list.append(
                load_model_from_checkpoint(checkpoint, args.device),
            )

        task = AttributionTask(
            loss_func=loss_func, model=model, checkpoints=checkpoints_list
        )

        projector_kwargs = {
            "proj_dim": 2048,
            "device": args.device,
            "use_half_precision": False,
        }

        attributor = TRAKAttributor(
            task=task,
            correct_probability_func=correctness_p,
            device=args.device,
            projector_kwargs=projector_kwargs,
        )

        with torch.no_grad():
            attributor.cache(train_loader)
            score = attributor.attribute(val_loader)

    if args.method in ["TracIn", "Grad-Dot", "Grad-Cos"]:
        normalized_grad = False
        if args.method == "Grad-Cos":
            normalized_grad = True

        ensemble = 1
        if args.method == "TracIn":
            ensemble = 10

        projector_kwargs = {}

        checkpoints_list = []
        for checkpoint in model_details["models_half"][0:ensemble]:
            checkpoints_list.append(
                load_model_from_checkpoint(checkpoint, args.device),
            )

        def loss_tracin(params, data_target_pair):
            x, y = data_target_pair
            x_t = x.unsqueeze(0)
            y_t = y.unsqueeze(0)
            _, loss = torch.func.functional_call(model, params, (x_t, y_t))
            return loss

        task = AttributionTask(
            model=model,
            loss_func=loss_tracin,
            checkpoints=checkpoints_list,
        )

        attributor = TracInAttributor(
            task=task,
            weight_list=torch.ones(ensemble) * 1e-3,
            normalized_grad=normalized_grad,
            device=args.device,
        )
        with torch.no_grad():
            score = attributor.attribute(train_loader, val_loader)

    best_result = 0
    best_config = None

    # compute metrics
    metric_score = lds(score, groundtruth)[0]
    metric_score = torch.mean(metric_score[~torch.isnan(metric_score)])

    print("lds:", metric_score)
    if metric_score > best_result:
        best_result = metric_score
        best_config = projector_kwargs
    print("complete\n")

    print(args.method, "RESULT:", best_config, "lds:", best_result)
