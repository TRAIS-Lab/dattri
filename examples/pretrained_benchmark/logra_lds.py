import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from dattri.algorithm.logra import LoGraAttributor
from dattri.benchmark.load import load_benchmark
from dattri.metric import lds
from dattri.task import AttributionTask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()

    # download the pre-trained benchmark
    # includes some trained model and ground truth
    model_details, groundtruth = load_benchmark(
        model="mlp", dataset="mnist", metric="lds"
    )

    def f(model, batch, device):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        return nn.functional.cross_entropy(outputs, targets)

    projector_kwargs = {
        "device": args.device,
        "proj_dim": 32,  # projection dimension = 32 * 32 = 1024
        "use_half_precision": False,
        "proj_max_batch_size": 32,
    }

    task = AttributionTask(
        model=model_details["model"].to(args.device),
        loss_func=f,
        checkpoints=model_details["models_full"][0],  # here we use one full model
    )

    attributor = LoGraAttributor(
        task=task,
        device=args.device,
        damping=5e-3,
        offload="cpu",
        projector_kwargs=projector_kwargs,
    )
    attributor.cache(
        DataLoader(
            model_details["train_dataset"],
            batch_size=5000,
            sampler=model_details["train_sampler"],
        )
    )

    score = attributor.attribute(
        DataLoader(
            model_details["train_dataset"],
            batch_size=5000,
            sampler=model_details["train_sampler"],
        ),
        DataLoader(
            model_details["test_dataset"],
            batch_size=5000,
            sampler=model_details["test_sampler"],
        ),
    )

    lds_score = lds(score, groundtruth)[0]
    print("lds:", torch.mean(lds_score[~torch.isnan(lds_score)]))
