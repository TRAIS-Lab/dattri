import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from dattri.algorithm.influence_function import IFAttributorCG
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

    def f(params, data_target_pair):
        image, label = data_target_pair
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model_details["model"], params, image)
        return loss(yhat, label.long())

    task = AttributionTask(
        model=model_details["model"].to(args.device),
        loss_func=f,
        checkpoints=model_details["models_full"][0],  # here we use one full model
    )

    attributor = IFAttributorCG(
        task=task, device=args.device, regularization=5e-3, max_iter=10
    )
    attributor.cache(
        DataLoader(
            model_details["train_dataset"],
            batch_size=5000,
            sampler=model_details["train_sampler"],
        )
    )

    with torch.no_grad():
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
