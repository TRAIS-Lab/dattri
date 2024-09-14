import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from dattri.algorithm.trak import TRAKAttributor
from dattri.benchmark.load import load_benchmark
from dattri.metric import loo_corr
from dattri.task import AttributionTask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()

    # download the pre-trained benchmark
    # includes some trained model and ground truth
    model_details, groundtruth = load_benchmark(
        model="lr", dataset="mnist", metric="loo"
    )
    model = model_details["model"].to(args.device)
    model.eval()

    def loss_func(params, data_target_pair):
        x, y = data_target_pair
        x_t = x.unsqueeze(0)
        y_t = y.unsqueeze(0)
        y_h = torch.func.functional_call(model, params, x_t)
        loss = nn.CrossEntropyLoss()
        logp = -loss(y_h, y_t)
        return logp - torch.log(1 - torch.exp(logp))

    def m(params, image_label_pair):
        image, label = image_label_pair
        image_t = image.unsqueeze(0)
        label_t = label.unsqueeze(0)
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model, params, image_t)
        p = torch.exp(-loss(yhat, label_t))
        return p

    projector_kwargs = {
        "device": args.device,
        "use_half_precision": False,
    }

    task = AttributionTask(
        model=model,
        loss_func=loss_func,
        checkpoints=model_details["models_half"][0],
    )

    attributor = TRAKAttributor(
        task=task,
        correct_probability_func=m,
        device=args.device,
        projector_kwargs=projector_kwargs,
    )

    with torch.no_grad():
        attributor.cache(
            DataLoader(
                model_details["train_dataset"],
                shuffle=False,
                batch_size=5000,
                sampler=model_details["train_sampler"],
            )
        )
        score = attributor.attribute(
            DataLoader(
                model_details["test_dataset"],
                shuffle=False,
                batch_size=500,
                sampler=model_details["test_sampler"],
            ),
        )

    loo_score = loo_corr(score, groundtruth)[0]
    print("loo:", torch.mean(loo_score[~torch.isnan(loo_score)]))
