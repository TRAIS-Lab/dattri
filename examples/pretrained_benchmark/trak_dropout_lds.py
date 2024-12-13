import argparse
from pathlib import PosixPath

import torch
from torch import nn
from torch.utils.data import DataLoader

from dattri.algorithm.trak import TRAKAttributor
from dattri.benchmark.load import load_benchmark
from dattri.metric import lds
from dattri.task import AttributionTask
from dattri.model_util.dropout import activate_dropout

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()

    # download the pre-trained benchmark
    # includes some trained model and ground truth
    model_details, groundtruth = load_benchmark(
        model="mlp", dataset="mnist", metric="lds"
    )

    # Here we use 0.1 dropout rate on the model
    model = activate_dropout(model_details["model"], dropout_prob=0.1)

    def dropout_checkpoint_load_func(model, checkpoint):
        if isinstance(checkpoint, (str, PosixPath)):
            checkpoint = torch.load(checkpoint,
                                    map_location=next(model.parameters()).device,)
        model.load_state_dict(checkpoint)
        model.eval()
        model = activate_dropout(model, dropout_prob=0.1)
        return model

    def f(params, data_target_pair):
        image, label = data_target_pair
        image_t = image.unsqueeze(0)
        label_t = label.unsqueeze(0)
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model, params, image_t)
        logp = -loss(yhat, label_t)
        return logp - torch.log(1 - torch.exp(logp))

    def m(params, image_label_pair):
        image, label = image_label_pair
        image_t = image.unsqueeze(0)
        label_t = label.unsqueeze(0)
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model, params, image_t)
        p = torch.exp(-loss(yhat, label_t))
        return p

    # here we use 10 same checkpoints
    task = AttributionTask(
        model=model.to(args.device),
        loss_func=f,
        checkpoints=[model_details["models_half"][0]] * 10,
        checkpoints_load_func = dropout_checkpoint_load_func
    )

    attributor = TRAKAttributor(
        task=task,
        correct_probability_func=m,
        device=args.device,
    )

    with torch.no_grad():
        attributor.cache(
            DataLoader(
                model_details["train_dataset"],
                batch_size=5000,
                sampler=model_details["train_sampler"],
            )
        )

        score = attributor.attribute(
            DataLoader(
                model_details["test_dataset"],
                batch_size=5000,
                sampler=model_details["test_sampler"],
            ),
        )

    lds_score = lds(score, groundtruth)[0]
    print("lds:", torch.mean(lds_score[~torch.isnan(lds_score)]))
