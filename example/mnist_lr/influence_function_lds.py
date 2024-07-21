from dattri.benchmark.load import load_benchmark
from dattri.task import AttributionTask
from dattri.metrics.metrics import lds
from dattri.algorithm.influence_function import IFAttributorCG
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

model_details, groundtruth = load_benchmark(model="mlp",
                                            dataset="mnist",
                                            metric="lds")

def f(params, data_target_pair):
    image, label = data_target_pair
    loss = nn.CrossEntropyLoss()
    yhat = torch.func.functional_call(model_details["model"], params, image)
    return loss(yhat, label.long())

task = AttributionTask(
    model=model_details["model"].cuda(),
    loss_func=f,
    checkpoints=model_details["models_full"][0]  # here we use one full model
)

attributor = IFAttributorCG(task=task, device="cuda", regularization=5e-3,  max_iter=10)
attributor.cache(DataLoader(model_details["train_dataset"],
                            shuffle=False, batch_size=500,
                            sampler=model_details["train_sampler"]))

with torch.no_grad():
    score = attributor.attribute(
        DataLoader(model_details["train_dataset"],
                shuffle=False, batch_size=5000,
                sampler=model_details["train_sampler"]),
        DataLoader(model_details["test_dataset"],
                shuffle=False, batch_size=5000,
                sampler=model_details["test_sampler"])
    )

lds_score = lds(score.cpu().T, groundtruth)[0]
print("lds:", torch.mean(lds_score[~torch.isnan(lds_score)]))
