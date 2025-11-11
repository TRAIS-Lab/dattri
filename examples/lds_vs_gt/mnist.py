import argparse
from functools import partial

import torch
from torch import nn

from dattri.model_util.retrain import retrain_lds
from dattri.metric.ground_truth import calculate_lds_ground_truth
from dattri.benchmark.utils import SubsetSampler
from dattri.task import AttributionTask
from dattri.metric import lds
from dattri.benchmark.datasets.mnist import (
    create_mnist_dataset, 
    train_mnist_mlp, train_mnist_lr, 
    loss_mnist_mlp, loss_mnist_lr
)
from dattri.algorithm.influence_function import (
    IFAttributorExplicit,
    IFAttributorCG,
    IFAttributorLiSSA,
    IFAttributorDataInf,
    IFAttributorArnoldi,
)

# partially define attributors
# TODO: regularization
ATTRIBUTOR_MAP = {
    "explicit": partial(IFAttributorExplicit, regularization=0.01),
    "cg": partial(IFAttributorCG, regularization=0.01),
    "lissa": partial(IFAttributorLiSSA, recursion_depth=100),
    "datainf": partial(IFAttributorDataInf, regularization=0.01),
    "arnoldi": partial(IFAttributorArnoldi, regularization=0.01, max_iter=10),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mnist_mlp")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--method", type=str, default="cg")
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    if args.model not in ["mnist_mlp", "mnist_lr"]:
        raise ValueError("The model is not supported.")
    
    ##############################
    # Data and model preparation
    ##############################
    # load the training dataset
    dataset, dataset_test = create_mnist_dataset("./data")

    # dataloaders
    # TODO: sampler size
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        sampler=SubsetSampler(range(1000)),
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=500,
        sampler=SubsetSampler(range(500)),
    )
    train_loader_attr = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
        sampler=SubsetSampler(range(1000)),
    )

    # prepare the training function and target function
    if args.model == "mnist_mlp":
        train_func = train_mnist_mlp
        target_func = loss_mnist_mlp
    elif args.model == "mnist_lr":
        train_func = train_mnist_lr
        target_func = loss_mnist_lr

    ##############################
    # Influence function score
    ##############################
    # train the full model
    model = train_func(train_loader)
    model.to(args.device)
    model.eval()

    # define loss function
    def f(params, data_target_pair):
        image, label = data_target_pair
        loss = nn.CrossEntropyLoss()
        yhat = torch.func.functional_call(model, params, image)
        return loss(yhat, label.long())

    # get influence function score
    task = AttributionTask(loss_func=f, model=model, checkpoints=model.state_dict())
    attributor = ATTRIBUTOR_MAP[args.method](
        task=task,
        device=args.device,
    )
    attributor.cache(train_loader_attr)
    with torch.no_grad():
        score = attributor.attribute(train_loader_attr, test_loader)

    ##############################
    # Ground truth
    ##############################
    # retrain the model for the Linear Datamodeling Score (LDS) metric calculation
    # TODO: num_subsets, num_runs_per_subset
    retrain_lds(
        train_func,
        train_loader,
        args.path,
        num_subsets=10,
        num_runs_per_subset=2,
        seed=1
    )

    # calculate the ground-truth values for the Linear Datamodeling Score (LDS) metric
    ground_truth = calculate_lds_ground_truth(
        target_func,
        args.path,
        test_loader
    )

    ##############################
    # Calculate and print LDS score
    ##############################
    lds_score = lds(score, ground_truth)[0]
    print("lds:", torch.mean(lds_score[~torch.isnan(lds_score)]))