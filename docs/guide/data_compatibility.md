# Data Type Compatibility for Loss and Target Functions

## Data structure to pass in Attributor
### Below is a comprehensive table of the data structure each attributior supports. The loss function each attributior takes is also provided.

|                      Family                       |                              Algorithms                              | tuple | list | dict | loss function |
| :-----------------------------------------------: | :------------------------------------------------------------------: | :---: |:---:| :---:| :---:|
|      [IF](https://arxiv.org/abs/1703.04730)       |             [Explicit](https://arxiv.org/abs/1703.04730)             | ✔️ | ✔️ | ❌ | [Code example](../../examples/brittleness/mnist_lr_brittleness.py) |
|                                                   | [CG](https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf) | ✔️ | ✔️ | ✔️ | [Code example](../../examples/brittleness/mnist_lr_brittleness.py) |
|                                                   |              [LiSSA](https://arxiv.org/abs/1602.03943)               | ✔️ | ✔️ | ❌ | [Code example](../../examples/brittleness/mnist_lr_brittleness.py) |
|                                                   |             [Arnoldi](https://arxiv.org/abs/2112.03052)              | ✔️ | ✔️ | ❌ | [Code example](../../examples/brittleness/mnist_lr_brittleness.py) |
|                                                   |             [DataInf](https://arxiv.org/abs/2310.00902)              | ✔️ | ✔️ | ❌ | [Code example](../../examples/brittleness/mnist_lr_brittleness.py) |
|                                                   |              [EK-FAC](https://arxiv.org/abs/2308.03296)              | ✔️ | ✔️ | ❌ | Coming soon |
|                                                   |             [RelatIF](https://arxiv.org/pdf/2003.11630)              | ✔️ | ✔️ | ❌ | Coming soon |
|                                                   |              [LoGra](https://arxiv.org/pdf/2405.13954)               | ✔️ | ✔️ | ❌ | Coming soon |
|                                                   |              [GraSS](https://arxiv.org/pdf/2505.18976)               | ✔️ | ✔️ | ❌ | Coming soon |
|    [TracIn](https://arxiv.org/abs/2002.08484)     |             [TracInCP](https://arxiv.org/abs/2002.08484)             | ✔️ | ✔️ | ✔️ | [Code example](../../examples/noisy_label_detection/tracin_noisy_label.py) |
|                                                   |             [Grad-Dot](https://arxiv.org/abs/2102.05262)             | ✔️ | ✔️ | ✔️ | [Code example](../../examples/noisy_label_detection/tracin_noisy_label.py) |
|                                                   |             [Grad-Cos](https://arxiv.org/abs/2102.05262)             | ✔️ | ✔️ | ✔️ | [Code example](../../examples/noisy_label_detection/tracin_noisy_label.py) |
|      [RPS](https://arxiv.org/abs/1811.09720)      |              [RPS-L2](https://arxiv.org/abs/1811.09720)              | ✔️ | ✔️ | ❌ | Coming soon |
|     [TRAK](https://arxiv.org/abs/2303.14186)      |               [TRAK](https://arxiv.org/abs/2303.14186)               | ✔️ | ✔️ | ✔️ | [Code example](../../examples/noisy_label_detection/trak_noisy_label.py) |
| [Shapley Value](https://arxiv.org/abs/1904.02868) |    [KNN-Shapley](https://dl.acm.org/doi/10.14778/3342263.3342637)    | ✔️ | ✔️ | ❌ | Coming soon |

### Below is more detailed description of the data structure each attributor can take.

Currently we support 2 types of data structures (`tuple` and `list`) to be passed in any Attributor, including Influence Function, TRAK and TracIn, etc.

Here is an example on how to define a dataset of type `tuple` to pass to Attributor:
```bash
# load the MNIST dataset
dataset, dataset_test = create_mnist_dataset("./data")

# dataloaders
...
train_loader_attr = torch.utils.data.DataLoader(
    dataset,
    batch_size=500,
    sampler=SubsetSampler(range(1000)),
)

...

# get influence function score
task = AttributionTask(...)
attributor = IFAttributorCG(task=task)
attributor.cache(train_loader_attr)
with torch.no_grad():
    score = attributor.attribute(train_loader_attr, ...)
```

We also support using `dict` for Influence Function (CG), TRAK and TracIn. However, it is worth noting that `Dataloader` would only work properly if the values in the dictionary are `torch.tensor`.

Here is an example on how to define a dataset of type `tuple` to pass to Attributor:
```bash
...

# get original dataset
train_dataset = lm_datasets["train"]
eval_dataset = lm_datasets["validation"]

# convert values in the dict to torch.tensor
train_dataset = [
    {k: torch.tensor(v, dtype=torch.long) for k, v in d.items()}
    for d in train_dataset
]
eval_dataset = [
    {k: torch.tensor(v, dtype=torch.long) for k, v in d.items()}
    for d in eval_dataset
]

# dataloaders
train_dataloader = DataLoader(
    train_dataset,
    ...
)
eval_dataloader = DataLoader(
    eval_dataset,
    ...
)

...
```

## Definition of loss/target function
The loss/target function to be passed in Attributor requires great care in the dimensionality of data passed in. The reason, in short, is that we sometimes placed a dummy leading dimension for different Attributors to work properly.

### Influence Function
In Influence Function, we have placed the dummy leading dimension (batch) for loss functions such as `CrossEntropyLoss` to work. An example loss function looks like this:
```bash
def f(params, data_target_pair):
    image, label = data_target_pair
    loss = nn.CrossEntropyLoss()
    yhat = torch.func.functional_call(model, params, image)
    return loss(yhat, label.long())
```
On the other hand, if there are any loss functions that don't need a dummy dimension, you may need to manually delete the dummy dimension we have defined in the loss function. An example looks like this:
```bash
def f(params, batch):
    input_ids = batch["input_ids"].squeeze(0).long().cuda()
    attention_mask = batch["attention_mask"].squeeze(0).long().cuda()
    labels = batch["labels"].squeeze(0).long().cuda()

    outputs = torch.func.functional_call(
        model,
        params,
        input_ids,
        kwargs={"attention_mask": attention_mask, "labels": labels},
    )
    logp = -outputs.loss
    return logp - torch.log(1 - torch.exp(logp))
```

### TRAK or TracIn
It is worth noting that TRAK and TracIn work the opposite way, where we don't place the dummy dimension. Thus, an example for `CrossEntropyLoss` in TRAK looks like this:
```bash
def f(params, data_target_pair):
    image, label = data_target_pair
    image_t = image.unsqueeze(0)
    label_t = label.unsqueeze(0)
    loss = nn.CrossEntropyLoss()
    yhat = torch.func.functional_call(model, params, image_t)
    logp = -loss(yhat, label_t)
    return logp - torch.log(1 - torch.exp(logp))
```
If there are any loss functions that don't need a dummy dimension, you can reference to this example:
```bash
def f(params, batch):
    input_ids, attention_mask, labels = batch

    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    labels = labels.cuda()

    outputs = torch.func.functional_call(
        model,
        params,
        input_ids,
        kwargs={"attention_mask": attention_mask, "labels": labels},
    )
    logp = -outputs.loss
    return logp - torch.log(1 - torch.exp(logp))
```