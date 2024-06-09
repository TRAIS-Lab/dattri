# `dattri`: A Library for Efficient Data Attribution

`dattri` is a PyTorch library for deploying, developing and benchmarking **efficient data attribution algorithms**. You may use `dattri` to

- Deploy existing data attribution methods (e.g., Influence Function, TracIN, RPS, TRAK, ...) to PyTorch models.
- Develop new data attribution methods with efficient implementation of low-level utility functions (e.g., HVP/IHVP, random projection, dropout ensembling, ...).
- Benchmark data attribution methods with standard benchmark settings (e.g., MNIST-10+LR/MLP, CIFAR-10/2+ResNet-9, MusicTransformer+MAESTRO, NanoGPT+Shakespeare/tinystories, ImageNet+ResNet-18, ...).

## Quick Start

### Installation

```python
pip install dattri
```

If you want to use all features on CUDA and accelerate the library, you may install the full version by

```python
pip install dattri[all]
```

### Apply Data Attribution methods on Existing PyTorch Models

One can apply different data attribution methods on existing PyTorch Models. One only need to define a target function (e.g., `f`), a trained model parameter (e.g., `model_params`) and the data loader for training samples and test samples (e.g., `train_loader`, `test_loader`).

```python
from dattri.algorithm import IFAttributor, TracInAttributor, TRAKAttributor, RPSAttributor

@flatten_func(model) # a decorator that helps simplify user API
def f(params, data): # an example of target function using CE loss
    x, y = data
    loss = nn.CrossEntropyLoss()
    yhat = torch.func.functional_call(model, params, x)
    return loss(yhat, y)

model_params = {k: p for k, p in model.named_parameters() if p.requires_grad}

attributor = IFAttributor(
    target_func=f,
    params=model_params,
    **attributor_hyperparams # e.g., ihvp solver, ...
) # same for other attributors: TracInAttributor, TRAKAttributor, RPSAttributor

attributor.cache(train_loader) # optional pre-processing to accelerate the attribution
score = attributor.attribute(train_loader, test_loader)
```

### Use low-level utility functions to build data attribution methods

#### IHVP
This example shows how to use the CG implementation of the IHVP implementation.

```python
from dattri.func.ihvp import ihvp_cg, ihvp_at_x_cg

def f(x, param): # target function
    return torch.sin(x / param).sum()

x = torch.randn(2)
param = torch.randn(1)
v = torch.randn(5, 2)

# ihvp_cg method
ihvp_func = ihvp_cg(f, argnums=0, max_iter=2) # argnums=0 indicates that the param of (x, param) to be passed to ihvp_func is the model parameter
ihvp_result_1 = ihvp_func((x, param), v) # both (x, param) and v as the inputs
# ihvp_at_x_cg method: (x, param) is cached
ihvp_at_x_func = ihvp_at_x_cg(f, x, param, argnums=0, max_iter=2)
ihvp_result_2 = ihvp_at_x_func(v) # only v as the input
# the above two will give the same result
assert torch.allclose(ihvp_result_1, ihvp_result_2)
```

## Algorithms Supported
| Family |               Algorithms              |
|:------:|:-------------------------------------:|
|   [IF](https://arxiv.org/abs/1703.04730)   | [Explicit](https://arxiv.org/abs/1703.04730) |
|        |       CG~\citep{martens2010deep}      |
|        |    LiSSA~\citep{agarwal2017second}    |
|        |  Arnoldi~\citep{schioppa2022scaling}  |
| TracIn | TracInCP~\citep{pruthi2020estimating} |
|        |   Grad-Dot~\citep{charpiat2019input}  |
|        |   Grad-Cos~\citep{charpiat2019input}  |
|   RPS  |   RPS-L2~\citep{yeh2018representer}   |
|  TRAK  |       TRAK~\citep{park2023trak}       |

## Benchmark settings
|   Dataset   |       Model       |         Task         | Sample size (train,test) | Parameter size |   Metrics   |          Data Source         |
|:-----------:|:-----------------:|:--------------------:|:------------------------:|:--------------:|:-----------:|:----------------------------:|
|   MNIST-10  |         LR        | Image Classification |        (5000,500)        |      7840      | LOO/LDS/AUC |      \citepdeng2012mnist     |
|   MNIST-10  |        MLP        | Image Classification |        (5000,500)        |      0.11M     | LOO/LDS/AUC |      \citepdeng2012mnist     |
|   CIFAR-2   |      ResNet-9     | Image Classification |        (5000,500)        |      4.83M     |     LDS     | \citepkrizhevsky2009learning |
|   CIFAR-10  |      ResNet-9     | Image Classification |        (5000,500)        |      4.83M     |     AUC     | \citepkrizhevsky2009learning |
|   MAESTRO   | Music Transformer |   Music Generation   |        (5000,178)        |      13.3M     |     LDS     |  \citephawthorne2018enabling |
| Shakespeare |      NanoGPT      |    Text Generation   |        (3921,435)        |      10.7M     |     LDS     |     \citepShakespeare2015    |

<!-- ## Highlighted Features
- **Minimal code invasion**: allowing the data attribution methods to be integrated into most common PyTorch-based machine learning pipelines with only a few lines of code changed.
- **Low-level utility functions**: such as Hessian-vector product, inverse-Hessian-vector product or random projection.
- **Comprehensive list of evaluation metrics**: We aim to provide a benchmark suite with diverse benchmark settings, including generative AI settings. In addition to the code for the evaluation metrics and benchmark experiments, we also provide the trained model checkpoints for each benchmark setting. -->
