# `dattri`: A Library for Efficient Data Attribution

<!-- add some demos, list of methods and benchmark settings available, benchmark results, and some development plan -->

`dattri` is a PyTorch library for **developing, benchmarking, and deploying efficient data attribution algorithms**. You may use `dattri` to

- Deploy existing data attribution methods to PyTorch models
  - e.g., Influence Function, TracIn, RPS, TRAK, ...
- Develop new data attribution methods with efficient implementation of low-level utility functions
  - e.g., HVP/IHVP, random projection, dropout ensembling, ...
- Benchmark data attribution methods with standard benchmark settings
  - e.g., MNIST-10+LR/MLP, CIFAR-10/2+ResNet-9, MAESTRO + Music Transformer, Shakespeare + nanoGPT, ...

## Quick Start

### Installation

```python
pip install dattri
```

If you want to use all features on CUDA and accelerate the library, you may install the full version by

```python
pip install dattri[all]
```

> [!NOTE]
> It's highly recommended to use a device support CUDA to run `dattri`, especially for moderately large or larger models or datasets. And it's required to have CUDA if you want to install the full version `dattri`.

### Apply Data Attribution methods on PyTorch Models

One can apply different data attribution methods on PyTorch Models. One only needs to define:
1. a target function (e.g., `f`)
2. trained model parameters (e.g., `model_params`)
3. the data loaders for training samples and test samples (e.g., `train_loader`, `test_loader`).

The following is an example to use `Attributor` to apply data attribution to a PyTorch model.

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

### Use low-level utility functions to develop new data attribution methods

#### HVP/IHVP
Hessian-vector product (HVP), inverse-Hessian-vector product
(IHVP) are widely used in data attribution methods. `dattri` provides efficient implementation to these operators by `torch.func`. This example shows how to use the CG implementation of the IHVP implementation.

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

#### Random Projection
It has been shown that long vectors will retain most of their relative information when projected down to a smaller feature dimension. To reduce the computational cost, random projection is widely used in data attribution methods. Following is an example to use `random_project`. The implementation leaverges [`fast_jl`](https://pypi.org/project/fast-jl/).

```python
from dattri.func.random_projection import random_project

# initialize the projector based on users' needs
project_func = random_project(tensor, tensor.size(0), proj_dim=512)

# obtain projected tensors
projected_tensor = project_func(torch.full_like(tensor))
```

Normally speaking, `tensor` is probably the gradient of target function and has a large dimension (i.e., the number of parameters).

#### Dropout Ensemble
Recent studies found that ensemble methods can significantly improve the performance of data attribution, [DROPOUT ENSEMBLE](https://arxiv.org/pdf/2405.17293) is one of these ensemble methods. One may prepare their model with

```python
from dattri.model_utils.dropout import activate_dropout

# initialize a torch.nn.Module model
model = MLP()

# (option 1) activate all dropout layers
model = activate_dropout(model, dropout_prob=0.2)

# (option 2) activate specific dropout layers
# here "dropout1" and "dropout2" are the names of dropout layers within the model
model = activate_dropout(model, ["dropout1", "dropout2"], dropout_prob=0.2)
```

## Algorithms Supported
| Family |               Algorithms              |
|:------:|:-------------------------------------:|
|   [IF](https://arxiv.org/abs/1703.04730)   | [Explicit](https://arxiv.org/abs/1703.04730) |
|        |       [CG](https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf)      |
|        |    [LiSSA](https://arxiv.org/abs/1602.03943)    |
|        |  [Arnoldi](https://arxiv.org/abs/2112.03052)  |
| | [DataInf](https://arxiv.org/abs/2310.00902)|
| | [EK-FAC](https://arxiv.org/abs/2011.13609) |
| [TracIn](https://arxiv.org/abs/2002.08484) | [TracInCP](https://arxiv.org/abs/2002.08484) |
|        |   [Grad-Dot](https://arxiv.org/abs/2102.05262)  |
|        |   [Grad-Cos](https://arxiv.org/abs/2102.05262)  |
|   [RPS](https://arxiv.org/abs/1811.09720)  |   [RPS-L2](https://arxiv.org/abs/1811.09720)   |
|  [TRAK](https://arxiv.org/abs/2303.14186)  |       [TRAK](https://arxiv.org/abs/2303.14186)       |

## Metrics Supported
- Leave-one-out (LOO) correlation
- Linear datamodeling score (LDS)
- Area under the ROC curve (AUC) for noisy label detection

## Benchmark Settings Supported
|   Dataset   |       Model       |         Task         | Sample size (train,test) | Parameter size |   Metrics   |          Data Source         |
|:-----------:|:-----------------:|:--------------------:|:------------------------:|:--------------:|:-----------:|:----------------------------:|
|   MNIST-10  |         LR        | Image Classification |        (5000,500)        |      7840      | LOO/LDS/AUC |      [link](http://yann.lecun.com/exdb/mnist/)     |
|   MNIST-10  |        MLP        | Image Classification |        (5000,500)        |      0.11M     | LOO/LDS/AUC |      [link](http://yann.lecun.com/exdb/mnist/)     |
|   CIFAR-2   |      ResNet-9     | Image Classification |        (5000,500)        |      4.83M     |     LDS     | [link](https://www.cs.toronto.edu/~kriz/cifar.html) |
|   CIFAR-10  |      ResNet-9     | Image Classification |        (5000,500)        |      4.83M     |     AUC     | [link](https://www.cs.toronto.edu/~kriz/cifar.html) |
|   MAESTRO   | Music Transformer |   Music Generation   |        (5000,178)        |      13.3M     |     LDS     |  [link](https://magenta.tensorflow.org/datasets/maestro) |
| Shakespeare |      nanoGPT      |    Text Generation   |        (3921,435)        |      10.7M     |     LDS     |     [link](https://github.com/karpathy/nanoGPT)    |

## Benchmark Results
### MNIST+LR/MLP
![mnist-result](assets/images/benchmark-result-mnist.png)

### LDS performance on larger models
![larger-lds-result](assets/images/benchmark-result-larger-lds.png)

### AUC performance
![larger-lds-result](assets/images/benchmark-result-auc.png)

## Development Plan
- More (larger) benchmark settings to come
  - ImageNet + ResNet-18
  - Tinystories + nanoGPT
  - Comparison with other libraries
- More algorithms and low-level utility functions to come
  - KNN filter
  - TF-IDF filter
  - RelativeIF
- Better documentation
  - Quick start colab notebooks
