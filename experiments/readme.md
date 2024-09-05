# Experiment script

This folder is used to reproduce the benchmark results. If you want to find some code snippets, You may find some easier and clearer examples in `/example` folder.

## Usage
The `benchmark_result.py` is designed to be an integrated script to run the experiments in one line. The usage of this script is as following
```
usage: benchmark_result.py [-h] [--dataset {mnist,cifar2}] [--model {lr,mlp,resnet9}] [--method {if-explicit,if-cg,if-lissa,if-arnoldi,TRAK-1,TRAK-10,TRAK-50,TracIn,Grad-Dot,Grad-Cos,RPS}]
                           [--metric {lds,loo}] [--device DEVICE]

options:
  -h, --help            show this help message and exit
  --dataset {mnist,cifar2}
                        The dataset of the benchmark.
  --model {lr,mlp,resnet9}
                        The model of the benchmark.
  --method {if-explicit,if-cg,if-lissa,if-arnoldi,TRAK-1,TRAK-10,TRAK-50,TracIn,Grad-Dot,Grad-Cos,RPS}
                        The TDA method to benchmark.
  --metric {lds,loo}    The metric to evaluate the TDA method.
  --device DEVICE       The device to run the experiment.
```

For example, if you want to check the influence function with CG solver's performance on mnist+lr benchmark evaluated by LDS, you may run
```bash
python benchmark_result.py --model lr --dataset mnist --method if-cg --metric lds
```
The script will search for a hyperparameter space of the TDA method (usually a small space), and report the best result as following

```
if-cg RESULT: {'regularization': 0.1, 'max_iter': 10} lds: tensor(0.7670)
```
