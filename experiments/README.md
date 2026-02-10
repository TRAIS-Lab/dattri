# Experiment script

This folder is used to reproduce the benchmark results. If you want to find some code snippets, You may find some easier and clearer examples in `/example` folder.

## CIFAR-2 / MNIST
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

## ShakeSpeare + nanoGPT
Shakespeare dataset needs to be download. Once you install *dattri*, you may download it in one line.
```bash
dattri_retrain_nanogpt --dataset shakespeare_char --only_download_dataset
```
Then you may run the script `benchmark_result_nanogpt` designed for nanogpt.
```
usage: benchmark_result_nanogpt.py [-h] [--method {TRAK-1,TRAK-10,TRAK-50,TracIn,Grad-Dot,Grad-Cos}] [--device DEVICE]

options:
  -h, --help            show this help message and exit
  --method {TRAK-1,TRAK-10,TRAK-50,TracIn,Grad-Dot,Grad-Cos}
                        The TDA method to benchmark.
  --device DEVICE
```

For example,
```bash
python benchmark_result_nanogpt.py --method TRAK-10 --device cuda
```
The result will be like
```bash
TRAK-10 RESULT: {'proj_dim': 2048, 'device': 'cuda'} lds: tensor(0.1419)
```

## MAESTRO + MusicTransformer
The `benchmark_result_mt.py` is an intergrated script to run MusicTransformer experiments in one line. The usage of this script is as following
```
usage: benchmark_result_mt.py [-h] [--method {TRAK-1,TRAK-10,TRAK-50,TracIn,Grad-Dot,Grad-Cos}] [--device DEVICE]

options:
  -h, --help            show this help message and exit
  --method {TRAK-1,TRAK-10,TRAK-50,TracIn,Grad-Dot,Grad-Cos}
                        The TDA method to benchmark.
  --device DEVICE       The device to run the experiment.
```

For example,
```bash
python benchmark_result_mt.py --method TRAK-10 --device cuda
```
The result will be like
```bash
TRAK-10 RESULT: {'proj_dim': 2048, 'device': 'cuda'} lds: tensor(0.3243)
```
