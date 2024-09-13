# `dattri` examples
This folder contains bite-sized examples which can help users to build their own application by `dattri`.

## Noisy label detection
This section contains using different attributors to detect the noisy label in various datasets.

[Use influence function to detect noisy label in Mnist10 + Logistic regression.](./noisy_label_detection/influence_function_noisy_label.py)

[Use TracIN to detect noisy label in Mnist10 + MLP.](./noisy_label_detection/tracin_noisy_label.py)

[Use TRAK to detect noisy label in CIFAR10 + ResNet-9.](./noisy_label_detection/trak_noisy_label.py)

## Use pretrained checkpoints and pre-calculated ground truth

This section contains examples to use the pretrained checkpoints and pre-calculated ground truth provided by `dattri` to evaluate the data attribution methods.

[Use pre-trained Mnist10 + MLP benchmark setting and evaluate Influence Function (CG) algorithm by LDS](./pretrained_benchmark/influence_function_lds.py)

[Use pre-trained Mnist10 + LR benchmark setting and evaluate TRAK algorithm by LOO correlation](./pretrained_benchmark/trak_lds.py)

## Estimate the brittleness

This section contains examples to use attribution score to estimate the brittleness of a model.

[Use influence function to estimate the brittleness of losigitc regression trained on Mnist10](./brittleness/mnist_lr_brittleness.py)
