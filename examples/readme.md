# *dattri* examples
This folder contains bite-sized examples that can help users build their own applications with *dattri*.

## Noisy label detection
This section contains examples using different attributors to detect noisy labels in various datasets.

[Use influence function to detect noisy labels in Mnist10 + Logistic regression.](./noisy_label_detection/influence_function_noisy_label.py)

[Use TracIN to detect noisy labels in Mnist10 + MLP.](./noisy_label_detection/tracin_noisy_label.py)

[Use TRAK to detect noisy labels in CIFAR10 + ResNet-9.](./noisy_label_detection/trak_noisy_label.py)

## Use pretrained checkpoints and pre-calculated ground truth

This section contains examples using the pretrained checkpoints and pre-calculated ground truth provided by *dattri* to evaluate the data attribution methods.

[Use pre-trained Mnist10 + MLP benchmark setting and evaluate Influence Function (CG) algorithm by LDS](./pretrained_benchmark/influence_function_lds.py)

[Use pre-trained Mnist10 + LR benchmark setting and evaluate TRAK algorithm by LOO correlation](./pretrained_benchmark/trak_lds.py)

[Use pre-trained MNIST10 + MLP benchmark setting and evaluate TRAK + dropout ensemble by LDS](./pretrained_benchmark/trak_dropout_lds.py) 

[Use pre-trained MNIST10 + MLP benchmark setting and evaluate LoGra by LDS](./pretrained_benchmark/logra_lds.py)

## Estimate the brittleness

This section contains examples using attribution scores to estimate the brittleness of a model.

[Use influence function to estimate the brittleness of losigitc regression trained on Mnist10](./brittleness/mnist_lr_brittleness.py)

## Data cleaning

This section contains examples using attribution scores to find the data points that can be removed from the training set and improve the test performance.

[Use influence function to find the low-quality data points in MNIST-10 and evaluate the performance](./data_cleaning/influence_function_data_cleaning.py)
