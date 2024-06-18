from pathlib import Path
import torch

from dattri.benchmark.datasets.cifar2 import (
    create_cifar2_dataset,
    train_cifar2_resnet9,
    create_resnet9_model,
    loss_cifar2_resnet9
)


previous_model_root = "/home/junweid2/toolkit/cifar-2-checkpoints/checkpoint_without_dropout"
new_model_root = "/home/shared/dattri-dataset/cifar2_lds_test/models"
previous_script_new_model_root = "/home/twli/Dropout-Training-Data-Attribution/cifar2-experiment/cifar_checkpoint_test"



# read in prev model param
model = create_resnet9_model()
model.load_state_dict(torch.load(Path(previous_model_root) / "checkpoint_0.pt"))
model.eval()

print()
print("junwei ckpt info: ")
for name, param in model.named_parameters():
    if param.requires_grad:
        if "linear" in name:
            print (name, param.data)

with open(Path(previous_model_root) / "selected_indices_seed_0.txt") as f:
    l = [int(line) for line in f]
print(l[:10])



# read in new model param
model = create_resnet9_model()
model.load_state_dict(torch.load(Path(new_model_root) / "0/model_weights_0.pt"))
model.eval()

print()
print("dattri retrain ckpt info: ")
for name, param in model.named_parameters():
    if param.requires_grad:
        if "linear" in name:
            print (name, param.data)


with open(Path(new_model_root) / "0/indices.txt") as f:
    l = [int(line) for line in f]
print(l[:10])


# train_dataset, test_dataset = create_cifar2_dataset("./data")
# counter = 0
# for data in train_dataset:
#     # print(data)
#     if counter == 4999:
#         print(data)
#     counter += 1


print()
print("use old script to produce new model: ")
model = create_resnet9_model()
model.load_state_dict(torch.load(Path(previous_script_new_model_root) / "checkpoint_0.pt"))
model.eval()

print("dattri retrain ckpt info: ")
for name, param in model.named_parameters():
    if param.requires_grad:
        if "linear" in name:
            print (name, param.data)

with open(Path(previous_script_new_model_root) / "selected_indices_seed_0.txt") as f:
    l = [int(line) for line in f]
print(l[:10])