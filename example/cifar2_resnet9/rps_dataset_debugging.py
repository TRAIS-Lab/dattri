"""This example shows how to use the IF to detect noisy labels in the MNIST."""

import time
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import Sampler
from torchvision import transforms

from dattri.algorithm.rps import RPSAttributor
from dattri.benchmark.datasets.cifar2 import create_cifar2_dataset, train_cifar2_resnet9
from dattri.benchmark.utils import flip_label


def get_cifar_indices_and_adjust_labels(dataset, subset_indice, p=0.1):

    dataset.targets, flip_index = flip_label(torch.tensor(dataset.targets)[subset_indice], p=p)
    return flip_index


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    train_dataset, _ = create_cifar2_dataset("./data")

    subset_size = 5000
    full_train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=subset_size,
        sampler=SubsetSampler(range(subset_size)),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        sampler=SubsetSampler(range(subset_size)),
    )


    flip_index = get_cifar_indices_and_adjust_labels(train_dataset, range(subset_size))
    print("flipped index size: ", len(flip_index))
    # simulate checkpoints
    model = train_cifar2_resnet9(train_loader, device="cuda", num_epochs=10)

    # only need one model definition
    model.cuda()
    model.eval()

    # define the loss function
    def f(pre_activation_list, label_list):
        return binary_cross_entropy_with_logits(pre_activation_list, label_list)

    model_params = {k: p for k, p in model.named_parameters() if p.requires_grad}
    attributor = RPSAttributor(
        target_func=f,
        model=model,
        final_linear_layer_name="linear",
        device=torch.device("cuda"),
    )

    attributor.cache(full_train_loader)
    torch.cuda.reset_peak_memory_stats("cuda")
    start_attribute = time.time()
    score = attributor.attribute(train_loader, train_loader).squeeze(1).abs()
    # print(score)
    peak_memory = torch.cuda.max_memory_allocated("cuda") / 1e6  # Convert to MB
    print(f"Peak memory usage: {peak_memory} MB")
    end_attribute = time.time()
    print("Attribution time: ", end_attribute-start_attribute)
    print("score shape: ", score.shape)
    _, indices = torch.sort(-score)

    cr = 0
    cr_list = []
    for idx, index in enumerate(indices):
        if (idx+1) % 100 == 0:
            cr_list.append((idx+1, cr))
        if int(index) in set(flip_index):
            cr += 1
    print(cr_list)
    print(f"{'Checked Data Sample':<25}{'Found flipped Sample':25}")
    print("-" * 50)

    for row in cr_list:
        print(f"{row[0]:<25}{row[1]:<25}")
