# This is a simple example to demonstrate how to use TracInAttributor with a TestDataloaderGroup.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dattri.algorithm.tracin import DataloaderGroup, TracInAttributor

def main():
    # trivial linear model and synthetic data for demonstration
    torch.manual_seed(42)
    input_dim, n_train, n_test = 2, 10, 5

    model = nn.Linear(input_dim, 1, bias=False)
    model.weight.data.fill_(1.0)

    train_loader = DataLoader(TensorDataset(torch.randn(n_train, input_dim), torch.randn(n_train, 1)), batch_size=2)
    test_loader = DataLoader(TensorDataset(torch.randn(n_test, input_dim), torch.randn(n_test, 1)), batch_size=2)


    def func(params, data):
        x, y = data
        w = params['weight']
        return ((x @ w.t()) - y) * x


    def func_group(params, loader):
        x, y = loader
        w = params['weight']
        return torch.sum(((x @ w.t()) - y) * x, dim=0, keepdim=True)

    class SimpleTask:
        def get_checkpoints(self): return [0]
        def get_param(self, *args, **kwargs): return dict(model.named_parameters()), None
        def get_grad_loss_func(self, *args, **kwargs): return func
        def get_grad_target_func(self, *args, **kwargs):
            return func_group

    attributor = TracInAttributor(
        task=SimpleTask(),
        weight_list=torch.tensor([1.0]),
        normalized_grad=False
    )
    attributor.projector_kwargs = None

    test_group = TestDataloaderGroup(test_loader)
    scores = attributor.attribute(train_loader, test_group)

    # The TracInAttributor should compute the influence scores for each training example with respect to the test dataloader group.
    print(f"Test Dataloader Group.")
    print(f"Score Shape: {scores.shape}")
    print(f"Calculated Scores:\n{scores.flatten()}")

if __name__ == "__main__":
    main()
