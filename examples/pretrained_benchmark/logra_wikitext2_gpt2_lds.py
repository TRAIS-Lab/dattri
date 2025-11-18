import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, default_data_collator
from transformers.pytorch_utils import Conv1D

from dattri.algorithm.logra import LoGraAttributor
from dattri.benchmark.load import load_benchmark
from dattri.metric import lds
from dattri.task import AttributionTask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()

    model_details, groundtruth = load_benchmark(
        model="gpt2", dataset="wikitext2", metric="lds",
    )

    def replace_conv1d_modules(model):
        """Replace all Conv1D modules in a model with Linear modules.

        Args:
            model: The model to replace Conv1D modules in.

        Returns:
            model: The model with all Conv1D modules replaced with Linear modules.
        """
        # GPT-2 is defined in terms of Conv1D.
        # Here, we convert these Conv1D modules to linear modules recursively.
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                replace_conv1d_modules(module)

            if isinstance(module, Conv1D):
                new_module = nn.Linear(
                    in_features=module.weight.shape[0],
                    out_features=module.weight.shape[1],
                )
                new_module.weight.data.copy_(module.weight.data.t())
                new_module.bias.data.copy_(module.bias.data)
                setattr(model, name, new_module)
        return model

    def checkpoints_load_func(model, checkpoint):
        # start fresh model
        model = AutoModelForCausalLM.from_pretrained(
            "openai-community/gpt2",
            config=AutoConfig.from_pretrained("openai-community/gpt2"),
            from_tf=False,
        )

        # load from stored pretrained.
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model = replace_conv1d_modules(model)

        model.to(args.device)
        model.eval()
        return model

    def f(model, batch, device):
        model.to(device)
        inputs = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        outputs = model(**inputs)
        return outputs.loss

    model = model_details["model"]
    model = replace_conv1d_modules(model)  # replace conv1d with linear 
    train_dataset = model_details["train_dataset"]
    eval_dataset = model_details["test_dataset"]

    train_sampler = model_details["train_sampler"]
    test_sampler = model_details["test_sampler"]

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        batch_size=4,
        sampler=train_sampler,
    )

    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=4, sampler = test_sampler, shuffle=False,
    )

    projector_kwargs = {
        "device": args.device,
        "proj_dim": 32,  # projection dimension = 32 * 32 = 1024
        "proj_max_batch_size": 32,
    }

    task = AttributionTask(
        model=model.to(args.device),
        loss_func=f,
        checkpoints=model_details["models_full"][0],
        checkpoints_load_func=checkpoints_load_func,
    )

    attributor = LoGraAttributor(
        task=task,
        device=args.device,
        damping=1e-2,
        offload="cpu",
        projector_kwargs=projector_kwargs,
    )

    attributor.cache(train_dataloader)
    score = attributor.attribute(train_dataloader, eval_dataloader)

    score = score.cpu()
    groundtruth = (groundtruth[0].cpu(), groundtruth[1].cpu())

    lds_score = lds(score, groundtruth)[0]
    print("lds:", torch.mean(lds_score[~torch.isnan(lds_score)]))
