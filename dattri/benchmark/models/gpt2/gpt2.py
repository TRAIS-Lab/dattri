"""This file imports the gpt2 model."""

from torch import nn


def create_gpt2_model() -> nn.Module:
    from transformers import AutoModelForCausalLM, AutoConfig

    config = AutoConfig.from_pretrained(
        "openai-community/gpt2",
    )
    model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2",
        config=config,
        from_tf=False,
    )

    return model
