"""This file imports the gpt2 model."""

from torch import nn
from transformers import AutoModelForCausalLM


def create_gpt2_model() -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2",
        from_tf=False,
    )

    return model
