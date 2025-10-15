# Experiment script for GPT-2 + WikiText-2

This folder is used to reproduce the benchmark results for GPT-2 + WikiText-2.

The code is adapted from
https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py

Four scripts are included in this example
- `train.py`, the code is unchanged. Specific parameters are shown in following section.
- `score_logra.py`, code after "# ... dattri Code begins here ..." are `dattri` specific code, which runs LoGra and save the score file in `score_logra.pt`
- `score_TRAK.py`, code after "# ... dattri Code begins here ..." are `dattri` specific code, which runs TRAK-5 (5 independent ensemble on TRAK) and save the score file in `score_TRAK.pt`
- `groundtruth.py`, code after "# ... dattri Code begins here ..." are `dattri` specific code. The original code calculate the LDS groundtruth for 50 checkpoints saved by `train.py`. The groundtruth is saved in `gt.pt`.
- `spearman.py`, calculate the lds score.

This experiment could only be run on cuda device.

## Enviroment

```bash
pip install -r requirements.txt
```

### Troubleshooting: vmap over calling .item() Error in Transformers
After installing transformers, you might encounter the following error:
```bash
We don't support vmap over calling .item() on a Tensor, please try to rewrite what you're doing with other operations.
```
This issue arises due to certain optimizations in PyTorch and the transformers library. To fix it, you need to modify specific files in `site-packages/transformers/`.

1. Locate the file modeling_attn_mask_utils.py
Find the file in your environment:
```bash
<your_python_env>/lib/pythonX.X/site-packages/transformers/modeling_attn_mask_utils.py
```
Inside the function:
```bash
def _ignore_causal_mask_sdpa(
    attention_mask: Optional[torch.Tensor],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
    is_training: bool = False,
) -> bool:
```
Comment out the following lines:
```bash
# elif not is_tracing and torch.all(attention_mask == 1):
#     if query_length == 1 or key_value_length == query_length:
#         # For query_length == 1, causal attention and bi-directional attention are the same.
#         # ignore_causal_mask = True
```

2. Modify _prepare_4d_attention_mask_for_sdpa
In the same file, locate the function:
```bash
def _prepare_4d_attention_mask_for_sdpa(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
```
Comment out these lines:
```bash
# torch.jit.trace, symbolic_trace and torchdynamo with fullgraph=True are unable to capture data-dependent controlflows.
# if not is_tracing and torch.all(mask == 1):
#     return None
# else:
#     return AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)
```
Then, add the following line:
```bash
return AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)
```

## Training

First train (fine-tune) multiple models with 50% dataset.

```bash
python train.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path openai-community/gpt2 \
    --output_dir ./checkpoints/${SLURM_ARRAY_TASK_ID} \
    --block_size 512 \
    --subset_ratio 0.5\
    --seed ${SLURM_ARRAY_TASK_ID}  # 50 models
```

## Calculate the attribution score

Currently we have LoGra and TRAK as two examples of data attribution methods.

```bash
python score_logra.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path openai-community/gpt2 \
    --output_dir ./checkpoints \
    --block_size 512 \
    --seed 0
```

```shell
python score_TRAK.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path openai-community/gpt2 \
    --output_dir ./checkpoints \
    --block_size 512 \
    --method TRAK-5 \
    --seed 0
```

## Calculate the ground truth for LDS evaluation

```bash
python groundtruth.py\
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path openai-community/gpt2 \
    --output_dir ./checkpoints \
    --block_size 512 \
    --seed 0
```


# Calculate the LDS

```bash
python spearman.py \
     --score_path "score_logra.pt" 
```

```bash
python spearman.py \
    --score_path "score_TRAK.pt"
```

```bash
> score shape: torch.Size([4656, 481])
> ...
> 0.1613172442573241
```

## Troubleshooting: NumPy Version Compatibility Issue

If you encounter the following error:

```bash
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.3
```

this means that some dependencies or compiled extensions were built with NumPy **1.x** and are incompatible with **NumPy 2.x**.

### Solution: Downgrade NumPy

To resolve this issue, downgrade NumPy to a compatible version **(≥1.25 but still in the 1.x range)**:

```bash
pip install "numpy>=1.25,<2.0"
```

This ensures that you have at least NumPy 1.25 but avoid upgrading to NumPy 2.x, preventing compatibility issues.
