# Experiment script for GPT-2 + WikiText-2

This folder is used to reproduce the benchmark results for GPT-2 + WikiText-2.

The code is adapted from
https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py

Four scripts are included in this example
- `train.py`, the code is unchanged. Specific parameters are shown in following section.
- `score.py`, code after "# ... dattri Code begins here ..." are `dattri` specific code. The original code run TRAK-5 (5 independent ensemble on TRAK) and save the score file in `score.pt`
- `groundtruth.py`, code after "# ... dattri Code begins here ..." are `dattri` specific code. The original code calculate the LDS groundtruth for 50 checkpoints saved by `train.py`. The groundtruth is saved in `gt.pt`.
- `spearman.py`, calculate the lds score.

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

```bash
python score.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path openai-community/gpt2 \
    --output_dir ./checkpoints \
    --block_size 512 \
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


# calculate the LDS

```bash
python spearman.py
```

```bash
> score shape: torch.Size([4656, 481])
> ...
> 0.1613172442573241
```
