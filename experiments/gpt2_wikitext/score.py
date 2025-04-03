#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import csv
from pathlib import PosixPath

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from dattri.benchmark.utils import SubsetSampler
from dattri.func.utils import flatten_func, flatten_params
from dattri.task import AttributionTask
from dattri.algorithm.trak import TRAKAttributor
from dattri.algorithm.tracin import TracInAttributor

check_min_version("4.46.0")

logger = get_logger(__name__)
require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune + run TRAK/TracIn on a causal language modeling task")

    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--validation_split_percentage", default=5)
    parser.add_argument("--model_name_or_path", type=str, required=False)
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
                        choices=["linear", "cosine", "cosine_with_restarts",
                                 "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model_type", type=str, default=None, choices=MODEL_TYPES)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--preprocessing_num_workers", type=int, default=None)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--no_keep_linebreaks", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str)
    parser.add_argument("--hub_token", type=str)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--checkpointing_steps", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--report_to", type=str, default="all")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--subset_ratio", type=float, default=1.0)

    parser.add_argument(
        "--method",
        type=str,
        default="TRAK-5",
        help=(
            "Which attribution method to run. "
            "Examples: 'TRAK-1', 'TRAK-5', 'TracIn', 'Grad-Dot', 'Grad-Cos'. "
            "Use 'TRAK-k' to load k checkpoints and run TRAK."
        ),
    )

    args = parser.parse_args()

    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`train_file` should be a csv, json or txt file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or txt file.")

    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    return args


def main():
    args = parse_args()

    send_example_telemetry("run_clm_no_trainer", args)

    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.push_to_hub:
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.dataset_name is not None:
        raw_datasets = load_dataset(
            args.dataset_name, args.dataset_config_name, trust_remote_code=args.trust_remote_code
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                trust_remote_code=args.trust_remote_code,
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                trust_remote_code=args.trust_remote_code,
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )

    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. "
            "You can do it from another script, save it, and load it here via --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )
        model = model.cuda()
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)
        model = model.cuda()

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, config.max_position_embeddings)} instead."
            )
            block_size = min(1024, config.max_position_embeddings)
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the model's max_length "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    train_sampler = SubsetSampler(range(len(train_dataset)))

    logger.info(f"The training dataset length: {len(train_dataset)}.")
    logger.info(f"The eval dataset length: {len(eval_dataset)}.")

    def custom_collate_fn(batch):
        batch = default_data_collator(
            [
                {k: v for k, v in item.items() if k in ["input_ids", "attention_mask", "labels"]}
                for item in batch
            ]
        )
        return batch

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=custom_collate_fn,
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=custom_collate_fn,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
    )

    def f(params, batch):
        """
        Log-odds objective for TRAK:
          logp = -loss => p = exp(-loss).
          We compute: logp - log(1 - exp(logp)) = log( p / (1 - p) ).
        """
        outputs = torch.func.functional_call(
            model, params, batch["input_ids"].cuda(),
            kwargs={"attention_mask": batch["attention_mask"].cuda(),
                    "labels": batch["labels"].cuda()}
        )
        logp = -outputs.loss
        return logp - torch.log(1 - torch.exp(logp))

    def m(params, batch):
        """
        Probability of correctness for TRAK:
          p = exp(-loss).
        """
        outputs = torch.func.functional_call(
            model, params, batch["input_ids"].cuda(),
            kwargs={"attention_mask": batch["attention_mask"].cuda(),
                    "labels": batch["labels"].cuda()}
        )
        p = torch.exp(-outputs.loss)
        return p

    def loss_tracin(params, batch):
        """
        Plain cross-entropy loss for TracIn / Grad-based similarity
        (TracIn sums over checkpoint updates of gradient dot-products).
        """
        input_ids, attention_mask, labels = batch
        outputs = torch.func.functional_call(
            model, params, input_ids.cuda(),
            kwargs={"attention_mask": attention_mask.cuda(),
                    "labels": labels.cuda()}
        )
        return outputs.loss

    method = args.method
    if method.startswith("TRAK-"):
        parts = method.split("-")
        if len(parts) == 2 and parts[1].isdigit():
            num_checkpoints = int(parts[1])
        else:
            raise ValueError("Invalid method name for TRAK, must be like 'TRAK-5' or 'TRAK-10'.")
        checkpoints = [f"{args.output_dir}/{i}" for i in range(num_checkpoints)]
    elif method in ["TracIn", "Grad-Dot", "Grad-Cos"]:
        num_checkpoints = 5
        checkpoints = [f"{args.output_dir}/{i}" for i in range(num_checkpoints)]
    else:
        raise ValueError(
            f"Unknown --method {method}. Try 'TRAK-5', 'TracIn', 'Grad-Dot', or 'Grad-Cos'."
        )

    def checkpoints_load_func(model, checkpoint_path):
        new_model = AutoModelForCausalLM.from_pretrained(checkpoint_path).cuda()
        new_model.eval()
        return new_model

    if method.startswith("TRAK"):
        task = AttributionTask(
            loss_func=f,
            model=model,
            checkpoints=checkpoints,
            target_func=None,
            checkpoints_load_func=checkpoints_load_func,
        )
    else:
        task = AttributionTask(
            loss_func=loss_tracin,
            model=model,
            checkpoints=checkpoints,
            target_func=None,
            checkpoints_load_func=checkpoints_load_func,
        )

    if method.startswith("TRAK"):
        projector_kwargs = {
            "device": "cuda",
            "proj_dim": 2048,
            "use_half_precision": False,
        }
        attributor = TRAKAttributor(
            task=task,
            correct_probability_func=m,
            device="cuda",
            projector_kwargs=projector_kwargs,
        )

    else:
        normalized_grad = False
        if method == "Grad-Cos":
            normalized_grad = True

        weight_list = torch.ones(num_checkpoints) * 1e-3

        projector_kwargs = {
            "device": "cuda",
            "proj_dim": 2048,
            "use_half_precision": False,
        }

        attributor = TracInAttributor(
            task=task,
            weight_list=weight_list,
            normalized_grad=normalized_grad,
            device="cuda",
            projector_kwargs=projector_kwargs,
        )

    with torch.no_grad():
        if isinstance(attributor, TRAKAttributor):
            attributor.cache(train_dataloader)
            score = attributor.attribute(eval_dataloader)
        else:
            score = attributor.attribute(train_dataloader, eval_dataloader)

    torch.save(score, "score.pt")
    logger.info("Attribution scores saved to score.pt")


if __name__ == "__main__":
    main()
