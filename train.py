#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import math
import os
import sys

import datasets
import numpy as np
import torch
import transformers
from aac_metrics import evaluate
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    BartConfig,
    get_inverse_sqrt_schedule,
    get_scheduler,
)

from data.collator import DataCollatorForEnClapBart, EvalDataCollatorForEnClapBart
from data.preprocess import Preprocessor
from modeling.enclap_bart import EnClapBartForConditionalGeneration, EnClapBartConfig

logger = get_logger(__name__)
metric_list = ["meteor", "spider"]


def main():
    # Load Configuration
    cfg_path = sys.argv[1]
    args = OmegaConf.load(cfg_path)

    # Initialize Logging
    accelerator_log_kwargs = {}
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        split_batches=args.split_batches,
        kwargs_handlers=[ddp_kwargs],
        **accelerator_log_kwargs,
    )
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args.yaml"), "w") as f:
            OmegaConf.save(args, f)
    accelerator.wait_for_everyone()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "train_log.txt"))
    logger.logger.addHandler(file_handler)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets
    data_files = {}
    data_files_eval = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files_eval["validation"] = args.validation_file

    extension = args.train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    raw_datasets_eval = load_dataset(extension, data_files=data_files_eval)

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if args.config_name_or_path is not None:
        config = EnClapBartConfig.from_pretrained(args.config_name_or_path)
    else:
        config = None

    if args.model_name_or_path is not None:
        if config is None:
            model = EnClapBartForConditionalGeneration.from_pretrained(
                args.model_name_or_path
            )
        else:
            model = EnClapBartForConditionalGeneration.from_pretrained(
                args.model_name_or_path, config=config
            )
    else:
        model = EnClapBartForConditionalGeneration(config=config)

    # Set the generation config
    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    # Set max encodec length based on the shape of the positional encoding
    max_encodec_length = model.config.max_position_embeddings - 2
    label_pad_token_id = (
        -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    preprocessor = Preprocessor(
        args.encodec_base_path,
        args.clap_base_path,
        tokenizer,
        model.config.max_position_embeddings,
        args.encodec_masking_prob,
        args.encodec_masking_span,
        label_pad_token_id,
        model.config.encodec_vocab_size,
        args.eval_num_captions,
    )

    with accelerator.main_process_first():
        train_dataset = raw_datasets["train"].map(
            preprocessor.preprocess_train,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        train_dataset.set_format(
            "pt",
            columns=[
                "input_ids",                
                "clap_embedding",
                "encodec_mask",
                "attention_mask",
                "mcm_labels",
                "labels",
                "decoder_attention_mask",
            ],
        )

        # Temporarily set max_target_length for validation.
        eval_dataset = raw_datasets_eval["validation"].map(
            preprocessor.preprocess_eval,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        eval_dataset.set_format(
            "pt",
            columns=["input_ids", "attention_mask", "clap_embedding"],
            output_all_columns=True,
        )

    train_data_collator = DataCollatorForEnClapBart(
        tokenizer=tokenizer,
        model=model,
        return_tensors="pt",
        label_pad_token_id=label_pad_token_id,
        max_length=max_encodec_length,
        num_rvq=config.num_rvq,
        input_pad_token_id=config.encodec_pad_token_id
    )
    valid_data_collator = EvalDataCollatorForEnClapBart(
        tokenizer=tokenizer,
        model=model,
        return_tensors="pt",
        label_pad_token_id=label_pad_token_id,
        max_length=max_encodec_length,
        num_rvq=config.num_rvq,
        input_pad_token_id=config.encodec_pad_token_id
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=train_data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=valid_data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.lr_scheduler_type == "inverse_sqrt" and hasattr(args, "time_scale"):
        lr_scheduler = get_inverse_sqrt_schedule(
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            timescale=args.time_scale,
        )
    else:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

    # Prepare everything with our `accelerator`.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        accelerator.init_trackers(args.logging_dir)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    if args.split_batches:
        total_batch_size = int(total_batch_size / accelerator.num_processes)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if not args.overwrite_output_dir and os.path.exists(
        os.path.join(args.output_dir, "checkpoints")
    ):
        if args.resume_from_checkpoint is not None:
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [
                f
                for f in os.scandir(os.path.join(args.output_dir, "checkpoints"))
                if f.is_dir()
            ]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ].name  # Sorts folders by date modified, most recent checkpoint is the last
            accelerator.print(f"Resumed from checkpoint: {dirs[-1]}")
            accelerator.load_state(dirs[-1])
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_stepp

    # update the progress_bar if load from checkpoint
    if args.with_tracking:
        total_loss = 0
        logging_loss = 0
        before_epoch_loss = 0

        if args.encodec_masking_prob > 0:
            total_mcm_loss = 0
            logging_mcm_loss = 0
            before_epoch_mcm_loss = 0

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader
        logger.info(f"***** Running epoch {epoch} *****")
        epoch_iterator = tqdm(
            active_dataloader,
            desc="Training",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
            colour="CYAN",
        )
        for step, batch in enumerate(epoch_iterator):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += outputs.lm_loss.item()
                    if args.encodec_masking_prob > 0:
                        if outputs.mcm_loss is not None:
                            total_mcm_loss += outputs.mcm_loss.item()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), max_norm=args.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    completed_steps += 1
                    # Add loss information to tqdm
                    epoch_iterator.set_postfix(loss=total_loss / completed_steps)

                    if completed_steps % args.logging_steps == 0:
                        train_log = {
                            "train/learning_rate": lr_scheduler.get_last_lr()[0]
                        }
                        train_log["train/loss"] = (
                            total_loss - logging_loss
                        ) / args.logging_steps
                        logging_loss = total_loss
                        if args.encodec_masking_prob > 0:
                            train_log["train/mcm_loss"] = (
                                total_mcm_loss - logging_mcm_loss
                            ) / args.logging_steps
                            logging_mcm_loss = total_mcm_loss
                        accelerator.log(train_log, step=completed_steps)

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(
                            args.output_dir, "checkpoints", output_dir
                        )
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        gen_kwargs = {
            "max_length": args.val_max_target_length,
        }
        predictions = []
        references = []
        eval_iterator = tqdm(
            eval_dataloader,
            desc="Validation",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
            colour="MAGENTA",
        )
        for step, batch in enumerate(eval_iterator):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    clap_embedding=batch["clap_embedding"],
                    encodec_mask=batch["encodec_mask"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                generated_tokens = generated_tokens.cpu().numpy()
                captions = batch["captions"]

                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )

                predictions.extend(decoded_preds)
                references.extend(captions)

        logger.info("Evaluating predictions...")
        result = evaluate(predictions, references, metrics=metric_list)

        # Gather Result
        result = {k: v.cuda() for k, v in result[0].items()}
        result = accelerator.gather_for_metrics(result)
        # Log the average of metrics among the processes
        if accelerator.num_processes > 1:
            result = {f"eval/{k}": round(v.mean().item(), 4) for k, v in result.items()}
        else:
            result = {f"eval/{k}": round(v.item(), 4) for k, v in result.items()}
        logger.info(result)

        if args.with_tracking:
            result["train/epoch_train_loss"] = (total_loss - before_epoch_loss) / len(
                train_dataloader
            )
            result["train/steps"] = completed_steps
            before_epoch_loss = total_loss
            if args.encodec_masking_prob > 0:
                result["train/epoch_mcm_loss"] = (
                    total_mcm_loss - before_epoch_mcm_loss
                ) / len(train_dataloader)
                before_epoch_mcm_loss = total_mcm_loss
            accelerator.log(result, step=epoch)

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, "checkpoints", output_dir)
            accelerator.save_state(output_dir)
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.config.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
