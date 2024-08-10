import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
from huggingface_hub import Repository, create_repo
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
from args import parse_args

from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from dataloader import ChartParametersDataset, ChartEditsDataset, DataCollator
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup

logger = get_logger(__name__)


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_config"] = ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.log_dir)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
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

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    # Load the processor and collator
    processor = AutoProcessor.from_pretrained(args.processor_config)
    collator = DataCollator(tokenizer=processor.tokenizer, 
                            processor=processor,
                            max_length = args.max_length)
    
    # Load the model
    model = Pix2StructForConditionalGeneration.from_pretrained(args.model_name_or_path)
    
    # Choose the loss function
    if args.new_loss:
        from chartedits_loss import ChartEdits_CrossEntropy
        loss_fn = ChartEdits_CrossEntropy()
    
    # Select the correct dataset
    if args.mode == "train":
        Dataset = ChartParametersDataset
    elif args.mode == "finetune":
        Dataset = ChartEditsDataset


    ## When mode is train, train the model with pre-training dataloader ==========
    # Load the dataset and create dataloaders
    train_dataset = Dataset(args.train_file)
    eval_dataset = Dataset(args.validation_file)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=collator,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )

    # Calculate the number of training steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # create optimizer and warmup scheduler
    optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, 
                        lr=args.learning_rate, weight_decay=args.weight_decay)

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=args.num_warmup_steps * accelerator.num_processes, 
                                                num_training_steps=args.max_train_steps
                                                if overrode_max_train_steps
                                                else args.max_train_steps * accelerator.num_processes,)
    
    # Prepare everything with our `accelerator`.

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)
    

    # After accelerator.prepare, so we need to recalculate our total training steps 
    # as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = "consine scheduler with warm up"
        accelerator.init_trackers(args.run_name, experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Get the trainer state
    train_state = {
        "run_name":args.run_name,
        "max_length":args.max_length,
        "per_device_train_batch_size":args.per_device_train_batch_size,
        "per_device_eval_batch_size":args.per_device_eval_batch_size,
        "gradient_accumulation_steps":args.gradient_accumulation_steps,
        "learning_rate":args.learning_rate,
        "weight_decay":args.weight_decay,
        "num_warmup_steps":args.num_warmup_steps,
        "train_batch_size":total_batch_size,
        "num_train_epochs":args.num_train_epochs,
        "max_steps":args.max_train_steps,
        "checkpointing_steps":args.checkpointing_steps,
    }
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    # TODO: Need to update this later.
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        path = os.path.basename(args.resume_from_checkpoint)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                diff_idx = batch.pop("diff_idx")
                outputs = model(**batch)
                if args.new_loss:
                    loss = loss_fn(outputs, batch, diff_idx)
                else:
                    loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if args.with_tracking:
                accelerator.log(
                    {
                        "train_loss": loss.item(),
                        "step": completed_steps,
                        "lr": lr_scheduler.get_last_lr()[0],
                    },
                    step=completed_steps,
                )

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    model.eval()
                    losses = []
                    for step, batch in enumerate(eval_dataloader):
                        with torch.no_grad():
                            diff_idx = batch.pop("diff_idx")
                            outputs = model(**batch)
                            if args.new_loss:
                                loss = loss_fn(outputs, batch, diff_idx)
                            else:
                                loss = outputs.loss

                        losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

                    losses = torch.cat(losses)
                    try:
                        eval_loss = torch.mean(losses)
                        perplexity = math.exp(eval_loss)
                    except OverflowError:
                        perplexity = float("inf")

                    logger.info(f"step {completed_steps}: perplexity: {perplexity} eval_loss: {eval_loss}")

                    if args.with_tracking:
                        accelerator.log(
                            {
                                "step_perplexity": perplexity,
                                "step_eval_loss": eval_loss.item(),
                                "step_train_loss": total_loss.item() / (completed_steps - epoch * len(train_dataloader)),
                                "epoch": completed_steps / len(train_dataloader),
                                "step": completed_steps,
                                "current_lr": lr_scheduler.get_last_lr()[0],
                            },
                            step=completed_steps,
                        )
                    
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

                    train_state["epoch"] = epoch
                    train_state["global_step"] = completed_steps
                    train_state["train_loss"] = loss.item()
                    train_state["eval_loss"] = eval_loss.item()
                    train_state["current_lr"] = lr_scheduler.get_last_lr()[0]
                    with open(output_dir + "/train_state.json", "w") as f:
                        json.dump(train_state, f, indent=2)
                    
                    model.train()
            if completed_steps >= args.max_train_steps:
                break


        if args.checkpointing_steps == "epoch":
            model.eval()
            losses = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    diff_idx = batch.pop("diff_idx")
                    outputs = model(**batch)
                    if args.new_loss:
                        loss = loss_fn(outputs, batch, diff_idx)
                    else:
                        loss = outputs.loss

                losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

            losses = torch.cat(losses)
            try:
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")

            logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

            if args.with_tracking:
                accelerator.log(
                    {
                        "epoch_perplexity": perplexity,
                        "epoch_eval_loss": eval_loss.item(),
                        "epoch_train_loss": total_loss.item() / len(train_dataloader),
                        "epoch": epoch,
                        "step": completed_steps,
                        "current_lr": lr_scheduler.get_last_lr()[0],
                    },
                    step=completed_steps,
                )
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            train_state["epoch"] = epoch
            train_state["global_step"] = completed_steps
            train_state["train_loss"] = total_loss.item() / len(train_dataloader)
            train_state["eval_loss"] = eval_loss.item()
            train_state["current_lr"] = lr_scheduler.get_last_lr()[0]
            with open(output_dir + "/train_state.json", "w") as f:
                json.dump(train_state, f, indent=2)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir + "/final/", is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
    


if __name__ == "__main__":
    main()