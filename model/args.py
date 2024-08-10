import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--run_name", type=str, default='run1', help="A descriptor for the run. This will be used as the folder name."
    )
    parser.add_argument(
        "--mode", type=str, default='train', help="A selection from 'train', 'finetune'"
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A folder contains folders for image, json file for training."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A folder contains folders for image, json file for validation/test."
    )
    parser.add_argument(
        "--inference_file", type=str, default=None, help="A folder contains file for inference"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--processor_config",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
        required=True,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="The maximum length of the sequence to be generated.",
        required=True,
    )
    parser.add_argument(
        "--new_loss",
        action="store_true",
        help="Whether to use the new loss function for the chart edits finetune",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="The number of processes to use for the dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", 
                        type=float, 
                        default=1e-5, 
                        help="Weight decay to use."
    )
    parser.add_argument("--num_train_epochs", 
                        type=int, 
                        default=15, 
                        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=1000, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs/",
        help=(
            'The logging directory to save logs and results. Only applicable when `--with_tracking` is passed. '
        ),
    )
    args = parser.parse_args()
        # Sanity checks
    
    if args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    
    args.output_dir = args.output_dir + args.run_name
    args.log_dir = args.log_dir + args.run_name
    
    return args