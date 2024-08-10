CUDA_VISIBLE_DEVICES=1,2,3,4 \
accelerate launch --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision fp16 --dynamo_backend no \
train_new.py \
--run_name  "finetune" \
--mode 'finetune' \
--train_file "../../data/train/" \
--validation_file "../../data/val/" \
--model_name_or_path './checkpoints/pretrain/step_48000_best' \
--processor_config './processor_config' \
--max_length 1024 \
--per_device_train_batch_size 3 \
--per_device_eval_batch_size 4 \
--dataloader_num_workers 4 \
--learning_rate 2e-5 \
--weight_decay 1e-5 \
--num_train_epochs 20 \
--gradient_accumulation_steps 1 \
--num_warmup_steps 500 \
--checkpointing_steps 5000 \
--output_dir "./checkpoints/" \
--with_tracking \
--report_to 'wandb' \
--log_dir './logs/'  \
--new_loss