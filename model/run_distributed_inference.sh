CUDA_VISIBLE_DEVICES=1,2,4,5 \
accelerate launch --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision fp16 --dynamo_backend no --main_process_port 29500 \
distributed_inference_task2.py \