import os
from PIL import Image
from tqdm import tqdm
from utils import resize_and_pad
from multiprocessing import Process
from accelerate import PartialState, Accelerator
import argparse 
import json 
import random

from transformers import AutoProcessor, Pix2StructForConditionalGeneration

workspace = "/scratch2/pyan4/Workspace/"

test_path = workspace + "/ChartDerender/data/test_small/"
pre_save_path = "./results/data_small/"
os.makedirs(pre_save_path, exist_ok=True)
model_path = "./checkpoints/finetune/step_20000_best"


distributed_state = PartialState()

processor = AutoProcessor.from_pretrained("./processor_config")
model = Pix2StructForConditionalGeneration.from_pretrained(model_path)
model.to(distributed_state.device)

MAX_LENGTH = 1024

def process_input(summary):
    inputs = []
    for item in summary:
        chart_id = os.path.splitext(os.path.basename(item["original_image"]))[0]
        
        image = Image.open(test_path + item["original_image"]).convert('RGB')
        image = resize_and_pad(image, target_size=(800,800))
        prompts = item["editing_prompts"]
        prompt = random.choice(prompts)
        processed_input = processor(images=image,
                                    text=prompt,
                                    return_tensors="pt", 
                                    add_special_tokens=True,
                                    max_patches=512,
                                    max_length=MAX_LENGTH)
        processed_input = processed_input.to(distributed_state.device)
        inputs.append({"chart_id":chart_id, 
                       "input":processed_input, 
                       "prompt":prompt})

    return inputs

def chunk_dict(dictionary, chunk_size):
    """
    Yield successive chunk-sized chunks from a dictionary.

    Args:
        dictionary (dict): The dictionary to yield chunks from.
        chunk_size (int): The size of each chunk.

    Yields:
        dict: Chunk-sized chunks from the dictionary.
    """
    keys = list(dictionary.keys())
    for i in range(0, len(keys), chunk_size):
        yield {k: dictionary[k] for k in keys[i:i + chunk_size]}

def chunks_list(lst, chunk_size):
    """Yield successive chunk-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


with open(test_path + "summary.json") as f:
    summary_j = json.load(f)
summary_list = list(summary_j.values())

chunck_summary = list(chunks_list(summary_list, 1000))

accelerator = Accelerator()
pbar=tqdm(total=len(summary_list)) 

for i, summary in enumerate(chunck_summary):
    print("Processing {}/{} chuncks".format(i, len(chunck_summary)))
    with distributed_state.split_between_processes(summary) as summary:
        inputs = process_input(summary)
        batch_generated_id = []
        for input in inputs:
            generated_id = model.generate(flattened_patches=input["input"]['flattened_patches'], 
                                        attention_mask=input["input"]['attention_mask'],
                                        max_new_tokens=MAX_LENGTH,
                                        use_cache=True)
            batch_generated_id.append({"chart_id":input["chart_id"], 
                                       "generated_id":generated_id, 
                                       "prompt":input["prompt"]})
            
            pbar.update(accelerator.num_processes)

        for item in batch_generated_id:
            generated_caption = processor.batch_decode(item["generated_id"], skip_special_tokens=True)[0]
            with open(pre_save_path + item["chart_id"] + ".json",'w') as f:
                f.write(generated_caption)
            with open(pre_save_path + item["chart_id"] + "_prompt.txt",'w') as f:
                f.write(item["prompt"])

