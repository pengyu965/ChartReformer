import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import resize_and_pad
from multiprocessing import Process
from accelerate import PartialState, Accelerator

from transformers import AutoProcessor, Pix2StructForConditionalGeneration


workspace = "/scratch2/pyan4/Workspace/"
test_img_path = workspace + "/ChartDerender/data/new_edits_data/test/images/"

test_pre_path = "./results/pretraining_new_fixed_data/pre_on_original_images/"
os.makedirs(test_pre_path, exist_ok=True)

distributed_state = PartialState()

processor = AutoProcessor.from_pretrained("./processor_config")
model = Pix2StructForConditionalGeneration.from_pretrained("./checkpoints/pretrain_new_edits_data/step_48000_best")
model.to(distributed_state.device)

PROMPT = "Generate underlying data and visual attributes of the chart images below:"
MAX_LENGTH = 1024

def process_input(chart_ids):
    inputs = []
    for chart_id in chart_ids:
        image = Image.open(test_img_path + chart_id + ".png").convert('RGB')
        image = resize_and_pad(image, target_size=(800,800))
        processed_input = processor(images=image,
                                    text=PROMPT,
                                    return_tensors="pt",
                                    add_special_tokens=True,
                                    max_patches=512,
                                    max_length=MAX_LENGTH)
        processed_input = processed_input.to(distributed_state.device)
        inputs.append({"chart_id":chart_id, "input":processed_input})

    return inputs

def chunks(lst, chunk_size):
    """Yield successive chunk-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


chart_ids_list = [x[:-4] for x in os.listdir(test_img_path)]
chunck_chartids_list = list(chunks(chart_ids_list, 1000))

accelerator = Accelerator()
pbar=tqdm(total=len(chart_ids_list)) 

for i, chart_ids in enumerate(chunck_chartids_list):
    print("Processing {}/{} chuncks".format(i, len(chunck_chartids_list)))
    with distributed_state.split_between_processes(chart_ids) as chart_ids:
        inputs = process_input(chart_ids)
        batch_generated_id = []
        for input in inputs:
            generated_id = model.generate(flattened_patches=input["input"]['flattened_patches'], 
                                        attention_mask=input["input"]['attention_mask'],
                                        max_new_tokens=MAX_LENGTH,
                                        use_cache=True)
            batch_generated_id.append({"chart_id":input["chart_id"], "generated_id":generated_id})
            
            pbar.update(accelerator.num_processes)

        for item in batch_generated_id:
            generated_caption = processor.batch_decode(item["generated_id"], skip_special_tokens=True)[0]
            with open(test_pre_path + item["chart_id"] + ".json",'w') as f:
                f.write(generated_caption)
