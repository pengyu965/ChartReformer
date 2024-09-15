import os
from PIL import Image
from tqdm import tqdm
from utils import resize_and_pad
import json_repair
import json 
import random
import torch
import shutil
from Chart_Reploter import Replot

from transformers import AutoProcessor, Pix2StructForConditionalGeneration


workspace = "/scratch2/pyan4/Workspace/"
test_path = workspace + "/ChartDerender/data/human_eval/PMC/"

pre_save_path = "./results/human_eval/PMC/"
os.makedirs(pre_save_path, exist_ok=True)

model_path = "./checkpoints/pretrain/step_21000_best"

device = "cuda:1" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("./processor_config")
model = Pix2StructForConditionalGeneration.from_pretrained(model_path)
model = model.to(device)

# prompt = "Generate underlying data and visual attributes of the chart images below:"
prompt = "Generate underlying data table and plotting parameters of the figure below:"

for chart_type in os.listdir(test_path):
    # if chart_type == "h_bar":
    #     continue
    os.makedirs(os.path.join(pre_save_path, chart_type), exist_ok=True)

    chart_type_path = os.path.join(test_path, chart_type)
    for image_f in tqdm(os.listdir(chart_type_path)):
        chart_id = os.path.splitext(os.path.basename(image_f))[0]
        chart_path = os.path.join(chart_type_path, image_f)

        image = Image.open(chart_path).convert('RGB')
        image = resize_and_pad(image, target_size=(800,800))

        processed_input = processor(images=image,
                                    text=prompt,
                                    return_tensors="pt", 
                                    add_special_tokens=True,
                                    max_patches=512,
                                    max_length=1024)
        
        processed_input = processed_input.to(device)

        output = model.generate(flattened_patches=processed_input['flattened_patches'], 
                                attention_mask=processed_input['attention_mask'],
                                max_new_tokens=1024,
                                use_cache=True)
        generated_caption = processor.batch_decode(output, skip_special_tokens=True)[0]
        
        with open(os.path.join(pre_save_path, chart_type, chart_id + ".json"), "w") as f:
            f.write(generated_caption)
        
        shutil.copy2(chart_path, os.path.join(pre_save_path, chart_type, image_f))
    
        try:
            pre_json = json_repair.loads(generated_caption)
            if 'underlying_data' in pre_json.keys():
                pre_json['data_table'] = pre_json['underlying_data']
                del pre_json['underlying_data']
            replot = Replot(pre_json)
            replot.fig.savefig(os.path.join(pre_save_path, chart_type, image_f[:-4] + "_edited.jpg"))
        except:
            continue 
        
