import transformers
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset 
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from utils import resize_and_pad
from typing import Any, List, Dict, Union, Optional
import transformers
from dataclasses import dataclass
import json
from tqdm import tqdm

# PROMPT = "Generate underlying data table of the figure below:"

class ChartParametersDataset(Dataset):
    def __init__(self, data_root) -> None:
        self.data_root = data_root
        self.summary = data_root + "/summary.json"
        self.file_list = self._process_summary()
        self.prompt = "Generate underlying data and visual attributes of the chart images below:"

    def _process_summary(self):
        with open(self.summary, 'r') as f:
            summary = json.load(f)
        file_list = []
        print("preparing dataset:")
        for uid in tqdm(summary.keys()):
            item = summary[uid]
            image_path = os.path.join(self.data_root + item["original_image"])
            text = {**item["underlying_data"],
                    **item["visual_attributes"]}
            text = json.dumps(text)
            file_list.append((image_path, text))

            edited_image_path = os.path.join(self.data_root + item["edited_image"])
            edited_text = {**item["edited_underlying_data"],
                           **item["edited_visual_attributes"]}
            edited_text = json.dumps(edited_text)
            file_list.append((edited_image_path, edited_text))
        return file_list

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        item = self.file_list[idx]
        img = Image.open(item[0]).convert("RGB")
        img = resize_and_pad(img, target_size=(800,800))
        txt = item[1]
        
        inputs = {
            "text":txt,
            "prompt":self.prompt,
            "image":img}
        return inputs

class ChartEditsDataset(Dataset):
    def __init__(self, data_root) -> None:
        self.data_root = data_root
        self.summary = data_root + "/summary.json"
        self.file_list = self._process_summary()

    def _process_summary(self):
        with open(self.summary, 'r') as f:
            summary = json.load(f)
        file_list = []
        print("preparing dataset:")
        for uid in tqdm(summary.keys()):
            item = summary[uid]
            image_path = os.path.join(self.data_root + item["original_image"])
            text = {**item["edited_underlying_data"],
                    **item["edited_visual_attributes"]}
            text = json.dumps(text)

            original_text = {**item["visual_attributes"],
                             **item["underlying_data"]}
            original_text = json.dumps(original_text)
            for prompt in item['editing_prompts']:
                file_list.append((image_path, text, prompt, original_text))
        return file_list

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        item = self.file_list[idx]
        img = Image.open(item[0]).convert("RGB")
        img = resize_and_pad(img, target_size=(800,800))
        txt = item[1]
        prompt = item[2]
        original_txt = item[3]
        
        inputs = {
            "text":txt,
            "original_text":original_txt,
            "prompt":prompt,
            "image":img}
        return inputs

@dataclass
class DataCollator:
    tokenizer: transformers.PreTrainedTokenizerBase
    processor: AutoProcessor
    padding: str = "max_length"
    max_length: Optional[int] = 1024
    max_patches: Optional[int] = 512
    add_special_tokens: bool = True
    return_tensors: str = "pt"
    truncation: bool = True
    inference : bool = False

    def __call__(self, batch) -> Dict[str, Any]:
        new_batch = {}
        texts = [item["text"] for item in batch]
        images = [item["image"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        original_text = [item["original_text"] for item in batch]

        if not self.inference:
            text_inputs = self.tokenizer(text=texts, padding="max_length", 
                                     return_tensors=self.return_tensors,
                                     add_special_tokens=self.add_special_tokens, 
                                     max_length=self.max_length, 
                                     truncation=self.truncation)
            new_batch["labels"] = text_inputs.input_ids

            original_text_compare = self.tokenizer(text=original_text, padding="max_length", 
                                     return_tensors=self.return_tensors,
                                     add_special_tokens=self.add_special_tokens, 
                                     max_length=self.max_length, 
                                     truncation=self.truncation)
            
            diff_idx = (text_inputs.input_ids != original_text_compare.input_ids).int()
            new_batch["diff_idx"] = diff_idx
    
    
        encoding = self.processor(images=images, text=prompts,
                            return_tensors=self.return_tensors, 
                            add_special_tokens=self.add_special_tokens, 
                            max_patches=self.max_patches)

        new_batch["flattened_patches"] = encoding["flattened_patches"]
        new_batch["attention_mask"] = encoding["attention_mask"]

        return new_batch