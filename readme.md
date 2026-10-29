<div align="center">

# ChartReformer: Natural Language-Driven Chart Image Editing

**Pengyu Yan, [Mahesh Bhosale](https://bhosalems.github.io/), Jay Lal, Bikhyat Adhikari, David Doermann**

University at Buffalo

**ICDAR 2024**: International Conference on Document Analysis and Recognition

[![Paper](https://img.shields.io/badge/Paper-ICDAR%202024-blue)](https://link.springer.com/chapter/10.1007/978-3-031-70533-5_26)
[![arXiv](https://img.shields.io/badge/arXiv-2403.00209-b31b1b.svg)](https://arxiv.org/abs/2403.00209)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-ChartReformer-ffd21e)](https://huggingface.co/Pengyu965/ChartReformer)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-ChartCraft-ffd21e)](https://huggingface.co/datasets/Pengyu965/ChartCraft)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

This is the official repository for the ICDAR 2024 paper **ChartReformer**, the first work to systematically study
**chart image editing**: changing a chart image from a natural-language instruction, without access to its original
data or plotting code.

## 📰 News

- **[Oct 2026]** For easier access, the **model checkpoints** and the **ChartCraft dataset** are now also hosted on
  Hugging Face: [🤗 Model](https://huggingface.co/Pengyu965/ChartReformer) ·
  [🤗 Dataset](https://huggingface.co/datasets/Pengyu965/ChartCraft). These are the original 2024 releases,
  unchanged. They are not a new version or new work.
- **[2024]** ChartReformer was published at **ICDAR 2024**
  ([Springer](https://link.springer.com/chapter/10.1007/978-3-031-70533-5_26)).
- **[Mar 2024]** Preprint released on [arXiv](https://arxiv.org/abs/2403.00209).

## ✨ Overview

Most charts we meet are just **images**: in papers, reports, slides, and on the web. Their data tables and plotting
code are usually gone. So changing a chart's colors, turning a line chart into a bar chart, or adding a data series
used to mean rebuilding the chart from scratch.

ChartReformer edits the chart image directly. Given a chart and an instruction such as *"Convert this into a stacked
bar chart"* or *"Remove the data series for 2010"*, the model predicts the **data table** and **visual attributes** of
the edited chart. A replotter then renders them into the new chart image.

**Contributions**

- 🥇 **The first work to systematically study chart editing.** We define and standardize the task, with a taxonomy
  of edits: **style**, **layout**, **format**, and **data-centric**.
- 📚 **ChartCraft**, a chart-editing dataset of about 100K pairs. Each pair has the original and edited chart images,
  their data tables, their visual attributes, and natural-language instructions. To our knowledge, it was the first
  public chart-editing dataset.
- 📏 **Evaluation metrics for chart editing.** The Visual Attribute Edit Score (VAES) measures the predicted visual
  attributes. RMS measures the data table, and SSIM measures the re-rendered image.
- 🛠️ **The ChartReformer model.** It predicts a structured description of the edited chart instead of plotting
  code, so 99.9% of its outputs render successfully and the edits are precise.

### Edit taxonomy

![Taxonomy](images/chart_reformer_samples.png?raw=true)

| Edit category | Aspect | Supported edits |
|---|---|---|
| **Style** | Colors / Line / Pattern / Text | plot color · line style, marker · bar hatch pattern · font family, font size |
| **Layout** | Axes / Legend | grid lines · legend position and internal layout |
| **Format** | Plot type | line ↔ grouped bar · line ↔ stacked bar · grouped bar ↔ stacked bar |
| **Data-centric** | Filtering / Addition | range- and series-based filtering · add or remove data points and data series |

### Framework

![Framework](images/chart_reformer_method_diagram.png?raw=true)

ChartReformer is built on Pix2Struct and initialized from [DePlot](https://huggingface.co/google/deplot). It is
trained in two stages:

1. **Chart de-rendering pre-training.** The model learns to extract the data table and the full set of visual
   attributes from a chart image.
2. **Chart-editing fine-tuning.** The model learns to take a chart and an edit instruction, and to output the data
   and visual attributes of the **edited** chart.

## 📦 Model and dataset

| Resource | 🤗 Hugging Face (recommended) | Original download |
|---|---|---|
| **ChartReformer checkpoints** (chart editing + de-rendering) | [`Pengyu965/ChartReformer`](https://huggingface.co/Pengyu965/ChartReformer) | [Google Drive](https://drive.google.com/file/d/1gu8gy_K2BvHd3OYi7aXYeVgBHUF7CYNc/view?usp=sharing) |
| **ChartCraft dataset** (about 100K edit pairs) | [`Pengyu965/ChartCraft`](https://huggingface.co/datasets/Pengyu965/ChartCraft) | [Google Drive](https://drive.google.com/file/d/1Q5TfuwkrcQrK7V3usQ3w3oaVSEYvJIBq/view?usp=sharing) |

**Checkpoints.** The Hugging Face model repo has two checkpoints:

- the **chart-editing** model (stage 2) at the root, which is the default;
- the **chart de-rendering** model (stage 1) in the `pretrain/` subfolder.

**Dataset splits.**

| Split | Edit pairs |
|---|---:|
| `train` | 88,800 |
| `validation` | 11,100 |
| `test` | 5,550 |
| `test_small` | 555 |

There are five chart types: line, plus grouped and stacked bar charts in vertical and horizontal orientation.

```python
from datasets import load_dataset
ds = load_dataset("Pengyu965/ChartCraft", split="test")
```

The [dataset card](https://huggingface.co/datasets/Pengyu965/ChartCraft) has a snippet that exports the Hugging Face
version back to the original folder layout (`images/`, `edited_images/`, `summary.json`) used by the training code
below.

## 🚀 Quick start

```bash
pip install torch transformers pillow
```

```python
import json, torch
from PIL import Image
from transformers import AutoProcessor, Pix2StructForConditionalGeneration

def resize_and_pad(image, target_size=(800, 800), background_color=(255, 255, 255)):
    """Keep the aspect ratio and pad to 800x800, as done during training."""
    w, h = image.size
    ratio = min(target_size[0] / w, target_size[1] / h)
    new_size = (int(w * ratio), int(h * ratio))
    canvas = Image.new("RGB", target_size, background_color)
    canvas.paste(image.resize(new_size), ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2))
    return canvas

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("Pengyu965/ChartReformer")
model = Pix2StructForConditionalGeneration.from_pretrained("Pengyu965/ChartReformer").to(device).eval()

chart = resize_and_pad(Image.open("chart.png").convert("RGB"))
instruction = "Change the line chart into a grouped vertical bar chart"

inputs = processor(images=chart, text=instruction, return_tensors="pt", max_patches=512).to(device)
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=1024)
edited = json.loads(processor.batch_decode(output_ids, skip_special_tokens=True)[0])

# Render the edited chart with the replotter in this repo (run from the model/ folder)
from Chart_Reploter import Replot
Replot(edited).fig.savefig("edited_chart.png")
```

The [model card](https://huggingface.co/Pengyu965/ChartReformer) also covers de-rendering with the `pretrain/`
checkpoint and describes the output JSON format.

## 🏋️ Training and evaluation

All code is in [`model/`](model). The checkpoints were trained with `transformers==4.37.2` and 🤗 Accelerate on
4 GPUs.

```bash
pip install torch transformers accelerate datasets pandas matplotlib scikit-image json_repair tqdm wandb
```

| File | Purpose |
|---|---|
| `train_new.py`, `args.py` | Training entry point for both stages (`--mode train` or `--mode finetune`) |
| `dataloader.py` | `ChartParametersDataset` (de-rendering) and `ChartEditsDataset` (editing) |
| `chartedits_loss.py` | Optional edit-aware loss that balances changed and unchanged tokens (`--new_loss`) |
| `run_train.sh` | Stage 1: de-rendering pre-training, initialized from DePlot |
| `run_finetune.sh` | Stage 2: chart-editing fine-tuning, initialized from the stage-1 checkpoint |
| `simple_inference.py`, `distributed_inference_task{1,2}.py` | Single-GPU and multi-GPU inference for de-rendering (task 1) and editing (task 2) |
| `Chart_Reploter.py` | Renders a chart from predicted data and visual attributes (Matplotlib) |
| `eval_metrics.py`, `SSIM_eval.py` | VAES, RMS, SSIM, and success-rate evaluation |
| `processor_config/` | Pix2Struct processor and tokenizer configuration |

To train:

1. Put the dataset in the original layout: `data/{train,val,test,test_small}/` with `images/`, `edited_images/`,
   and `summary.json`. Either unpack the Google Drive archive or export it from Hugging Face.
2. Stage 1: `bash run_train.sh`. Set `--model_name_or_path` to `google/deplot` or to a local copy of it.
3. Stage 2: `bash run_finetune.sh`. Point `--model_name_or_path` to the best stage-1 checkpoint.

Data and checkpoint paths are set inside the shell and Python scripts. Edit them to match your setup.

## 📊 Results

Results of ChartReformer on the ChartCraft test set (5.5K samples):

| Metric | Style | Layout | Format | Data-centric | Total |
|---|---:|---:|---:|---:|---:|
| VAES (F1) | 86.63 | 89.65 | 90.53 | 84.24 | 86.32 |
| RMS (F1) | 90.35 | 89.23 | 91.22 | 88.36 | 89.33 |
| SSIM | 84.50 | 82.06 | 85.19 | 83.07 | 83.65 |
| Success rate | 99.81 | 100 | 99.90 | 99.77 | 99.90 |

Comparison with ChartLlama on a 10% test subset, shown as SSIM / success rate:

| Method | Style | Layout | Format | Data-centric | Total |
|---|---:|---:|---:|---:|---:|
| ChartLlama | 73.09 / 10.63 | 64.77 / 26.47 | 64.32 / 8.04 | 67.68 / 19.33 | 67.46 / 16.95 |
| **ChartReformer** | **83.30 / 100** | **82.55 / 100** | **84.22 / 100** | **81.43 / 100** | **82.39 / 100** |

### Qualitative results

![Qualitative Evaluations](images/qualitatively_eval_self.jpg?raw=true)

## 📝 Citation

If you find ChartReformer or ChartCraft useful, please cite:

```bibtex
@inproceedings{yan2024chartreformer,
  title={ChartReformer: Natural Language-Driven Chart Image Editing},
  author={Yan, Pengyu and Bhosale, Mahesh and Lal, Jay and Adhikari, Bikhyat and Doermann, David},
  booktitle={International Conference on Document Analysis and Recognition},
  pages={453--469},
  year={2024},
  organization={Springer}
}
```

## 📄 License

This project is released under the [MIT License](LICENSE).
