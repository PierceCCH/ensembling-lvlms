# [WIP] Enhancing prompts with image tags for mitigating hallucination in MLLMs

## Setup

Use conda or venv to create a new python 3.12 environment and install the required packages.
```
pip install -r requirements.txt
```

## Download RAM weights

If you want to generate image tags, download [RAM++ weights](https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth) from the following link and place them in the `models/ram/pretrained` directory.


### Benchmarks

Benchmarks are loaded from Huggingface datasets.

#### [POPE](https://github.com/RUCAIBox/POPE)
