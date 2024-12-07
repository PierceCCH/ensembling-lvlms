# Enhancing prompts with image tags for mitigating hallucination in LVLMs

___

## Setup

Use conda or venv to create a new python 3.12 environment and install the required packages.
```
pip install -r requirements.txt
```

### Download RAM weights

If you want to generate image tags, download [RAM++ weights](https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth) from the following link and place them in the `models/recognize-anything/pretrained` directory.

___

## Testing

### Benchmark datasets

Benchmarks are loaded using the Huggingface datasets library. In this project, we used [POPE](https://huggingface.co/datasets/lmms-lab/POPE) and [HallusionBench](https://huggingface.co/datasets/lmms-lab/HallusionBench).

### Generating RAM tags

RAM tags for POPE and HallusionBench are generated using either `models/recognize-anything/generate_POPE_tags.ipynb` and `models/recognize-anything/generate_hallusionBench_tags.ipynb` respectively. Each notebook will generate a json file containing tags for each unique image in the dataset.

The tags that we generated can be found in `/results/pope_tags.json` and `/results/hallusionBench_tags.json`.

### Generating LVLM responses

The notebook `/evaluations/prompt_enhancement_pipeline.ipynb` contains the code for generating LVLM responses using our pipeline. In the notebook, specify which benchmark to generate responses for, and whether to double prompt the LVLM. All responses will be generated and saved in the `/results` directory.

### Evaluation of responses

The notebook `/results/results_analysis.ipynb` contains the code for evaluating the generated responses. The notebook will output the accurary, precision, recall and F1 score for the base model, the model with prompts enhanced with image tags, and the model with prompts enhanced with image tags and double prompting. The notebook also generates a confusion matrix for each model.

