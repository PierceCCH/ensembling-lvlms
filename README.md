# [WIP] Ensembling methods for mitigating hallucination in MLLMs

## Setup

Use conda or venv to create a new python 3.12 environment and install the required packages.
```
pip install -r requirements.txt
```

### Benchmarks

Benchmarks are loaded from Huggingface datasets.

#### [POPE](https://github.com/RUCAIBox/POPE)
#### [HallucinationBench](https://github.com/tianyi-lab/HallusionBench)


### Directory structure
```
|-- evaluations
|   |-- HallusionBench
|   |-- POPE
|   |-- ...
|-- models
|   |-- (more methods to be added)
|   |-- ...
|-- README.md
|-- requirements.txt

```