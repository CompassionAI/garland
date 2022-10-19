# CompassionAI project Garland - neural machine translation from classical Tibetan

Machine translation from classical literary Tibetan. Current focus is on:

- Custom neural machine translation models that build on Hugging Face Transformers.
- Using short sentence translation models published by the big research labs, such as FAIR, as backbone models for translating long texts.

Eventually these techniques may generalize to other low resource languages.

## Installation

There are two modes for this library - inference and research. We provide instructions for Linux.

 - Inference should work on MacOS and Windows _mutatis mutandis_.
 - We *very strongly* recommend doing research *only* on Linux. We will not provide any support to people trying to perform research tasks without installing Linux.

### Virtual environment

We strongly recommend using a virtual environment for all your Python package installations, including anything from CompassionAI. To facilitate this, we provide a simple Conda environment YAML file in the CompassionAI/common repo. We recommend first installing miniconda, see <https://docs.conda.io/en/main/miniconda.html>. We then recommend installing Mamba, see <https://github.com/mamba-org/mamba>.

```bash
bash Miniconda3-latest-Linux-x86_64.sh
conda install mamba -c conda-forge
cd compassionai/common
mamba env create -f env-minimal.yml -n my-env
conda activate my-env
```

### Inference

Just install with pip:

```bash
pip install compassionai-garland
```

### Research

Begin by installing for inference. Then install the CompassionAI data registry repo and set two environment variables:

```bash
$CAI_TEMP_PATH
$CAI_DATA_BASE_PATH
```

We strongly recommend setting them with conda in your virtual environment:

```bash
conda activate my-env
conda env config vars set CAI_TEMP_PATH=#directory on a mountpoint with plenty of space, does not need to be fast
conda env config vars set CAI_DATA_BASE_PATH=#absolute path to the CompassionAI data registry
```

Our code uses these environment variables to load datasets from the registry, output processed datasets and store training results.

## Usage

### Inference

This is a supporting library for our main inference repos, such as Lotsawa. You shouldn't need to use it directly.

### Research

This library implements neural machine translation models from classical Tibetan to English, with experiments for other target languages as well.

- Dataset preparation code, especially see `cai_garland/data/parallel_dataset_prep.py`.
- Implementation of modified tokenizers and neural model architectures that builds on Hugging Face Transformers.
- Training drivers to fine-tune models on tasks relevant to translation, such as translation itself or text segmentation.
- Utility code for the above, including simple libraries of preprocessors and segmenters, as well as a translation utility class that implements the core loops of our contextual translation algorithms.