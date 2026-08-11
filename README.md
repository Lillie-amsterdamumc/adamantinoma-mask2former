# Adamantinoma Mask2Former

**Semantic segmentation of adamantinoma histopathology images using Mask2Former**

This repository contains a customized implementation of [Mask2Former](https://github.com/facebookresearch/Mask2Former) developed for semantic segmentation of adamantinoma histopathology images.

The codebase has been adapted for the project-specific training pipeline, configuration, matching strategy, and whole-slide-image (WSI) processing workflow.

---

## Overview

The project uses **Mask2Former** for semantic segmentation of histopathology images of adamantinoma.

The repository contains:

* Project-specific Mask2Former configuration
* Customized training code
* Customized matching implementation
* WSI processing dependencies
* Reproducible Python environment specifications
* CUDA implementation of Multi-Scale Deformable Attention

The original Mask2Former architecture supports semantic, instance, and panoptic segmentation. In this project, it is used primarily for **semantic segmentation of adamantinoma tissue components**.

---

## Tested Environment

The project has been tested on an **NVIDIA H100 NVL GPU** with the following environment:

| Component                     | Version                                               |
| ----------------------------- | ----------------------------------------------------- |
| Python                        | 3.11.15                                               |
| PyTorch                       | 2.1.0 + CUDA 12.1                                     |
| TorchVision                   | 0.16.0 + CUDA 12.1                                    |
| Detectron2                    | Git commit `b599f139756bd3646a26a909caf86a1a159e53a7` |
| MultiScaleDeformableAttention | 1.0                                                   |
| OpenSlide                     | 1.4.3                                                 |
| FastSlide                     | 0.5.2                                                 |
| WSI-patching                  | 0.5.1                                                 |
| timm                          | 1.0.26                                                |
| NumPy                         | 1.26.4                                                |
| SciPy                         | 1.17.1                                                |

> The versions above correspond to the environment used for development and testing of this project.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/Lillie-amsterdamumc/adamantinoma-mask2former.git
cd adamantinoma-mask2former
```

## 2. Create a Python environment

Create and activate a Python 3.11 environment:

```bash
conda create -n adamantinoma-mask2former python=3.11 -y
conda activate adamantinoma-mask2former
```

Alternatively, an existing compatible Python environment can be used.

## 3. Install dependencies

The main project dependencies are listed in:

```text
requirements.txt
```

Install them with:

```bash
pip install -r requirements.txt
```

For the complete dependency snapshot, see:

```text
requirements-full.txt
requirements-lock.txt
```

> `requirements.txt` is the recommended installation file.
> `requirements-full.txt` and `requirements-lock.txt` are provided for environment reproducibility.

## 4. CUDA and MSDeformAttn

Mask2Former requires the CUDA implementation of **Multi-Scale Deformable Attention (MSDeformAttn)**.

Make sure that the CUDA toolkit is available and that `CUDA_HOME` is correctly configured:

```bash
echo $CUDA_HOME
```

Then compile the CUDA extension:

```bash
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

After compilation, return to the repository root:

```bash
cd ../../../../
```

The CUDA extension should now be available under:

```text
mask2former/modeling/pixel_decoder/ops/
```

For detailed installation and CUDA troubleshooting, see:

**[INSTALL.md](INSTALL.md)**

---

# Project Structure

```text
adamantinoma-mask2former/
│
├── configs/
│   └── ade20k/
│       └── semantic-segmentation/
│
├── datasets/
│
├── demo/
│
├── mask2former/
│   └── modeling/
│       └── pixel_decoder/
│           └── ops/
│
├── mask2former_video/
│
├── tools/
│
├── train_net.py
├── predict.py
│
├── requirements.txt
├── requirements-full.txt
├── requirements-lock.txt
│
├── INSTALL.md
├── GETTING_STARTED.md
├── ADVANCED_USAGE.md
├── MODEL_ZOO.md
└── README.md
```

---

# Training

The main training entry point is:

```text
train_net.py
```

A project-specific configuration is located under:

```text
configs/
```

For example:

```text
configs/ade20k/semantic-segmentation/
```

Training can be launched using the Detectron2/Mask2Former training interface.

Example:

```bash
python train_net.py \
    --config-file configs/ade20k/semantic-segmentation/<CONFIG_FILE>.yaml
```

Additional options can be passed through the Detectron2 configuration system.

For cluster-based training, use the appropriate Slurm job script and GPU configuration for the target HPC environment.

---

# Adamantinoma Segmentation Pipeline

The repository is part of a computational pathology workflow for semantic segmentation of adamantinoma histopathology images.

The general workflow is:

```text
Histopathology WSI
        │
        ▼
WSI preprocessing / patch extraction
        │
        ▼
Annotated training patches
        │
        ▼
Mask2Former semantic segmentation
        │
        ▼
Pixel-level tissue segmentation
        │
        ▼
Downstream quantitative analysis
```

The model is trained to distinguish project-specific tissue components defined by the corresponding annotations and configuration.

---

# Configuration

Project-specific model and training settings are stored under:

```text
configs/
```

The customized training implementation is located in:

```text
train_net.py
```

The matching implementation has also been customized:

```text
mask2former/modeling/matcher.py
```

These modifications are part of the project-specific Mask2Former implementation and should be considered when reproducing the experiments.

---

# Whole-Slide Image Processing

The project includes dependencies for processing histopathology whole-slide images (WSIs), including:

* OpenSlide
* FastSlide
* WSI-patching
* OpenCV
* Rasterio
* HDF5

These dependencies are included in:

```text
requirements.txt
```

They support the preprocessing and patch-based workflow used for the histopathology data.

---

# Reproducibility

The repository tracks the source code, model configuration, and environment specifications required to reproduce the computational pipeline.

### Main environment

Use:

```bash
pip install -r requirements.txt
```

### Complete environment snapshot

```text
requirements-full.txt
```

contains the installed Python packages, including transitive dependencies.

### Development environment snapshot

```text
requirements-lock.txt
```

contains the dependency snapshot used during development/testing.

These files are intended to make it easier to recreate the computational environment on another compatible system.

---

# Data and Model Weights

Patient data, whole-slide images, annotations, trained model weights, and generated training outputs are **not included in this repository**.

These files are excluded from version control where appropriate through `.gitignore`.

The repository therefore contains the code and configuration needed to reproduce the pipeline, while the underlying research data remain in their designated storage locations.

---

# Getting Started

For project-specific installation instructions:

**[INSTALL.md](INSTALL.md)**

For general Mask2Former usage:

**[GETTING_STARTED.md](GETTING_STARTED.md)**

For advanced Mask2Former usage:

**[ADVANCED_USAGE.md](ADVANCED_USAGE.md)**

For pretrained models and the original model zoo:

**[MODEL_ZOO.md](MODEL_ZOO.md)**

---

# Original Mask2Former

This project is based on the original **Mask2Former** implementation:

> Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, Rohit Girdhar.
> *Masked-attention Mask Transformer for Universal Image Segmentation.*
> CVPR 2022.

Original resources:

* [Mask2Former repository](https://github.com/facebookresearch/Mask2Former)
* [Paper](https://arxiv.org/abs/2112.01527)
* [Project page](https://bowenc0221.github.io/mask2former)

---

# Citation

If you use Mask2Former in this project, please cite the original paper:

```bibtex
@inproceedings{cheng2021mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and Alexander Kirillov and Rohit Girdhar},
  journal={CVPR},
  year={2022}
}
```

The original MaskFormer work can also be cited as:

```bibtex
@inproceedings{cheng2021maskformer,
  title={Per-Pixel Classification is Not All You Need for Semantic Segmentation},
  author={Bowen Cheng and Alexander G. Schwing and Alexander Kirillov},
  journal={NeurIPS},
  year={2021}
}
```

---

# License

The majority of the original Mask2Former code is distributed under the MIT License.

Some components of the original project are distributed under separate licenses. See:

```text
LICENSE
```

for details.

---

# Acknowledgements

This project is based on:

* [Mask2Former](https://github.com/facebookresearch/Mask2Former)
* [MaskFormer](https://github.com/facebookresearch/MaskFormer)
* [Detectron2](https://github.com/facebookresearch/detectron2)

We acknowledge the authors and developers of these projects for making their work available to the research community.
