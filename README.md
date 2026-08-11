# Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation (CVPR 2022)

> **Adamantinoma Segmentation Project**
>
> This repository is a customized version of [Mask2Former](https://github.com/facebookresearch/Mask2Former) developed for semantic segmentation of adamantinoma histopathology images.
>
> The project has been tested on an NVIDIA H100 NVL GPU using:
>
> * Python 3.11.15
> * PyTorch 2.1.0 + CUDA 12.1
> * TorchVision 0.16.0 + CUDA 12.1
> * Detectron2 0.6
> * MultiScaleDeformableAttention 1.0
> * OpenSlide 1.4.3
> * FastSlide 0.5.2
> * WSI-patching 0.5.1
>
> The project-specific environment is defined in `requirements.txt`.
>
> For a complete snapshot of the installed Python environment, see:
>
> * `requirements-full.txt`
> * `requirements-lock.txt`

## Installation

### Requirements

* Linux
* Python 3.11.15
* NVIDIA GPU with a compatible CUDA environment
* PyTorch 2.1.0
* TorchVision 0.16.0
* Detectron2
* CUDA toolkit with `CUDA_HOME` configured

Create and activate a Python environment, then install the project dependencies:

```bash
pip install -r requirements.txt
```

For the complete environment snapshot:

```bash
pip install -r requirements-full.txt
```

### Detectron2

This project uses a specific Detectron2 commit:

```text
b599f139756bd3646a26a909caf86a1a159e53a7
```

The corresponding dependency is already specified in `requirements.txt`.

### CUDA kernel for MSDeformAttn

Mask2Former requires the CUDA implementation of Multi-Scale Deformable Attention.

Make sure that `CUDA_HOME` points to the installed CUDA toolkit:

```bash
echo $CUDA_HOME
```

Then compile the CUDA extension:

```bash
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

After successful compilation, return to the project root:

```bash
cd ../../../../
```

You can verify that the extension was built by checking the generated files under:

```text
mask2former/modeling/pixel_decoder/ops/
```

## Repository Structure

Important project files and directories:

```text
.
├── configs/
├── datasets/
├── demo/
├── mask2former/
├── mask2former_video/
├── tools/
├── train_net.py
├── predict.py
├── requirements.txt
├── requirements-full.txt
├── requirements-lock.txt
├── INSTALL.md
├── GETTING_STARTED.md
└── README.md
```

### Environment files

`requirements.txt` contains the main project dependencies required to run the customized Mask2Former pipeline.

`requirements-full.txt` contains a more complete snapshot of the Python environment, including transitive dependencies.

`requirements-lock.txt` contains the environment snapshot used during development and testing.

## Getting Started

For the original Mask2Former dataset preparation instructions, see:

```text
datasets/README.md
```

For general Mask2Former usage:

```text
GETTING_STARTED.md
```

For advanced usage:

```text
ADVANCED_USAGE.md
```

## Adamantinoma Segmentation

This repository contains modifications to the original Mask2Former implementation for semantic segmentation of adamantinoma histopathology images.

The customized pipeline includes project-specific:

* training configuration
* semantic segmentation classes
* matching/loss configuration
* training code
* whole-slide-image processing dependencies

The training configuration can be found under:

```text
configs/
```

The main training entry point is:

```text
train_net.py
```

## Reproducibility

The repository tracks the code and configuration required to reproduce the project environment.

Generated files, model weights, datasets, and training outputs are excluded through `.gitignore`.

The main environment specification is:

```text
requirements.txt
```

For a complete dependency snapshot:

```text
requirements-full.txt
requirements-lock.txt
```

## Original Mask2Former

This project is based on the original Mask2Former implementation by:

* Bowen Cheng
* Ishan Misra
* Alexander G. Schwing
* Alexander Kirillov
* Rohit Girdhar

Mask2Former provides a unified architecture for:

* Panoptic segmentation
* Instance segmentation
* Semantic segmentation

The original project supports datasets including ADE20K, Cityscapes, COCO and Mapillary Vistas.

Original resources:

* [Paper](https://arxiv.org/abs/2112.01527)
* [Project page](https://bowenc0221.github.io/mask2former)
* [Original repository](https://github.com/facebookresearch/Mask2Former)

## Model Zoo

The original Mask2Former model zoo and pretrained models are described in:

```text
MODEL_ZOO.md
```

## License

The majority of Mask2Former is licensed under the MIT License.

However, portions of the project are distributed under separate licenses:

* Swin-Transformer-Semantic-Segmentation — MIT License
* Deformable-DETR — Apache-2.0 License

See `LICENSE` for details.

## Citation

If you use Mask2Former in your research, please cite:

```bibtex
@inproceedings{cheng2021mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and Alexander Kirillov and Rohit Girdhar},
  journal={CVPR},
  year={2022}
}
```

If you find the MaskFormer code useful, please also consider:

```bibtex
@inproceedings{cheng2021maskformer,
  title={Per-Pixel Classification is Not All You Need for Semantic Segmentation},
  author={Bowen Cheng and Alexander G. Schwing and Alexander Kirillov},
  journal={NeurIPS},
  year={2021}
}
```

## Acknowledgement

This project is largely based on [MaskFormer](https://github.com/facebookresearch/MaskFormer) and [Mask2Former](https://github.com/facebookresearch/Mask2Former).
