Adamantinoma Mask2Former

Semantic segmentation of adamantinoma histopathology images using Mask2Former

This repository contains a project-specific implementation of Mask2Former for semantic segmentation of adamantinoma histopathology images. It includes the original framework together with a custom two-class training, inference, and GeoJSON export workflow for whole-slide-image (WSI) patches.

Current experiment: V10 merged bone classes

The V10 experiment treats the original bone and lamellair_bone annotations as one foreground class:

Model ID

Class

Source annotation labels

0

Background

0

1

Bone

1 (bone) and 2 (lamellair_bone)

Raw label 2 is remapped to label 1 in memory during both training and validation. The source annotation files are not modified during normal training.

The V10 workflow provides:

Two-class Mask2Former training

Runtime merging of the two original bone labels

Geometric training augmentation

Foreground-aware evaluation metrics

Validation-loss tracking

Best-checkpoint saving and early stopping based on foreground F1

Patch-level filtering and oversampling switches for ablation studies

Patch inference with grayscale masks, colour masks, and overlays

Conversion of predicted masks to QuPath-compatible GeoJSON

Tested environment

The project has been tested on an NVIDIA H100 NVL GPU with the following environment:

Component

Version

Python

3.11.15

PyTorch

2.1.0 + CUDA 12.1

TorchVision

0.16.0 + CUDA 12.1

Detectron2

Commit b599f139756bd3646a26a909caf86a1a159e53a7

MultiScaleDeformableAttention

1.0

OpenSlide

1.4.3

FastSlide

0.5.2

WSI-patching

0.5.1

timm

1.0.26

NumPy

1.26.4

SciPy

1.17.1

Installation

1. Clone the repository

git clone https://github.com/Lillie-amsterdamumc/adamantinoma-mask2former.git
cd adamantinoma-mask2former

2. Create a Python environment

conda create -n adamantinoma-mask2former python=3.11 -y
conda activate adamantinoma-mask2former

3. Install dependencies

pip install -r requirements.txt

For complete environment snapshots, see requirements-full.txt and requirements-lock.txt.

4. Build Multi-Scale Deformable Attention

Mask2Former requires the CUDA implementation of Multi-Scale Deformable Attention. Confirm that the CUDA toolkit is available:

echo $CUDA_HOME

Compile the extension from the repository root:

cd mask2former/modeling/pixel_decoder/ops
sh make.sh
cd ../../../../

See INSTALL.md for additional installation and CUDA troubleshooting information.

Project structure

adamantinoma-mask2former/
├── configs/
│   └── ade20k/
│       └── semantic-segmentation/
├── datasets/
├── demo/
├── mask2former/
│   └── modeling/
│       └── pixel_decoder/
│           └── ops/
├── tools/
│   ├── train_v10_merged_bone.py
│   ├── inference_semantic.py
│   └── export_geojson.py
├── train_net.py
├── requirements.txt
├── requirements-full.txt
├── requirements-lock.txt
├── INSTALL.md
├── GETTING_STARTED.md
├── ADVANCED_USAGE.md
├── MODEL_ZOO.md
└── README.md

Expected data layout

Training and validation roots must each contain matching image and mask paths:

<data-root>/
├── images/
│   ├── part_0000/
│   │   └── 0000001.png
│   └── ...
└── masks/
    ├── part_0000/
    │   └── 0000001.png
    └── ...

Masks may be stored as single-channel PNGs or as three-channel PNGs whose first channel contains the class IDs. During reading, the V10 training code converts a three-channel mask to one channel in memory and remaps raw label 2 to 1.

V10 training

Run the project-specific trainer from the repository root:

python tools/train_v10_merged_bone.py \
    --train-root /path/to/train/extracted \
    --val-root /path/to/validation/extracted \
    --output-dir /path/to/output_v10_merged_bone_classes

The paths can alternatively be supplied through environment variables:

export WSI_TRAIN_ROOT=/path/to/train/extracted
export WSI_VAL_ROOT=/path/to/validation/extracted
export WSI_OUTPUT_DIR=/path/to/output_v10_merged_bone_classes
python tools/train_v10_merged_bone.py

The Detectron2 configuration defaults to:

configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml

Use --detectron2-config or WSI_DETECTRON2_CONFIG to override it.

Training outputs

Output

Purpose

model_best_foreground_f1.pth

Best eligible checkpoint based on validation foreground F1

validation_progress.csv

Validation loss and segmentation metrics by evaluation step

validation_progress.png

Training-progress plots

Periodic checkpoints and logs

Standard Detectron2 training outputs

The default evaluation period is 2,000 iterations. The trainer records pixel accuracy, per-class precision/recall/F1, foreground precision/recall/F1, and validation loss.

--normalize-masks-on-disk is optional and disabled by default. Normal training reads and converts masks in memory without changing the source dataset.

Inference

The inference script expects case folders containing patches:

<input-root>/
└── <case-name>/
    ├── coords.csv
    └── patches/
        ├── part_0000/
        │   └── 0000001.png
        └── ...

Run inference from the repository root:

python tools/inference_semantic.py \
    --weights /path/to/model_best_foreground_f1.pth \
    --input-dir /path/to/inference_input/extracted \
    --output-dir /path/to/inference_outputs_v10_merged_bone

Optional arguments and environment-variable equivalents:

Argument

Environment variable

Purpose

--config-file

WSI_DETECTRON2_CONFIG

Mask2Former YAML configuration

--weights

WSI_INFERENCE_WEIGHTS

Trained V10 checkpoint

--input-dir

WSI_INFERENCE_INPUT

Root containing case folders

--output-dir

WSI_INFERENCE_OUTPUT

Inference output root

For every patch, the script writes:

*_mask.png: grayscale class mask (0 = background, 255 = bone)

*_color.png: RGB class visualization

*_overlay.png: prediction overlay on the input image

It preserves the nested patch structure and copies each case's coords.csv into the corresponding output folder.

GeoJSON export

Convert inference masks into QuPath-compatible annotations:

python tools/export_geojson.py \
    --root-dir /path/to/inference_outputs_v10_merged_bone

Alternatively:

export WSI_INFERENCE_OUTPUT=/path/to/inference_outputs_v10_merged_bone
python tools/export_geojson.py

The exporter:

Recursively finds *_mask.png files

Uses coords.csv to place each patch in slide coordinates

Excludes background from exported annotations

Removes small artifacts and optionally applies morphological opening

Optionally merges polygons belonging to the same class

Writes <case-name>_mask.geojson inside each case output folder

The inference and export scripts use the same grayscale mapping: 0 for background and 255 for merged bone.

Pipeline overview

Histopathology WSI
        │
        ▼
Patch extraction + coordinates
        │
        ▼
Annotated training patches
        │
        ▼
V10 two-class Mask2Former training
        │
        ▼
Patch-level semantic inference
        │
        ▼
Grayscale masks + visualizations
        │
        ▼
QuPath-compatible GeoJSON annotations

Original Mask2Former entry point

The repository retains the original train_net.py interface for standard Mask2Former configurations:

python train_net.py \
    --config-file configs/ade20k/semantic-segmentation/<CONFIG_FILE>.yaml

For the current adamantinoma merged-bone experiment, use tools/train_v10_merged_bone.py instead.

Reproducibility and data

The repository tracks source code, configuration, and environment specifications. Patient data, WSIs, annotations, trained model weights, and generated outputs are not included and should remain excluded through .gitignore.

Original Mask2Former

This project is based on:

Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention Mask Transformer for Universal Image Segmentation. CVPR 2022.

Mask2Former repository

Paper

Project page

Citation

@inproceedings{cheng2021mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and Alexander Kirillov and Rohit Girdhar},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}

The original MaskFormer work can also be cited:

@inproceedings{cheng2021maskformer,
  title={Per-Pixel Classification is Not All You Need for Semantic Segmentation},
  author={Bowen Cheng and Alexander G. Schwing and Alexander Kirillov},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}

License

Most of the original Mask2Former code is distributed under the MIT License. Some components use separate licenses; see LICENSE for details.

Acknowledgements

This project builds on:

Mask2Former

MaskFormer

Detectron2

We acknowledge the authors and developers of these projects for making their work publicly available.
