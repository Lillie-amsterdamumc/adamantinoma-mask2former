# Adamantinoma Mask2Former

This repository contains a customized Mask2Former implementation for semantic segmentation of adamantinoma tissue and bone histopathology images.

The project is based on [Mask2Former](https://github.com/facebookresearch/Mask2Former) and has been adapted for the adamantinoma segmentation workflow.

---

## Environment

The project has been developed and tested on a Linux HPC environment with an NVIDIA H100 NVL GPU.

The tested software environment is:

| Component                     | Version      |
| ----------------------------- | ------------ |
| Python                        | 3.11.15      |
| PyTorch                       | 2.1.0+cu121  |
| TorchVision                   | 0.16.0+cu121 |
| CUDA                          | 12.1         |
| Detectron2                    | 0.6          |
| timm                          | 1.0.26       |
| MultiScaleDeformableAttention | 1.0          |
| OpenCV                        | 4.13.0       |
| OpenSlide                     | 1.4.3        |
| FastSlide                     | 0.5.2        |
| WSI-patching                  | 0.5.1        |
| NumPy                         | 1.26.4       |
| SciPy                         | 1.17.1       |
| h5py                          | 3.16.0       |
| Shapely                       | 2.1.2        |
| Rasterio                      | 1.4.4        |
| Rtree                         | 1.4.1        |

The complete tested Python environment is documented in:

* `requirements.txt` — main runtime dependencies
* `requirements-full.txt` — full package snapshot
* `requirements-lock.txt` — locked environment snapshot

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/Lillie-amsterdamumc/adamantinoma-mask2former.git
cd adamantinoma-mask2former
```

If you are working on the Amsterdam UMC/SURF HPC environment, use the Python environment available on the cluster.

---

## 2. Create and activate a Python 3.11 environment

For a standard virtual environment:

```bash
python3.11 -m venv detectron2_env
source detectron2_env/bin/activate
```

Verify:

```bash
python --version
```

Expected:

```text
Python 3.11.x
```

---

## 3. Install PyTorch

This project was tested with:

```text
PyTorch 2.1.0
TorchVision 0.16.0
CUDA 12.1
```

Install the corresponding PyTorch/TorchVision CUDA 12.1 builds before installing the remaining dependencies.

Verify the installation:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

On a GPU node, the expected output should include:

```text
PyTorch: 2.1.0+cu121
CUDA: 12.1
CUDA available: True
GPU count: 1
```

---

## 4. Install project dependencies

Install the main dependencies:

```bash
pip install -r requirements.txt
```

For the complete tested package environment:

```bash
pip install -r requirements-full.txt
```

`requirements-full.txt` is provided as a snapshot of the tested environment and may contain packages that are not directly required by the segmentation pipeline.

---

## 5. Detectron2

This project uses Detectron2 0.6.

The repository is configured to use the tested Detectron2 revision:

```text
b599f139756bd3646a26a909caf86a1a159e53a7
```

Verify:

```bash
python -c "import detectron2; print('Detectron2:', detectron2.__version__)"
```

Expected:

```text
Detectron2: 0.6
```

---

# MultiScaleDeformableAttention

Mask2Former requires the Multi-Scale Deformable Attention CUDA operation.

This project uses:

```text
MultiScaleDeformableAttention 1.0
```

Verify that it can be imported:

```bash
python -c "import MultiScaleDeformableAttention as m; print('MSDeformAttn:', m.__file__)"
```

Expected:

```text
MSDeformAttn: .../MultiScaleDeformableAttention.cpython-311-x86_64-linux-gnu.so
```

If the module has to be compiled on a new system, make sure that the CUDA toolkit is available and that `CUDA_HOME` points to the appropriate CUDA installation before compiling the extension.

---

# Verify the complete environment

Run:

```bash
python -c "
import torch
import torchvision
import detectron2
import MultiScaleDeformableAttention
import cv2
import h5py
import scipy
import shapely
import rasterio
import openslide
import fastslide
import wsi_patching
import timm

print('Python: OK')
print('PyTorch:', torch.__version__)
print('TorchVision:', torchvision.__version__)
print('CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))

print('Detectron2:', detectron2.__version__)
print('MSDeformAttn: OK')
print('OpenCV:', cv2.__version__)
print('h5py:', h5py.__version__)
print('SciPy:', scipy.__version__)
print('Shapely:', shapely.__version__)
print('Rasterio:', rasterio.__version__)
print('OpenSlide:', openslide.__version__)
print('FastSlide: OK')
print('WSI-patching: OK')
print('timm:', timm.__version__)
"
```

A successfully configured environment should report all dependencies as available and, on a GPU node, show:

```text
CUDA available: True
GPU: NVIDIA H100 NVL MIG 2g.24gb
```

---

# Project Structure

```text
adamantinoma-mask2former/
│
├── configs/
│   └── ade20k/
│       └── semantic-segmentation/
│
├── mask2former/
│   ├── modeling/
│   └── ...
│
├── datasets/
│
├── demo/
│
├── mask2former_video/
│
├── tools/
│
├── predict.py
├── train_net.py
│
├── requirements.txt
├── requirements-full.txt
├── requirements-lock.txt
│
├── INSTALL.md
├── GETTING_STARTED.md
└── README.md
```

---

# Training

Training is performed using `train_net.py` and the appropriate Mask2Former configuration.

Example:

```bash
python train_net.py \
    --config-file configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml
```

Additional options can be passed using Detectron2's standard command-line configuration overrides.

For HPC/Slurm execution, use a Slurm job script requesting the required GPU, CPU, memory, and wall time.

---

# GPU / Slurm

The project has been tested on an NVIDIA H100 NVL MIG partition:

```text
NVIDIA H100 NVL MIG 2g.24gb
```

Check the allocated GPU with:

```bash
nvidia-smi
```

and:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

# Reproducibility

The repository includes environment specifications to make the computational environment reproducible.

### `requirements.txt`

Contains the primary dependencies required by the project.

### `requirements-full.txt`

Contains the complete package snapshot from the tested environment, including transitive dependencies.

### `requirements-lock.txt`

Contains the locked package snapshot used for the tested environment.

The tested environment should be preferred when reproducing experiments, particularly because CUDA, PyTorch, Detectron2, and compiled CUDA extensions must remain compatible.

---

# Important Notes

The original Mask2Former repository contains installation instructions for older software versions, including Python 3.8, PyTorch 1.9, and CUDA 11.1.

**Those versions are not the tested environment for this repository.**

For this project, use the versions documented above.

In particular:

```text
Python      3.11
PyTorch     2.1.0+cu121
TorchVision 0.16.0+cu121
CUDA        12.1
Detectron2  0.6
```

Do not downgrade to the old example environment unless reproducing the original Mask2Former implementation specifically.

---

# Original Mask2Former

This repository is based on the original Mask2Former implementation:

https://github.com/facebookresearch/Mask2Former

Original project:

> Cheng, B., Misra, I., Schwing, A. G., Kirillov, A., & He, K.
> Masked-attention Mask Transformer for Universal Image Segmentation.

Please refer to the original repository for the general Mask2Former architecture and methodology.

---

# License

This project follows the licensing terms of the original Mask2Former repository unless otherwise specified.

