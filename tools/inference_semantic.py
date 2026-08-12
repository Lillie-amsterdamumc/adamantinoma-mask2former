import argparse
import os
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import warnings

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config

# This file lives at <repo_root>/tools/inference_semantic.py.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from mask2former import add_maskformer2_config

warnings.filterwarnings("ignore")


print("=" * 80)
print("Mask2Former Semantic Segmentation Inference - V10")
print("Merged Bone Classes: Bone + Lamellair Bone -> Bone")
print("=" * 80)


############################################################
# PATHS
############################################################

parser = argparse.ArgumentParser(
    description="Run two-class Mask2Former inference (background vs merged bone)."
)
parser.add_argument(
    "--config-file",
    default=os.environ.get(
        "WSI_DETECTRON2_CONFIG",
        str(REPO_ROOT / "configs" / "ade20k" / "semantic-segmentation" /
            "maskformer2_R50_bs16_160k.yaml"),
    ),
    help="Mask2Former YAML config path.",
)
parser.add_argument(
    "--weights",
    default=os.environ.get("WSI_INFERENCE_WEIGHTS", ""),
    required=not bool(os.environ.get("WSI_INFERENCE_WEIGHTS")),
    help="V10 model checkpoint path (or set WSI_INFERENCE_WEIGHTS).",
)
parser.add_argument(
    "--input-dir",
    default=os.environ.get("WSI_INFERENCE_INPUT", ""),
    required=not bool(os.environ.get("WSI_INFERENCE_INPUT")),
    help="Root containing case folders (or set WSI_INFERENCE_INPUT).",
)
parser.add_argument(
    "--output-dir",
    default=os.environ.get(
        "WSI_INFERENCE_OUTPUT",
        str(REPO_ROOT / "inference_outputs_v10_merged_bone"),
    ),
    help="Inference output root (or set WSI_INFERENCE_OUTPUT).",
)
args = parser.parse_args()

CONFIG_FILE = args.config_file
WEIGHTS = args.weights
INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir

os.makedirs(OUTPUT_DIR, exist_ok=True)


############################################################
# CLASS DEFINITIONS
############################################################

# IMPORTANT:
# V10 is a 2-class model:
#   0 = Background
#   1 = Bone
#
# Original raw masks may have:
#   0 = Background
#   1 = Bone
#   2 = Lamellair Bone
#
# But during V10 training:
#   Lamellair Bone (2) -> Bone (1)
#
# Therefore inference can ONLY produce:
#   0 = Background
#   1 = Bone

CLASS_NAMES = {
    0: "Background",
    1: "Bone",
}

CLASS_COLORS = {
    0: (0, 0, 0),        # Background
    1: (255, 0, 0),      # Bone
}

# Grayscale output
ID2GRAY = {
    0: 0,        # Background
    1: 255,      # Bone
}


############################################################
# CONFIG
############################################################

cfg = get_cfg()

add_deeplab_config(cfg)
add_maskformer2_config(cfg)

cfg.merge_from_file(CONFIG_FILE)

cfg.MODEL.WEIGHTS = WEIGHTS

# 2 classes: Background + Bone
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2

# Explicitly set device
cfg.MODEL.DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

cfg.freeze()


############################################################
# PRINT CONFIGURATION
############################################################

print()
print("Configuration")
print("-" * 80)

print("Number of classes:",
      cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)

print("Classes:")

for k, v in CLASS_NAMES.items():
    print(f"  {k}: {v}")

print("Device :", cfg.MODEL.DEVICE)
print("Weights:", WEIGHTS)
print("Input  :", INPUT_DIR)
print("Output :", OUTPUT_DIR)

print()


############################################################
# CHECK FILES
############################################################

if not os.path.isfile(WEIGHTS):
    raise FileNotFoundError(
        f"Model weights not found:\n{WEIGHTS}"
    )

if not os.path.isdir(INPUT_DIR):
    raise FileNotFoundError(
        f"Input directory not found:\n{INPUT_DIR}"
    )


############################################################
# LOAD MODEL
############################################################

print("Loading Mask2Former model...")

predictor = DefaultPredictor(cfg)

print("Model loaded successfully.")
print()


############################################################
# IMAGE EXTENSIONS
############################################################

VALID_EXT = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
)


############################################################
# FIND ALL SLIDE / CASE FOLDERS
############################################################

folders = sorted(
    f
    for f in os.listdir(INPUT_DIR)
    if os.path.isdir(os.path.join(INPUT_DIR, f))
)

print(f"Found {len(folders)} cases")
print()


############################################################
# INFERENCE
############################################################

total = 0

for folder in folders:

    print("=" * 80)
    print(f"Processing case: {folder}")
    print("=" * 80)

    # Patches live under:
    #
    # <slide>/patches/part_XXXX/*.png
    #
    case_input = os.path.join(
        INPUT_DIR,
        folder,
        "patches"
    )

    case_output = os.path.join(
        OUTPUT_DIR,
        folder
    )

    if not os.path.isdir(case_input):
        print(
            f"WARNING: no 'patches' folder for {folder}, skipping"
        )
        continue

    ########################################################
    # FIND ALL PATCH IMAGES
    ########################################################

    image_paths = []

    for root, _, files in os.walk(case_input):

        for f in files:

            if f.lower().endswith(VALID_EXT):

                image_paths.append(
                    os.path.join(root, f)
                )

    image_paths.sort()

    print(
        f"{folder}: {len(image_paths)} images"
    )

    if len(image_paths) == 0:
        continue


    ########################################################
    # PROCESS PATCHES
    ########################################################

    for img_path in image_paths:

        image = cv2.imread(img_path)

        if image is None:

            print(
                f"WARNING: could not read image: {img_path}"
            )

            continue


        ####################################################
        # MODEL INFERENCE
        ####################################################

        with torch.no_grad():

            outputs = predictor(image)


        ####################################################
        # SEMANTIC SEGMENTATION
        ####################################################

        sem_seg = (
            outputs["sem_seg"]
            .argmax(0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        ####################################################
        # SAFETY CHECK
        ####################################################

        unique_classes = np.unique(sem_seg)

        illegal_classes = unique_classes[
            ~np.isin(
                unique_classes,
                np.array([0, 1])
            )
        ]

        if len(illegal_classes) > 0:

            print(
                f"WARNING: unexpected class IDs "
                f"{illegal_classes.tolist()} "
                f"in {img_path}"
            )


        ####################################################
        # GRAYSCALE MASK
        ####################################################

        # Background = 0
        # Bone       = 255

        gray = np.zeros_like(
            sem_seg,
            dtype=np.uint8
        )

        for cid, value in ID2GRAY.items():

            gray[sem_seg == cid] = value


        ####################################################
        # COLOR MASK
        ####################################################

        color = np.zeros(
            (
                sem_seg.shape[0],
                sem_seg.shape[1],
                3
            ),
            dtype=np.uint8
        )

        for cid, rgb in CLASS_COLORS.items():

            color[sem_seg == cid] = rgb


        ####################################################
        # OVERLAY
        ####################################################

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        overlay = cv2.addWeighted(
            image_rgb,
            0.6,
            color,
            0.4,
            0
        )


        ####################################################
        # KEEP SAME RELATIVE STRUCTURE
        #
        # Example:
        #
        # patches/
        #   part_0000/
        #       patch_000001.png
        #
        # becomes:
        #
        # case/
        #   part_0000/
        #       patch_000001_mask.png
        #
        ####################################################

        relative_path = os.path.relpath(
            img_path,
            case_input
        )

        relative_dir = os.path.dirname(
            relative_path
        )

        save_dir = os.path.join(
            case_output,
            relative_dir
        )

        os.makedirs(
            save_dir,
            exist_ok=True
        )


        ####################################################
        # OUTPUT FILENAMES
        ####################################################

        base = os.path.splitext(
            os.path.basename(img_path)
        )[0]

        gray_path = os.path.join(
            save_dir,
            base + "_mask.png"
        )

        color_path = os.path.join(
            save_dir,
            base + "_color.png"
        )

        overlay_path = os.path.join(
            save_dir,
            base + "_overlay.png"
        )


        ####################################################
        # SAVE OUTPUTS
        ####################################################

        cv2.imwrite(
            gray_path,
            gray
        )

        cv2.imwrite(
            color_path,
            cv2.cvtColor(
                color,
                cv2.COLOR_RGB2BGR
            )
        )

        cv2.imwrite(
            overlay_path,
            cv2.cvtColor(
                overlay,
                cv2.COLOR_RGB2BGR
            )
        )


        total += 1

        if total % 100 == 0:

            print(
                f"Processed {total} images"
            )


    ########################################################
    # COPY coords.csv
    ########################################################

    # reconstruct_mask.py needs this file to recover
    # x / y coordinates for each patch.

    src_csv = os.path.join(
        INPUT_DIR,
        folder,
        "coords.csv"
    )

    dst_csv = os.path.join(
        case_output,
        "coords.csv"
    )

    if os.path.exists(src_csv):

        os.makedirs(
            case_output,
            exist_ok=True
        )

        shutil.copy(
            src_csv,
            dst_csv
        )

        print(
            f"Copied coords.csv -> {dst_csv}"
        )

    else:

        print(
            f"WARNING: coords.csv not found for "
            f"{folder} ({src_csv})"
        )


############################################################
# FINISHED
############################################################

print()
print("=" * 80)
print("Inference Finished")
print("=" * 80)

print("Total images:", total)
print("Output:", OUTPUT_DIR)

print()
print("Classes:")
print("  0 = Background")
print("  1 = Bone (Bone + Lamellair Bone)")

print("=" * 80)
