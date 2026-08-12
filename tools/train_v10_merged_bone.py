"""
Ablation run v10: merge bone + lamellair_bone into a single "bone" class.
=============================================================================

Why this file exists
---------------------
Two hypotheses have been ruled out or are still being tested:
  - Augmentation destroying lamellar pixels: RULED OUT (augmentation_diagnostic.py
    showed a lossless pixel permutation across 15 samples).
  - Oversampling / class-weight imbalance: being tested in
    train_v9_oversampling_ablation.py.

This script tests a third, independent hypothesis: that "bone" vs
"lamellair_bone" is not actually a clean, separable distinction in this
dataset -- the boundary between the two bone subtypes may be inherently
fuzzy/subjective in the annotations, so the model has to guess at a
boundary that isn't consistently defined, and that's what destabilizes
training regardless of augmentation or sampling.

To test this: collapse both foreground classes into ONE "bone" class and
train/evaluate a strictly 2-class model (background vs bone). If this
2-class model trains smoothly with a stable, high foreground F1 -- while
the 3-class runs kept collapsing on the fine-grained split -- that's strong
evidence the instability lives in the bone/lamellair boundary itself, not
in augmentation or sampling.

What changed vs the v8 baseline (kept identical otherwise, so this is a
single clean comparison: 3-class vs 2-class):
  - NUM_CLASSES: 3 -> 2
  - CLASS_NAMES: ["background","bone","lamellair_bone"] -> ["background","bone"]
  - _read_mask() now remaps raw label 2 (lamellair_bone) to 1 (bone) at
    read time, so every downstream consumer (dataset stats, evaluator,
    mapper, oversampling) sees a merged 2-class problem automatically.
  - LAMELLAIR_CLASS_WEIGHT removed; BONE_CLASS_WEIGHT now covers both
    original foreground classes combined.
  - Augmentation settings (geometric on, photometric off) and oversampling
    (off) are left exactly as in the v8 baseline, so any difference in
    training behavior can be attributed to the class merge, not to some
    other setting changing at the same time.

What changed vs the previous v10 draft:
  - Added a one-time, safe on-disk mask normalization step
    (normalize_mask_channels_on_disk) that converts any 3-channel mask
    PNGs under TRAIN_ROOT/masks and VAL_ROOT/masks to single-channel,
    after taking a one-time backup of each masks/ folder. This runs
    automatically at the start of main() when NORMALIZE_MASKS_ON_DISK is
    True. _read_mask() already defends against 3-channel masks at read
    time, so this step is a belt-and-suspenders cleanup, not a
    correctness requirement -- but it makes every future read faster and
    removes the ambiguity for good. Set NORMALIZE_MASKS_ON_DISK = False
    once you've confirmed the masks/ folders are already single-channel.

Location in this repository
----------------------------
This script lives at tools/train_v10_merged_bone.py, next to the other
project-specific training entry points, and is run from the repository
root the same way as train_net.py. Because the mask2former package
already lives at the repo root, this script adds the repo root to
sys.path relative to its own location -- no absolute, machine-specific
path is required.

Configuration
-------------
No user-specific filesystem paths are hardcoded. Dataset locations and
the output directory are supplied via command-line arguments (with
environment-variable fallbacks for convenience in Slurm/cluster job
scripts). Run --help to see the full list:

    python tools/train_v10_merged_bone.py --help

Typical usage (run from the repository root):

    python tools/train_v10_merged_bone.py \
        --train-root /path/to/Adama_train/extracted \
        --val-root /path/to/Adam_validation/extracted \
        --output-dir /path/to/output_v10_merged_bone_classes

Or, via environment variables (handy for Slurm job scripts):

    export WSI_TRAIN_ROOT=/path/to/Adama_train/extracted
    export WSI_VAL_ROOT=/path/to/Adam_validation/extracted
    export WSI_OUTPUT_DIR=/path/to/output_v10_merged_bone_classes
    python tools/train_v10_merged_bone.py

The Detectron2/Mask2Former config file defaults to
configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml
relative to the repo root, matching the layout already used by
train_net.py. Override it with --detectron2-config if needed.
"""

import argparse
import csv
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances
from detectron2.data.build import trivial_batch_collator
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.samplers import InferenceSampler, RepeatFactorTrainingSampler
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm
from detectron2.utils.events import EventStorage

# This file lives at <repo_root>/tools/train_v10_merged_bone.py. The
# mask2former package lives at <repo_root>/mask2former, so we add the repo
# root to sys.path relative to this file -- no machine-specific path needed.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mask2former import add_maskformer2_config
from mask2former.data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

# =============================================================================
# Paths and experiment settings
#
# TRAIN_ROOT, VAL_ROOT, OUTPUT_DIR and DETECTRON2_CONFIG_PATH below are
# placeholders that get overwritten in main() from parsed CLI args /
# environment variables (see parse_args()). They exist as module-level
# names because several functions below (register_datasets,
# print_dataset_statistics, setup_config, etc.) read them directly.
# =============================================================================

TRAIN_ROOT = Path(os.environ.get("WSI_TRAIN_ROOT", ""))
VAL_ROOT = Path(os.environ.get("WSI_VAL_ROOT", ""))

# >>> MERGE CHANGE: new output dir, never overwrites v8/v9 results.
OUTPUT_DIR = os.environ.get(
    "WSI_OUTPUT_DIR", str(REPO_ROOT / "output_v10_merged_bone_classes")
)

DETECTRON2_CONFIG_PATH = os.environ.get(
    "WSI_DETECTRON2_CONFIG",
    str(REPO_ROOT / "configs" / "ade20k" / "semantic-segmentation" / "maskformer2_R50_bs16_160k.yaml"),
)

# >>> MERGE CHANGE: two classes instead of three.
NUM_CLASSES = 2
IGNORE_LABEL = 255
CLASS_NAMES = ["background", "bone"]
EVAL_PERIOD = 2000

# Raw mask files still contain label 2 (lamellair_bone) on disk. This flag is
# just documentation of intent; the actual remapping happens in _read_mask().
MERGE_BONE_CLASSES = True

# =============================================================================
# One-time on-disk mask normalization (3-channel -> single-channel)
# =============================================================================

NORMALIZE_MASKS_ON_DISK = True  # set False after you've run it once successfully
MASK_BACKUP_SUFFIX = "_backup"


def normalize_mask_channels_on_disk(*roots: Path) -> None:
    """Converts any 3-channel mask PNGs under the given roots to single-channel,
    in place, after making a one-time backup copy of each masks/ folder.

    Safe to call every run: if a backup already exists, it will NOT be
    overwritten, and files already single-channel are skipped untouched.

    Note: _read_mask() below already collapses 3-channel masks to a single
    channel at read time, so training is correct even without running this.
    This function just cleans up the files on disk once, so future reads
    are faster and there's no lingering ambiguity about mask shape.
    """
    for data_root in roots:
        mask_root = data_root / "masks"
        if not mask_root.exists():
            print(f"[normalize_masks] Skipping missing dir: {mask_root}")
            continue

        backup_root = mask_root.parent / f"masks{MASK_BACKUP_SUFFIX}"
        if not backup_root.exists():
            print(f"[normalize_masks] Backing up {mask_root} -> {backup_root}")
            shutil.copytree(mask_root, backup_root)
        else:
            print(f"[normalize_masks] Backup already exists, skipping: {backup_root}")

        files = list(mask_root.rglob("*.png"))
        print(f"[normalize_masks] {mask_root}: {len(files)} files found")

        converted = 0
        for i, f in enumerate(files):
            m = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if m is None:
                print(f"[normalize_masks] [WARNING] could not read: {f}")
                continue
            if m.ndim == 3:
                cv2.imwrite(str(f), m[:, :, 0])
                converted += 1
            if i % 200 == 0:
                print(f"[normalize_masks]   {i}/{len(files)} checked, {converted} converted so far")

        print(f"[normalize_masks] Done: {mask_root} -> {converted}/{len(files)} converted\n")


# =============================================================================
# Ablation switches -- IDENTICAL to the v8 baseline on purpose
# =============================================================================
ENABLE_PATCH_FOREGROUND_FILTER = False
ENABLE_PATCH_OVERSAMPLING = False
ENABLE_GEOMETRIC_AUGMENTATION = True
ENABLE_PHOTOMETRIC_AUGMENTATION = False

REPEAT_THRESHOLD = 0.15
MIN_FOREGROUND_FRACTION_TO_KEEP = 0.05

# Only one foreground class now, so only one set of oversampling thresholds.
BONE_MEDIUM_FRACTION = 0.02
BONE_HIGH_FRACTION = 0.10
BONE_MEDIUM_REPEAT = 2.0
BONE_HIGH_REPEAT = 3.0

HORIZONTAL_FLIP_PROB = 0.50
VERTICAL_FLIP_PROB = 0.50
ROTATION_90_PROB = 0.75
BRIGHTNESS_RANGE = (0.90, 1.10)
CONTRAST_RANGE = (0.90, 1.10)

# >>> MERGE CHANGE: single foreground weight (previously bone=2.0, lamellair=1.0
# were separate; the merged class keeps the higher of the two, matching how
# much emphasis the combined foreground region should get).
BACKGROUND_CLASS_WEIGHT = 0.20
BONE_CLASS_WEIGHT = 2.00

BEST_METRIC = "sem_seg/macro_f1_foreground"
PRECISION_GUARD_METRIC = "sem_seg/mean_precision_foreground"
MIN_FOREGROUND_PRECISION = 0.20
EARLY_STOPPING_PATIENCE = 4
EARLY_STOPPING_MIN_DELTA = 0.002
EARLY_STOPPING_START_ITER = 8000


# =============================================================================
# Dataset loading and validation
# =============================================================================

def _read_mask(mask_path: str | Path) -> np.ndarray:
    """Reads the raw mask AND merges lamellair_bone (2) into bone (1).

    This is the single point where the class merge happens. Every other
    function in this script (dataset stats, mapper, evaluator, oversampling)
    calls this function, so the merge propagates everywhere automatically.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = mask.astype(np.int64)

    if MERGE_BONE_CLASSES:
        mask[mask == 2] = 1  # lamellair_bone -> bone

    return mask


def load_dataset(data_root: Path) -> List[Dict]:
    dataset_dicts: List[Dict] = []
    img_root = data_root / "images"
    mask_root = data_root / "masks"

    if not img_root.exists():
        raise FileNotFoundError(img_root)
    if not mask_root.exists():
        raise FileNotFoundError(mask_root)

    image_files = sorted(img_root.rglob("*.png"))
    print(f"[{data_root.parent.name}] Found {len(image_files)} image patches")

    skipped = 0
    for idx, img_path in enumerate(image_files):
        relative_path = img_path.relative_to(img_root)
        mask_path = mask_root / relative_path

        if not mask_path.exists():
            print(f"[WARNING] Missing mask: {mask_path}")
            skipped += 1
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARNING] Could not read image: {img_path}")
            skipped += 1
            continue

        mask = _read_mask(mask_path)  # already merged
        h, w = image.shape[:2]
        if mask.shape[:2] != (h, w):
            print(
                f"[WARNING] Shape mismatch: image={image.shape[:2]}, "
                f"mask={mask.shape[:2]} for {img_path}"
            )
            skipped += 1
            continue

        valid_values = np.unique(mask)
        illegal_values = valid_values[
            ~np.isin(valid_values, np.array([0, 1, IGNORE_LABEL], dtype=np.int64))
        ]
        if illegal_values.size:
            raise ValueError(
                f"Mask {mask_path} contains invalid labels after merge: "
                f"{illegal_values.tolist()}. Expected only 0, 1, {IGNORE_LABEL}."
            )

        dataset_dicts.append(
            {
                "file_name": str(img_path),
                "sem_seg_file_name": str(mask_path),
                "height": h,
                "width": w,
                "image_id": len(dataset_dicts),
            }
        )

    print(
        f"[{data_root.parent.name}] Loaded {len(dataset_dicts)} image-mask pairs "
        f"(skipped={skipped})"
    )
    if not dataset_dicts:
        raise RuntimeError(f"No valid image-mask pairs found under {data_root}")
    return dataset_dicts


def register_datasets(train_root: Path, val_root: Path) -> None:
    for name in ("wsi_train", "wsi_val"):
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)

    DatasetCatalog.register("wsi_train", lambda: load_dataset(train_root))
    DatasetCatalog.register("wsi_val", lambda: load_dataset(val_root))

    for name in ("wsi_train", "wsi_val"):
        MetadataCatalog.get(name).set(
            stuff_classes=["Background", "Bone"],
            stuff_colors=[(0, 0, 0), (255, 0, 0)],
            evaluator_type="sem_seg",
            ignore_label=IGNORE_LABEL,
        )

    print("Datasets registered: wsi_train, wsi_val (bone + lamellair_bone merged)")


def print_dataset_statistics(dataset_name: str) -> None:
    records = DatasetCatalog.get(dataset_name)
    image_presence = np.zeros(NUM_CLASSES, dtype=np.int64)
    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for record in records:
        mask = _read_mask(record["sem_seg_file_name"])
        for class_id in range(NUM_CLASSES):
            count = int(np.sum(mask == class_id))
            pixel_counts[class_id] += count
            image_presence[class_id] += int(count > 0)

    valid_pixels = int(pixel_counts.sum())
    print(f"\nDataset statistics: {dataset_name} (merged classes)")
    for class_id, class_name in enumerate(CLASS_NAMES):
        image_fraction = image_presence[class_id] / max(len(records), 1)
        pixel_fraction = pixel_counts[class_id] / max(valid_pixels, 1)
        print(
            f"  {class_id}: {class_name:<10} "
            f"images={image_presence[class_id]:>6}/{len(records)} ({image_fraction:.4f}), "
            f"pixels={pixel_fraction:.6f}"
        )
    print()


# =============================================================================
# Patch-level foreground filtering and oversampling
# =============================================================================

def _patch_class_fractions(record: Dict) -> np.ndarray:
    mask = _read_mask(record["sem_seg_file_name"])
    valid = mask != IGNORE_LABEL
    denom = int(valid.sum())
    if denom == 0:
        return np.zeros(NUM_CLASSES, dtype=np.float64)
    return np.asarray(
        [np.count_nonzero((mask == class_id) & valid) / denom for class_id in range(NUM_CLASSES)],
        dtype=np.float64,
    )


def filter_low_foreground_patches(records: List[Dict]) -> List[Dict]:
    if not ENABLE_PATCH_FOREGROUND_FILTER or MIN_FOREGROUND_FRACTION_TO_KEEP <= 0:
        print("Patch foreground filtering: disabled (ENABLE_PATCH_FOREGROUND_FILTER=False)")
        return records

    kept, dropped = [], []
    for record in records:
        fractions = _patch_class_fractions(record)
        # Only one foreground class now (index 1).
        foreground_fraction = float(fractions[1])
        if foreground_fraction <= MIN_FOREGROUND_FRACTION_TO_KEEP:
            dropped.append(record)
        else:
            kept.append(record)

    print(
        "Patch foreground filtering: "
        f"kept={len(kept)}/{len(records)}, dropped={len(dropped)}, "
        f"threshold>{MIN_FOREGROUND_FRACTION_TO_KEEP:.3f}"
    )
    if not kept:
        raise RuntimeError(
            "Patch filtering removed every training patch. Lower "
            "MIN_FOREGROUND_FRACTION_TO_KEEP."
        )
    return kept


def compute_patch_repeat_factors(
    dataset_dicts: List[Dict], repeat_threshold: float
) -> torch.Tensor:
    n = len(dataset_dicts)
    if not ENABLE_PATCH_OVERSAMPLING:
        print("Patch-level repeat-factor sampling: disabled (ENABLE_PATCH_OVERSAMPLING=False)\n")
        return torch.ones(n, dtype=torch.float32)

    fractions = np.zeros((n, NUM_CLASSES), dtype=np.float64)
    presence = np.zeros((n, NUM_CLASSES), dtype=np.bool_)
    for i, record in enumerate(dataset_dicts):
        fractions[i] = _patch_class_fractions(record)
        presence[i] = fractions[i] > 0

    class_frequency = presence.mean(axis=0)
    class_repeat = np.ones(NUM_CLASSES, dtype=np.float64)
    if repeat_threshold > 0:
        for class_id in range(1, NUM_CLASSES):
            freq = float(class_frequency[class_id])
            if freq > 0:
                class_repeat[class_id] = max(1.0, math.sqrt(repeat_threshold / freq))

    repeats = np.ones(n, dtype=np.float32)
    for i in range(n):
        r = 1.0
        present_fg = np.flatnonzero(presence[i, 1:]) + 1
        if present_fg.size:
            r = max(r, float(np.max(class_repeat[present_fg])))

        bone_fraction = float(fractions[i, 1])
        if bone_fraction >= BONE_HIGH_FRACTION:
            r = max(r, BONE_HIGH_REPEAT)
        elif bone_fraction >= BONE_MEDIUM_FRACTION:
            r = max(r, BONE_MEDIUM_REPEAT)

        repeats[i] = r

    print("Patch-level repeat-factor sampling (single merged foreground class)")
    print(
        f"  Bone: >= {BONE_MEDIUM_FRACTION:.3f} -> {BONE_MEDIUM_REPEAT:.1f}x, "
        f">= {BONE_HIGH_FRACTION:.3f} -> {BONE_HIGH_REPEAT:.1f}x"
    )
    print(
        f"  repeat factors: min={repeats.min():.3f}, "
        f"mean={repeats.mean():.3f}, max={repeats.max():.3f}"
    )
    unique, counts = np.unique(repeats, return_counts=True)
    print("  distribution: " + ", ".join(f"{u:.2f}x={c}" for u, c in zip(unique, counts)))
    print()
    return torch.as_tensor(repeats, dtype=torch.float32)


# =============================================================================
# Training mapper -- unchanged logic, just runs against the merged mask
# =============================================================================

class AugmentedMaskFormerSemanticMapper:
    def __init__(self, cfg):
        min_sizes = cfg.INPUT.MIN_SIZE_TRAIN
        if not isinstance(min_sizes, (tuple, list)):
            min_sizes = (min_sizes,)

        aug_steps = [
            T.ResizeShortestEdge(
                short_edge_length=list(min_sizes),
                max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                sample_style=cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            ),
        ]
        if ENABLE_GEOMETRIC_AUGMENTATION:
            aug_steps.append(T.RandomFlip(prob=HORIZONTAL_FLIP_PROB, horizontal=True, vertical=False))
            aug_steps.append(T.RandomFlip(prob=VERTICAL_FLIP_PROB, horizontal=False, vertical=True))
        if ENABLE_PHOTOMETRIC_AUGMENTATION:
            aug_steps.append(T.RandomBrightness(*BRIGHTNESS_RANGE))
            aug_steps.append(T.RandomContrast(*CONTRAST_RANGE))
        self.augmentations = T.AugmentationList(aug_steps)

        self.ignore_label = IGNORE_LABEL
        self.size_divisibility = int(cfg.INPUT.SIZE_DIVISIBILITY)

    @staticmethod
    def _maybe_rotate_90(image: np.ndarray, sem_seg: np.ndarray):
        if not ENABLE_GEOMETRIC_AUGMENTATION:
            return image, sem_seg
        if np.random.random() >= ROTATION_90_PROB:
            return image, sem_seg
        k = int(np.random.randint(1, 4))
        image = np.ascontiguousarray(np.rot90(image, k=k, axes=(0, 1)))
        sem_seg = np.ascontiguousarray(np.rot90(sem_seg, k=k, axes=(0, 1)))
        return image, sem_seg

    def __call__(self, dataset_dict: Dict) -> Dict:
        dataset_dict = dataset_dict.copy()
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        utils.check_image_size(dataset_dict, image)

        sem_seg = _read_mask(dataset_dict["sem_seg_file_name"]).astype("double")
        aug_input = T.AugInput(image, sem_seg=sem_seg)
        self.augmentations(aug_input)
        image = aug_input.image
        sem_seg = aug_input.sem_seg.astype("int64")

        image, sem_seg = self._maybe_rotate_90(image, sem_seg)

        image_shape = image.shape[:2]
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        dataset_dict["sem_seg"] = torch.as_tensor(
            np.ascontiguousarray(sem_seg), dtype=torch.long
        )

        classes = np.unique(sem_seg)
        classes = classes[(classes != self.ignore_label) & (classes >= 0) & (classes < NUM_CLASSES)]
        instances = Instances(image_shape)
        instances.gt_classes = torch.as_tensor(classes.astype("int64"), dtype=torch.int64)
        if len(classes) == 0:
            instances.gt_masks = torch.zeros((0, image_shape[0], image_shape[1]), dtype=torch.bool)
        else:
            masks = np.stack([sem_seg == class_id for class_id in classes], axis=0)
            instances.gt_masks = BitMasks(
                torch.as_tensor(np.ascontiguousarray(masks), dtype=torch.bool)
            ).tensor
        dataset_dict["instances"] = instances
        return dataset_dict


# =============================================================================
# Evaluator -- generalized over CLASS_NAMES/NUM_CLASSES (works for 2 or 3
# classes without hardcoding "lamellair_bone" anywhere)
# =============================================================================

class SemSegAccuracyF1Evaluator(DatasetEvaluator):
    def __init__(self, dataset_name: str, num_classes: int, ignore_label: int):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        records = DatasetCatalog.get(dataset_name)
        self.input_file_to_gt_file = {
            record["file_name"]: record["sem_seg_file_name"] for record in records
        }
        self.conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        self.conf_matrix.fill(0)

    def process(self, inputs, outputs) -> None:
        for inp, out in zip(inputs, outputs):
            pred = (
                out["sem_seg"]
                .argmax(dim=0)
                .detach()
                .to("cpu")
                .numpy()
                .astype(np.int64)
            )
            gt = _read_mask(self.input_file_to_gt_file[inp["file_name"]])  # already merged

            valid = gt != self.ignore_label
            gt_flat = gt[valid]
            pred_flat = pred[valid]

            legal = (gt_flat >= 0) & (gt_flat < self.num_classes)
            gt_flat = gt_flat[legal]
            pred_flat = pred_flat[legal]

            encoded = self.num_classes * gt_flat + pred_flat
            bincount = np.bincount(
                encoded, minlength=self.num_classes * self.num_classes
            )
            self.conf_matrix += bincount.reshape(self.num_classes, self.num_classes)

    def evaluate(self):
        comm.synchronize()
        matrices = comm.gather(self.conf_matrix, dst=0)
        if not comm.is_main_process():
            return {}

        cm = np.sum(matrices, axis=0).astype(np.float64)
        total = cm.sum()
        overall_acc = float(np.trace(cm) / total) if total > 0 else 0.0

        results = {"pixel_accuracy": overall_acc}
        f1s, precisions, recalls = [], [], []

        for class_id, class_name in enumerate(CLASS_NAMES):
            tp = cm[class_id, class_id]
            fp = cm[:, class_id].sum() - tp
            fn = cm[class_id, :].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            results[f"f1_{class_name}"] = float(f1)
            results[f"recall_{class_name}"] = float(recall)
            results[f"precision_{class_name}"] = float(precision)
            f1s.append(float(f1))
            precisions.append(float(precision))
            recalls.append(float(recall))

        # Foreground = everything except class 0 (background). This works
        # whether there are 1 or 2 foreground classes, no hardcoded names.
        results["macro_f1"] = float(np.mean(f1s))
        results["macro_f1_foreground"] = float(np.mean(f1s[1:]))
        results["mean_precision_foreground"] = float(np.mean(precisions[1:]))
        results["mean_recall_foreground"] = float(np.mean(recalls[1:]))
        return {"sem_seg": results}


# =============================================================================
# Deterministic validation-loss loader
# =============================================================================

def build_val_loss_loader(cfg, mapper, dataset_name: str, batch_size: int):
    records = DatasetCatalog.get(dataset_name)
    dataset = DatasetFromList(records, copy=False)
    dataset = MapDataset(dataset, mapper)
    sampler = InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size=batch_size, drop_last=False
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


def build_deterministic_val_mapper(cfg):
    val_cfg = cfg.clone()
    val_cfg.defrost()
    val_cfg.INPUT.RANDOM_FLIP = "none"
    val_cfg.INPUT.CROP.ENABLED = False

    min_size_test = cfg.INPUT.MIN_SIZE_TEST
    if isinstance(min_size_test, (tuple, list)):
        test_size = min_size_test[0]
    else:
        test_size = min_size_test
    val_cfg.INPUT.MIN_SIZE_TRAIN = (int(test_size),)
    val_cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    val_cfg.INPUT.MAX_SIZE_TRAIN = int(cfg.INPUT.MAX_SIZE_TEST)
    val_cfg.freeze()
    return MaskFormerSemanticDatasetMapper(val_cfg, is_train=True)


class ValidationLossHook(HookBase):
    def __init__(self, cfg, mapper, dataset_name: str):
        self.cfg = cfg.clone()
        self.mapper = mapper
        self.dataset_name = dataset_name
        self.period = int(cfg.TEST.EVAL_PERIOD)
        self.batch_size = max(1, min(2, int(cfg.SOLVER.IMS_PER_BATCH)))

    def after_step(self) -> None:
        if self.period <= 0:
            return

        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if next_iter % self.period != 0 and not is_final:
            return

        loader = build_val_loss_loader(
            self.cfg, self.mapper, self.dataset_name, self.batch_size
        )

        local_sum = 0.0
        local_count = 0
        with torch.no_grad():
            for data in loader:
                loss_dict = self.trainer.model(data)
                total_loss = sum(loss_dict.values())
                if torch.isfinite(total_loss):
                    local_sum += float(total_loss.detach().cpu())
                    local_count += 1

        stats = torch.tensor(
            [local_sum, float(local_count)],
            dtype=torch.float64,
            device=torch.device(self.cfg.MODEL.DEVICE),
        )
        if comm.get_world_size() > 1:
            torch.distributed.all_reduce(stats)

        global_sum, global_count = stats.tolist()
        mean_loss = global_sum / global_count if global_count > 0 else float("nan")
        self.trainer.storage.put_scalar("validation_loss", mean_loss, smoothing_hint=False)

        if comm.is_main_process():
            print(
                f"[ValidationLossHook] iter={next_iter} val_loss={mean_loss:.4f} "
                f"(n_batches={int(global_count)})"
            )


# =============================================================================
# Validation progress CSV + plots -- columns generalized to CLASS_NAMES so
# this works whether there are 2 or 3 classes.
# =============================================================================

class ValidationProgressHook(HookBase):
    def __init__(self, cfg):
        self.period = int(cfg.TEST.EVAL_PERIOD)
        self.output_dir = Path(cfg.OUTPUT_DIR)
        self.csv_path = self.output_dir / "validation_progress.csv"
        self.plot_path = self.output_dir / "validation_progress.png"

        self.columns = ["iteration", "validation_loss", "sem_seg/pixel_accuracy",
                         "sem_seg/macro_f1", "sem_seg/macro_f1_foreground",
                         "sem_seg/mean_precision_foreground", "sem_seg/mean_recall_foreground"]
        for name in CLASS_NAMES:
            self.columns += [f"sem_seg/f1_{name}", f"sem_seg/precision_{name}", f"sem_seg/recall_{name}"]
        self.columns.append("lr")

    @staticmethod
    def _latest_value(latest, key: str) -> float:
        if key not in latest:
            return float("nan")
        value = latest[key]
        if isinstance(value, (tuple, list)):
            value = value[0]
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def _read_existing_rows(self):
        rows = []
        if not self.csv_path.exists():
            return rows
        try:
            with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if row.get("iteration"):
                        rows.append(row)
        except Exception as exc:
            print(f"[ValidationProgressHook] Could not read existing CSV: {exc}")
        return rows

    def _write_csv(self, row):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rows = self._read_existing_rows()

        iteration_text = str(row["iteration"])
        rows = [r for r in rows if r.get("iteration") != iteration_text]
        rows.append({key: row.get(key, float("nan")) for key in self.columns})
        rows.sort(key=lambda r: int(float(r["iteration"])))

        tmp_path = self.csv_path.with_suffix(".csv.tmp")
        with tmp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, self.csv_path)

    def _plot(self):
        rows = self._read_existing_rows()
        if not rows:
            return

        def values(key):
            result = []
            for row in rows:
                try:
                    result.append(float(row.get(key, "nan")))
                except (TypeError, ValueError):
                    result.append(float("nan"))
            return np.asarray(result, dtype=np.float64)

        iterations = values("iteration")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

        ax = axes[0, 0]
        ax.plot(iterations, values("sem_seg/macro_f1_foreground"), marker="o", label="Foreground macro F1")
        for name in CLASS_NAMES[1:]:
            ax.plot(iterations, values(f"sem_seg/f1_{name}"), marker="o", label=f"{name} F1")
        ax.set_title("Foreground F1 scores (merged classes)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("F1")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[0, 1]
        ax.plot(iterations, values("sem_seg/mean_precision_foreground"), marker="o", label="Mean foreground precision")
        ax.plot(iterations, values("sem_seg/mean_recall_foreground"), marker="o", label="Mean foreground recall")
        ax.plot(iterations, values("sem_seg/pixel_accuracy"), marker="o", label="Pixel accuracy", alpha=0.65)
        ax.axhline(MIN_FOREGROUND_PRECISION, linestyle="--", linewidth=1.2, label="Precision guard")
        ax.set_title("Foreground precision and recall")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1, 0]
        ax.plot(iterations, values("validation_loss"), marker="o", label="Validation loss")
        ax.set_title("Validation loss")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1, 1]
        ax.plot(iterations, values("lr"), marker="o", label="Learning rate")
        ax.set_title("Learning-rate schedule")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Learning rate")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.suptitle("Mask2Former validation progress (v10: bone classes merged)", fontsize=16)
        tmp_path = self.plot_path.with_suffix(".png.tmp")
        fig.savefig(tmp_path, dpi=160, format="png")
        plt.close(fig)
        os.replace(tmp_path, self.plot_path)

    def after_step(self) -> None:
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if self.period <= 0 or (next_iter % self.period != 0 and not is_final):
            return
        if not comm.is_main_process():
            return

        latest = self.trainer.storage.latest()
        row = {"iteration": next_iter}
        for key in self.columns[1:]:
            row[key] = self._latest_value(latest, key)

        self._write_csv(row)
        self._plot()
        print(f"[ValidationProgressHook] CSV updated: {self.csv_path}")
        print(f"[ValidationProgressHook] Plot updated: {self.plot_path}")


# =============================================================================
# Save best checkpoint + early stopping based on validation macro-F1
# =============================================================================

class BestMetricEarlyStoppingHook(HookBase):
    def __init__(
        self,
        cfg,
        metric_name: str,
        patience: int,
        min_delta: float,
        start_iter: int,
        guard_metric_name: str | None = None,
        min_guard_value: float | None = None,
    ):
        self.cfg = cfg.clone()
        self.metric_name = metric_name
        self.period = int(cfg.TEST.EVAL_PERIOD)
        self.patience = patience
        self.min_delta = min_delta
        self.start_iter = start_iter
        self.guard_metric_name = guard_metric_name
        self.min_guard_value = min_guard_value
        self.best_value = -float("inf")
        self.bad_evaluations = 0
        self.checkpointer = None

    def before_train(self) -> None:
        self.checkpointer = DetectionCheckpointer(
            self.trainer.model,
            self.cfg.OUTPUT_DIR,
            trainer=self.trainer,
        )

    def after_step(self) -> None:
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if self.period <= 0 or (next_iter % self.period != 0 and not is_final):
            return

        latest = self.trainer.storage.latest()
        if self.metric_name not in latest:
            if comm.is_main_process():
                print(
                    f"[BestMetricHook] Metric '{self.metric_name}' was not found at "
                    f"iteration {next_iter}; skipping."
                )
            return

        value = float(latest[self.metric_name][0])

        guard_ok = True
        guard_value = None
        if self.guard_metric_name is not None and self.min_guard_value is not None:
            if self.guard_metric_name not in latest:
                guard_ok = False
                if comm.is_main_process():
                    print(
                        f"[BestMetricHook] Guard metric '{self.guard_metric_name}' "
                        f"was not found at iteration {next_iter}; checkpoint not updated."
                    )
            else:
                guard_value = float(latest[self.guard_metric_name][0])
                guard_ok = guard_value >= self.min_guard_value

        improved = guard_ok and value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.bad_evaluations = 0
            if comm.is_main_process():
                self.checkpointer.save(
                    "model_best_foreground_f1",
                    iteration=self.trainer.iter,
                    best_metric=value,
                    metric_name=self.metric_name,
                    foreground_precision=guard_value,
                    precision_guard_metric=self.guard_metric_name,
                )
                print(
                    f"[BestMetricHook] New best {self.metric_name}={value:.6f} "
                    f"with {self.guard_metric_name}={guard_value:.6f} "
                    f"at iter={next_iter}; checkpoint saved."
                )
        elif next_iter >= self.start_iter:
            self.bad_evaluations += 1
            if comm.is_main_process():
                guard_text = (
                    f", {self.guard_metric_name}={guard_value:.6f}"
                    if guard_value is not None
                    else ""
                )
                print(
                    f"[EarlyStopping] No eligible improvement in {self.metric_name}: "
                    f"current={value:.6f}, best={self.best_value:.6f}{guard_text}, "
                    f"bad_evals={self.bad_evaluations}/{self.patience}"
                )

            if self.bad_evaluations >= self.patience:
                if comm.is_main_process():
                    print(
                        f"[EarlyStopping] Stopping at iter={next_iter}. "
                        f"Best {self.metric_name}={self.best_value:.6f}."
                    )
                self.trainer.stop_requested = True


# =============================================================================
# Configuration
# =============================================================================

def setup_config():
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    cfg.merge_from_file(DETECTRON2_CONFIG_PATH)

    cfg.DATASETS.TRAIN = ("wsi_train",)
    cfg.DATASETS.TEST = ("wsi_val",)

    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.INPUT.CROP.ENABLED = False

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 5e-5
    cfg.SOLVER.MAX_ITER = 30000
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_FACTOR = 0.001
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.STEPS = ()

    cfg.SOLVER.CHECKPOINT_PERIOD = EVAL_PERIOD
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

    cfg.TEST.EVAL_PERIOD = EVAL_PERIOD
    cfg.OUTPUT_DIR = OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.freeze()
    return cfg


# =============================================================================
# Trainer
# =============================================================================

class WSITrainer(DefaultTrainer):
    stop_requested = False

    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)

        criterion = getattr(model, "criterion", None)
        empty_weight = getattr(criterion, "empty_weight", None)
        if criterion is None or empty_weight is None:
            raise AttributeError(
                "Could not find model.criterion.empty_weight. "
                "This Mask2Former version may use a different criterion layout."
            )
        if empty_weight.numel() != NUM_CLASSES + 1:
            raise ValueError(
                f"Expected {NUM_CLASSES + 1} criterion weights, "
                f"but found {empty_weight.numel()}."
            )

        with torch.no_grad():
            semantic_weights = torch.tensor(
                [BACKGROUND_CLASS_WEIGHT, BONE_CLASS_WEIGHT],
                dtype=empty_weight.dtype,
                device=empty_weight.device,
            )
            criterion.empty_weight[:NUM_CLASSES].copy_(semantic_weights)

        if comm.is_main_process():
            print(
                "Mask2Former semantic class CE weights: "
                f"background={BACKGROUND_CLASS_WEIGHT}, "
                f"bone(merged)={BONE_CLASS_WEIGHT}, "
                f"no_object={float(criterion.empty_weight[-1]):.4f}"
            )
        return model

    def train(self):
        self.stop_requested = False
        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(self.start_iter, self.max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                    if self.stop_requested:
                        break
                self.iter += 1
            finally:
                self.after_train()
        return getattr(self, "_last_eval_results", None)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = AugmentedMaskFormerSemanticMapper(cfg)
        all_records = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        records = filter_low_foreground_patches(all_records)
        repeat_factors = compute_patch_repeat_factors(records, REPEAT_THRESHOLD)
        sampler = RepeatFactorTrainingSampler(repeat_factors)
        return build_detection_train_loader(
            cfg, dataset=records, mapper=mapper, sampler=sampler
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return SemSegAccuracyF1Evaluator(
            dataset_name=dataset_name,
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            ignore_label=IGNORE_LABEL,
        )

    def build_hooks(self):
        hooks = super().build_hooks()
        val_mapper = build_deterministic_val_mapper(self.cfg)
        hooks.insert(-1, ValidationLossHook(self.cfg, val_mapper, self.cfg.DATASETS.TEST[0]))
        hooks.insert(
            -1,
            BestMetricEarlyStoppingHook(
                cfg=self.cfg,
                metric_name=BEST_METRIC,
                patience=EARLY_STOPPING_PATIENCE,
                min_delta=EARLY_STOPPING_MIN_DELTA,
                start_iter=EARLY_STOPPING_START_ITER,
                guard_metric_name=PRECISION_GUARD_METRIC,
                min_guard_value=MIN_FOREGROUND_PRECISION,
            ),
        )
        hooks.insert(-1, ValidationProgressHook(self.cfg))
        return hooks


# =============================================================================
# CLI argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Mask2Former on the merged 2-class (background/bone) "
            "adamantinoma dataset (v10 ablation: bone + lamellair_bone merged)."
        ),
    )
    parser.add_argument(
        "--train-root",
        type=str,
        default=os.environ.get("WSI_TRAIN_ROOT", ""),
        required=not bool(os.environ.get("WSI_TRAIN_ROOT")),
        help=(
            "Path to the training data root, containing images/ and masks/ "
            "subfolders (e.g. .../Adama_train/extracted). "
            "Falls back to the WSI_TRAIN_ROOT environment variable."
        ),
    )
    parser.add_argument(
        "--val-root",
        type=str,
        default=os.environ.get("WSI_VAL_ROOT", ""),
        required=not bool(os.environ.get("WSI_VAL_ROOT")),
        help=(
            "Path to the validation data root, containing images/ and masks/ "
            "subfolders (e.g. .../Adam_validation/extracted). "
            "Falls back to the WSI_VAL_ROOT environment variable."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("WSI_OUTPUT_DIR", str(REPO_ROOT / "output_v10_merged_bone_classes")),
        help=(
            "Directory where checkpoints, logs, and validation plots are written. "
            "Falls back to the WSI_OUTPUT_DIR environment variable, then to "
            "<repo_root>/output_v10_merged_bone_classes."
        ),
    )
    parser.add_argument(
        "--detectron2-config",
        type=str,
        default=os.environ.get("WSI_DETECTRON2_CONFIG", DETECTRON2_CONFIG_PATH),
        help=(
            "Path to the base Detectron2/Mask2Former YAML config. Defaults to "
            "configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml "
            "relative to the repository root."
        ),
    )
    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument(
        "--normalize-masks-on-disk",
        dest="normalize_masks",
        action="store_true",
        default=True,
        help=(
            "Convert any 3-channel mask PNGs under train/val masks/ to "
            "single-channel in place (with a one-time backup). Enabled by "
            "default; safe to re-run. See normalize_mask_channels_on_disk()."
        ),
    )
    normalize_group.add_argument(
        "--no-normalize-masks-on-disk",
        dest="normalize_masks",
        action="store_false",
        help="Skip the on-disk mask normalization step entirely.",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    # Wire parsed CLI args / env vars into the module-level names that the
    # rest of the script (register_datasets, setup_config, etc.) reads
    # directly. This keeps the rest of the file unchanged while removing
    # every hardcoded, machine-specific path from the top of the module.
    global TRAIN_ROOT, VAL_ROOT, OUTPUT_DIR, DETECTRON2_CONFIG_PATH, NORMALIZE_MASKS_ON_DISK
    TRAIN_ROOT = Path(args.train_root)
    VAL_ROOT = Path(args.val_root)
    OUTPUT_DIR = args.output_dir
    DETECTRON2_CONFIG_PATH = args.detectron2_config
    NORMALIZE_MASKS_ON_DISK = args.normalize_masks

    print("=" * 72)
    print("Mask2Former Training v10 - MERGED BONE CLASSES (bone + lamellair_bone -> bone)")
    print("=" * 72)

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train root       : {TRAIN_ROOT}")
    print(f"Val root         : {VAL_ROOT}")
    print(f"Detectron2 config: {DETECTRON2_CONFIG_PATH}")

    if NORMALIZE_MASKS_ON_DISK:
        normalize_mask_channels_on_disk(TRAIN_ROOT, VAL_ROOT)

    register_datasets(TRAIN_ROOT, VAL_ROOT)
    print_dataset_statistics("wsi_train")
    print_dataset_statistics("wsi_val")

    cfg = setup_config()

    print("Training configuration")
    print("----------------------")
    print(f"Output           : {cfg.OUTPUT_DIR}")
    print(f"Classes          : {CLASS_NAMES} (merged from 3 -> 2, MERGE_BONE_CLASSES={MERGE_BONE_CLASSES})")
    print(
        "Ablation switches : "
        f"patch_filter={ENABLE_PATCH_FOREGROUND_FILTER}, "
        f"oversampling={ENABLE_PATCH_OVERSAMPLING}, "
        f"geometric_aug={ENABLE_GEOMETRIC_AUGMENTATION}, "
        f"photometric_aug={ENABLE_PHOTOMETRIC_AUGMENTATION} "
        "(all identical to v8 baseline, only the class merge changed)"
    )
    print(f"Class CE weights : [{BACKGROUND_CLASS_WEIGHT}, {BONE_CLASS_WEIGHT}]")
    print()

    trainer = WSITrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print("\nTraining DONE")
    print(f"Best checkpoint: {Path(cfg.OUTPUT_DIR) / 'model_best_foreground_f1.pth'}")
    print(f"Validation CSV: {Path(cfg.OUTPUT_DIR) / 'validation_progress.csv'}")
    print(f"Validation plot: {Path(cfg.OUTPUT_DIR) / 'validation_progress.png'}")


if __name__ == "__main__":
    main()
