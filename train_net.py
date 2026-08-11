import os
import sys
from pathlib import Path

import cv2
import torch
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_train_loader,
)
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import SemSegEvaluator

sys.path.insert(0, "/net/beegfs/users/P098169/Mask2Former")

from mask2former import add_maskformer2_config
from mask2former.data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

DATA_ROOT = Path("/net/beegfs/users/P098169/ADAMANT_Bone/Adama_train/extracted/")
OUTPUT_DIR = "/net/beegfs/users/P098169/ADAMANT_Bone/output"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


def load_dataset(data_root):
    dataset_dicts = []
    image_id = 0

    for slide_dir in sorted(data_root.iterdir()):

        if not slide_dir.is_dir():
            continue

        img_dir = slide_dir / "images"
        mask_dir = slide_dir / "masks"

        if not img_dir.exists():
            continue

        if not mask_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*.png")):

            mask_path = mask_dir / img_path.name

            if not mask_path.exists():
                continue

            img = cv2.imread(str(img_path))

            if img is None:
                continue

            h, w = img.shape[:2]

            dataset_dicts.append(
                {
                    "file_name": str(img_path),
                    "sem_seg_file_name": str(mask_path),
                    "height": h,
                    "width": w,
                    "image_id": image_id,
                }
            )

            image_id += 1

    print(f"Loaded {len(dataset_dicts)} training samples")

    return dataset_dicts


def register_datasets(data_root):

    if "wsi_train" in DatasetCatalog:
        DatasetCatalog.remove("wsi_train")

    DatasetCatalog.register(
        "wsi_train",
        lambda: load_dataset(data_root),
    )

    MetadataCatalog.get("wsi_train").set(
        stuff_classes=[
            "Background",      # 0
            "Bone",            # 1
            "Lamellair Bone",  # 2
        ],
        stuff_colors=[
            (0, 0, 0),         # Background
            (255, 0, 0),       # Bone
            (0, 255, 0),       # Lamellair Bone
        ],
        evaluator_type="sem_seg",
        ignore_label=255,
    )

    print("Dataset registered.")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


def setup_config():

    cfg = get_cfg()

    add_maskformer2_config(cfg)

    cfg.merge_from_file(
        "/net/beegfs/users/P098169/Mask2Former/configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml"
    )

    cfg.DATASETS.TRAIN = ("wsi_train",)
    cfg.DATASETS.TEST = ()

    # Number of semantic classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3

    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.INPUT.MASK_FORMAT = "bitmask"

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 1e-4
    cfg.SOLVER.MAX_ITER = 30000
    cfg.SOLVER.STEPS = (20000, 25000)
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = False
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"

    cfg.TEST.EVAL_PERIOD = 0

    cfg.OUTPUT_DIR = OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------


class WSITrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):

        mapper = MaskFormerSemanticDatasetMapper(
            cfg,
            is_train=True,
        )

        return build_detection_train_loader(
            cfg,
            mapper=mapper,
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):

        return SemSegEvaluator(
            dataset_name,
            distributed=False,
            output_dir=cfg.OUTPUT_DIR,
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():

    print("=" * 60)
    print("Mask2Former Semantic Segmentation Training")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    print("GPU :", torch.cuda.get_device_name(0))

    register_datasets(DATA_ROOT)

    cfg = setup_config()

    print()
    print("Training configuration")
    print("----------------------")
    print("Dataset :", DATA_ROOT)
    print("Output  :", cfg.OUTPUT_DIR)
    print("Classes :", cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)
    print("Batch   :", cfg.SOLVER.IMS_PER_BATCH)
    print("LR      :", cfg.SOLVER.BASE_LR)
    print("Iters   :", cfg.SOLVER.MAX_ITER)
    print()
    print("Class mapping")
    print("0 -> Background")
    print("1 -> Bone")
    print("2 -> Lamellair Bone")
    print()

    trainer = WSITrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print("Training completed.")


if __name__ == "__main__":
    main()