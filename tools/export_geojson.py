#!/usr/bin/env python3
import argparse
import os
import cv2
import pandas as pd
import numpy as np
import json
import random
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from shapely import affinity
from rasterio import features

PATCH_SIZE = 256
TILE_BORDER = 2   # pixels to cut off each tile edge to avoid seam artifacts
MERGE_POLYGONS = True  # automatically merge polygons of same class across tiles (recommended)
SAMPLE_N = 100    # number of tiles to sample to detect classes
MORPH_KERNEL = 3  # kernel size for morphological opening (set 0 to disable)
MIN_AREA = 10     # minimum polygon area in pixels (filters out tiny artifacts)
SIMPLIFY_TOLERANCE = 1.0  # polygon simplification tolerance

# ---- REAL CLASS VALUES (from inference_semantic.py's ID2GRAY / CLASS_NAMES) ----
#
# This script previously listed three classes (Background=0, Bone=127,
# Lamellair Bone=255) left over from the old 3-class model. The v10 model
# used here was trained with bone + lamellair_bone MERGED
# into a single "Bone" class, so its inference output only ever contains
# two gray values: 0 (background) and 255 (bone). Tile validity is tracked
# with a separate boolean array, so 255 is safe as a real class value.
KNOWN_CLASS_NAMES = {
    0: "Background",
    255: "Bone",
}

KNOWN_CLASS_COLORS_RGB = {
    0: (0, 0, 0),        # Background
    255: (255, 0, 0),    # Bone
}

# ---- CLASS FILTERING ----
# If non-empty, these class IDs will be REMOVED. Background (0) is excluded
# by default since it isn't a real tissue class.
EXCLUDE_CLASS_IDS = {0}

# Optional: keep ONLY these classes (set to None to disable)
INCLUDE_CLASS_IDS = None   # e.g. {60, 240}


# --- helpers to generate QuPath colors ---
def rgb_to_qupath_color(r, g, b):
    """Convert RGB (0-255) to QuPath's signed integer color format"""
    unsigned = (255 << 24) | (r << 16) | (g << 8) | b
    if unsigned & (1 << 31):
        signed = unsigned - (1 << 32)
    else:
        signed = unsigned
    return int(signed)


def qupath_color_to_rgb(color):
    """Convert QuPath's signed color to RGB tuple"""
    unsigned = color & 0xFFFFFFFF
    r = (unsigned >> 16) & 0xFF
    g = (unsigned >> 8) & 0xFF
    b = unsigned & 0xFF
    return (r, g, b)


def distinct_colors_for_n(n):
    """Return n visually distinct RGB tuples (0..255) using evenly spaced hues"""
    colors = []
    for i in range(n):
        h = float(i) / max(n, 1)
        s = 0.9
        v = 0.95
        i_h = int(h * 6)
        f = (h * 6) - i_h
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        i_mod = i_h % 6
        if i_mod == 0:
            r_, g_, b_ = v, t, p
        elif i_mod == 1:
            r_, g_, b_ = q, v, p
        elif i_mod == 2:
            r_, g_, b_ = p, v, t
        elif i_mod == 3:
            r_, g_, b_ = p, q, v
        elif i_mod == 4:
            r_, g_, b_ = t, p, v
        else:
            r_, g_, b_ = v, p, q
        colors.append((int(r_ * 255), int(g_ * 255), int(b_ * 255)))
    return colors


# --- mask -> polygons (per tile) ---
def mask_to_polygons(mask, tile_border=TILE_BORDER, simplify_tolerance=SIMPLIFY_TOLERANCE,
                      min_area=MIN_AREA, morph_kernel=MORPH_KERNEL):
    """
    Convert a labeled mask (2D numpy array) into a list of shapely polygons with class ids.

    NOTE ON "border invalidity": tile border pixels are marked invalid using an
    explicit boolean mask (`valid`), NOT by overwriting them with a magic
    sentinel pixel value. This is the robust version of the earlier
    approach -- with a boolean mask, a real class can validly use ANY gray
    value (including 255) with zero risk of collision, because "is this
    pixel inside the usable tile area" and "what class did the model predict
    here" are tracked as two completely separate arrays instead of being
    overloaded onto one.
    """
    polygons = []
    m = mask.copy()

    # `valid` marks pixels that are eligible to become polygons. Border
    # pixels are excluded here (as booleans), not by mutating class values.
    valid = np.ones(m.shape, dtype=bool)
    if tile_border > 0:
        valid[:tile_border, :] = False
        valid[-tile_border:, :] = False
        valid[:, :tile_border] = False
        valid[:, -tile_border:] = False

    unique_vals = np.unique(m[valid])
    if unique_vals.size == 0:
        return []

    if morph_kernel and morph_kernel > 0:
        k = np.ones((morph_kernel, morph_kernel), np.uint8)
    else:
        k = None

    for val in unique_vals:
        binary_mask = ((m == val) & valid).astype('uint8')
        if k is not None:
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, k)

        shapes = features.shapes(binary_mask, mask=(binary_mask == 1))
        for geom, v in shapes:
            if v != 1:
                continue
            poly = shape(geom)

            if poly.area < min_area:
                continue

            if simplify_tolerance and simplify_tolerance > 0:
                try:
                    poly = poly.simplify(simplify_tolerance, preserve_topology=True)
                except Exception:
                    pass

            if not (poly.is_valid and not poly.is_empty):
                continue

            polygons.append({
                "geometry": poly,
                "class_id": int(val)
            })
    return polygons


def find_mask_files(folder_path):
    """
    Recursively find every *_mask.png under folder_path (masks live in
    nested part_XXXX/ subfolders, not directly under folder_path).
    """
    mask_paths = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith("_mask.png"):
                mask_paths.append(os.path.join(root, f))
    return mask_paths


def sample_classes_from_folder(folder_path, sample_n=SAMPLE_N):
    """
    Sample up to sample_n mask PNGs (searched recursively) to detect which
    label values are present. Excludes the real background class (0).

    NOT called anywhere in the current pipeline (reconstruct_to_geojson now
    detects classes from the actual generated polygons instead -- see the
    note in that function). Kept as a standalone utility. If you do call
    this directly, background pixels with value 0 are excluded.
    """
    mask_files = find_mask_files(folder_path)
    if len(mask_files) == 0:
        return set()

    sample_files = random.sample(mask_files, min(len(mask_files), sample_n))
    found = set()
    for fpath in sample_files:
        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        unique = np.unique(img)
        for u in unique:
            if int(u) != 0:
                found.add(int(u))
    return found


def build_name_and_color_maps(class_ids):
    """
    Given an iterable of numeric class_ids, return CLASS_NAMES / CLASS_COLORS
    (as QuPath signed ints). Known classes (from run_inference.py) get their
    real name/color; any unexpected value falls back to a generic distinct one.
    """
    sorted_ids = sorted(list(class_ids))
    names = {}
    colors = {}

    unknown_ids = [cid for cid in sorted_ids if cid not in KNOWN_CLASS_NAMES]
    unknown_colors_rgb = distinct_colors_for_n(len(unknown_ids))
    unknown_color_iter = iter(unknown_colors_rgb)

    for cid in sorted_ids:
        if cid in KNOWN_CLASS_NAMES:
            names[cid] = KNOWN_CLASS_NAMES[cid]
            r, g, b = KNOWN_CLASS_COLORS_RGB[cid]
        else:
            names[cid] = f"Class_{cid}"
            r, g, b = next(unknown_color_iter)
        colors[cid] = rgb_to_qupath_color(r, g, b)

    return names, colors


def class_is_allowed(class_id, include_ids=None, exclude_ids=None):
    if include_ids is not None:
        return class_id in include_ids
    if exclude_ids:
        return class_id not in exclude_ids
    return True


def reconstruct_to_geojson(folder_path,
                            simplify_tolerance=SIMPLIFY_TOLERANCE,
                            merge_polygons=MERGE_POLYGONS,
                            min_area=MIN_AREA,
                            tile_border=TILE_BORDER,
                            morph_kernel=MORPH_KERNEL,
                            sample_n=SAMPLE_N):
    folder_name = os.path.basename(folder_path)

    # coords.csv is copied into inference_outputs/<slide>/ by run_inference.py
    csv_path = os.path.join(folder_path, "coords.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"coords.csv not found in {folder_name}: expected {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"index", "filename", "x", "y"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV columns missing. Required: {required_cols}. Found: {df.columns.tolist()}")

    # NOTE: classes are named/colored based on what's ACTUALLY found in the
    # processed polygons below, not on an early random sample -- a sample
    # can miss rare classes (e.g. a class that only appears in a handful
    # of tiles), which previously caused a KeyError when reporting.

    all_polygons = []
    missing = 0
    processed = 0

    print(f"[run] Processing {folder_name} ({len(df)} tiles listed in CSV)...")

    for row_idx, row in df.iterrows():
        try:
            x = int(row["x"])
            y = int(row["y"])
        except Exception as e:
            print(f"[error] reading CSV row {row_idx}: {e}")
            continue

        # filename in coords.csv is e.g. "part_0000/0000001.png"; the mask
        # sits at the same relative path with a "_mask.png" suffix
        rel_stem, _ = os.path.splitext(str(row["filename"]))
        fname = f"{rel_stem}_mask.png"
        fpath = os.path.join(folder_path, fname)

        if not os.path.exists(fpath):
            missing += 1
            continue

        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            missing += 1
            continue

        try:
            tile_polygons = mask_to_polygons(
                img,
                tile_border=tile_border,
                simplify_tolerance=simplify_tolerance,
                min_area=min_area,
                morph_kernel=morph_kernel
            )
        except Exception as e:
            print(f"[error] extracting polygons from {fname}: {e}")
            print(f"   Image shape: {img.shape}, dtype: {img.dtype}")
            print(f"   Unique values: {np.unique(img)}")
            raise

        for poly_data in tile_polygons:
            geom = poly_data["geometry"]
            class_id = poly_data["class_id"]
            # tile's x,y is its top-left corner in level0/canvas coordinates
            translated = affinity.translate(geom, xoff=x, yoff=y)
            if translated.area < min_area or translated.is_empty:
                continue

            if class_is_allowed(class_id, include_ids=INCLUDE_CLASS_IDS, exclude_ids=EXCLUDE_CLASS_IDS):
                all_polygons.append({
                    "geometry": translated,
                    "class_id": class_id
                })

        processed += 1
        if processed % 200 == 0:
            print(f"  Processed {processed}/{len(df)} tiles...")

    print(f"[ok] Processed {processed} tiles, {missing} missing")
    print(f"  Extracted {len(all_polygons)} polygons (before optional merge)")

    if merge_polygons and len(all_polygons) > 0:
        print("[run] Merging polygons by class (this can take time for many polygons)...")
        merged = []
        class_groups = {}
        for p in all_polygons:
            cid = p["class_id"]
            class_groups.setdefault(cid, []).append(p["geometry"])

        for cid, geoms in class_groups.items():
            try:
                combined = unary_union(geoms)
            except Exception as e:
                print(f"[warn] error during unary_union for class {cid}: {e}. Falling back to incremental union.")
                combined = geoms[0]
                for g in geoms[1:]:
                    combined = combined.union(g)

            if combined.geom_type == "Polygon":
                merged.append({"geometry": combined, "class_id": cid})
            elif combined.geom_type == "MultiPolygon":
                for poly in combined.geoms:
                    merged.append({"geometry": poly, "class_id": cid})
            else:
                try:
                    for poly in combined.geoms:
                        if poly.geom_type in ("Polygon", "MultiPolygon"):
                            merged.append({"geometry": poly, "class_id": cid})
                except Exception:
                    pass

        all_polygons = merged
        print(f"  Merged to {len(all_polygons)} polygons")

    features_out = []
    unique_class_ids = sorted({p["class_id"] for p in all_polygons})
    names_map, colors_map = build_name_and_color_maps(unique_class_ids)

    if len(unique_class_ids) > 0:
        print(f"[info] Detected classes: {unique_class_ids}")
        for cid in unique_class_ids:
            print(f"   -> {names_map.get(cid, f'Class_{cid}')} (id={cid}, color={qupath_color_to_rgb(colors_map[cid])})")

    for i, poly_data in enumerate(all_polygons):
        cid = poly_data["class_id"]
        cname = names_map.get(cid, f"Class_{cid}")
        ccolor = colors_map.get(cid, rgb_to_qupath_color(255, 0, 0))
        feat = {
            "type": "Feature",
            "id": f"feature_{i}",
            "geometry": mapping(poly_data["geometry"]),
            "properties": {
                "classification": {
                    "name": cname,
                    "colorRGB": ccolor
                },
                "objectType": "annotation",
                "name": cname,
                "measurements": []
            }
        }
        features_out.append(feat)

    geojson = {
        "type": "FeatureCollection",
        "features": features_out
    }

    out_path = os.path.join(folder_path, f"{folder_name}_mask.geojson")
    with open(out_path, "w") as f:
        json.dump(geojson, f)

    print(f"[done] Saved {len(features_out)} features to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert V10 merged-bone inference masks to QuPath GeoJSON."
    )
    parser.add_argument(
        "--root-dir",
        default=os.environ.get("WSI_INFERENCE_OUTPUT", ""),
        required=not bool(os.environ.get("WSI_INFERENCE_OUTPUT")),
        help=(
            "Inference output root created by inference_semantic.py "
            "(or set WSI_INFERENCE_OUTPUT)."
        ),
    )
    args = parser.parse_args()
    root_dir = args.root_dir

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Inference output directory not found: {root_dir}")

    folders = [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    print(f"Found {len(folders)} folders")
    for folder in folders:
        print("=" * 60)
        print("Processing:", folder)
        try:
            reconstruct_to_geojson(folder)
        except Exception as e:
            import traceback
            print("[ERROR]", e)
            print(traceback.format_exc())
        print()


if __name__ == "__main__":
    main()
