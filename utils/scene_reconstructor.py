## helper file
## constructs whole scene from patches

from pathlib import Path
import numpy as np
from PIL import Image


def parse_patch_name(filename: str):
    """
    Expected:
    patch_287_14_by_14_LC08_L1TP_029041_20160720_20170222_01_T1.TIF

    Returns:
    patch_idx, a, b, scene_id
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) < 6 or parts[0] != "patch":
        raise ValueError(f"Unexpected patch filename format: {filename}")

    patch_idx = int(parts[1])
    a = int(parts[2])

    if parts[3] != "by":
        raise ValueError(f"Unexpected patch filename format: {filename}")

    b = int(parts[4])
    scene_id = "_".join(parts[5:])

    return patch_idx, a, b, scene_id


def interpret_grid(a: int, b: int, grid_mode: str):
    if grid_mode == "row_col":
        row, col = a, b
    elif grid_mode == "col_row":
        row, col = b, a
    else:
        raise ValueError(f"Invalid GRID_MODE: {grid_mode}")
    return row, col


def load_mask(mask_path: Path) -> np.ndarray:
    arr = np.array(Image.open(mask_path))
    return (arr > 0).astype(np.uint8)


def load_gt(scene_id: str, scene_gt_dir: Path) -> np.ndarray:
    gt_path = scene_gt_dir / f"edited_corrected_gts_{scene_id}.TIF"
    gt = np.array(Image.open(gt_path))
    return gt.astype(np.uint8)


def save_mask(mask: np.ndarray, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    img.save(save_path)


def group_patch_files_by_scene(pred_patch_dir: Path):
    scene_to_files = {}

    patch_files = sorted(pred_patch_dir.glob("*.TIF"))
    if not patch_files:
        patch_files = sorted(pred_patch_dir.glob("*.tif"))

    if not patch_files:
        raise FileNotFoundError(f"No TIFF patch predictions found in: {pred_patch_dir}")

    for patch_file in patch_files:
        _, _, _, scene_id = parse_patch_name(patch_file.name)
        scene_to_files.setdefault(scene_id, []).append(patch_file)

    return scene_to_files


def reconstruct_scene(
    scene_id: str,
    patch_files: list[Path],
    scene_gt_dir: Path,
    grid_mode: str = "row_col",
    origin: str = "top_left",
):
    sample_patch = load_mask(patch_files[0])
    patch_h, patch_w = sample_patch.shape

    parsed = []
    max_row = 0
    max_col = 0

    for pf in patch_files:
        patch_idx, a, b, _ = parse_patch_name(pf.name)
        row, col = interpret_grid(a, b, grid_mode)
        parsed.append((pf, patch_idx, row, col))
        max_row = max(max_row, row)
        max_col = max(max_col, col)

    canvas_h = max_row * patch_h
    canvas_w = max_col * patch_w
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    for pf, patch_idx, row, col in parsed:
        patch = load_mask(pf)

        if origin == "top_left":
            r0 = (row - 1) * patch_h
        elif origin == "bottom_left":
            r0 = canvas_h - row * patch_h
        else:
            raise ValueError(f"Invalid ORIGIN: {origin}")

        c0 = (col - 1) * patch_w
        r1 = r0 + patch_h
        c1 = c0 + patch_w

        canvas[r0:r1, c0:c1] = patch

    gt = load_gt(scene_id, scene_gt_dir)
    gt_h, gt_w = gt.shape
    reconstructed = canvas[:gt_h, :gt_w]

    info = {
        "patch_h": patch_h,
        "patch_w": patch_w,
        "max_row": max_row,
        "max_col": max_col,
        "canvas_h": canvas_h,
        "canvas_w": canvas_w,
        "gt_h": gt_h,
        "gt_w": gt_w,
        "num_patches": len(patch_files),
    }

    return reconstructed, gt, info