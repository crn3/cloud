from pathlib import Path
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from metrics import compute_metrics


# =========================================================
# CONFIG
# =========================================================

# Point this at ONE model folder, for example:
# patch_predictions/20260417_123456/LC08_L1TP_029041_20160720_20170222_01_T1/unet
# or later:
# patch_predictions/20260417_123456/unet
PRED_PATCH_DIR = Path(
    r"C:\Users\racha\OneDrive\Desktop\cloud\cloud\patch_predictions\20260417_033744\LC08_L1TP_029041_20160720_20170222_01_T1\unetplusplus"
)

SCENE_GT_DIR = Path(r"C:\Users\racha\Desktop\Dataset\38-Cloud_test\Entire_scene_gts")

OUTPUT_DIR = Path(r"C:\Users\racha\OneDrive\Desktop\cloud\cloud\reconstructed_scenes")

GRID_MODE = "row_col"   
ORIGIN = "top_left"     

SAVE_PREVIEW_PNGS = True
COMPUTE_METRICS = True


###############
### HELPERS ###
###############

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


def interpret_grid(a: int, b: int):
    if GRID_MODE == "row_col":
        row, col = a, b
    elif GRID_MODE == "col_row":
        row, col = b, a
    else:
        raise ValueError(f"Invalid GRID_MODE: {GRID_MODE}")
    return row, col


def load_mask(mask_path: Path) -> np.ndarray:
    arr = np.array(Image.open(mask_path))
    return (arr > 0).astype(np.uint8)


def load_gt(scene_id: str) -> np.ndarray:
    gt_path = SCENE_GT_DIR / f"edited_corrected_gts_{scene_id}.TIF"
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


def reconstruct_scene(scene_id: str, patch_files: list[Path]):
    sample_patch = load_mask(patch_files[0])
    patch_h, patch_w = sample_patch.shape

    parsed = []
    max_row = 0
    max_col = 0

    for pf in patch_files:
        patch_idx, a, b, _ = parse_patch_name(pf.name)
        row, col = interpret_grid(a, b)
        parsed.append((pf, patch_idx, row, col))
        max_row = max(max_row, row)
        max_col = max(max_col, col)

    canvas_h = max_row * patch_h
    canvas_w = max_col * patch_w
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    for pf, patch_idx, row, col in parsed:
        patch = load_mask(pf)

        if ORIGIN == "top_left":
            r0 = (row - 1) * patch_h
        elif ORIGIN == "bottom_left":
            r0 = canvas_h - row * patch_h
        else:
            raise ValueError(f"Invalid ORIGIN: {ORIGIN}")

        c0 = (col - 1) * patch_w
        r1 = r0 + patch_h
        c1 = c0 + patch_w

        canvas[r0:r1, c0:c1] = patch

    gt = load_gt(scene_id)
    gt_h, gt_w = gt.shape

    reconstructed = canvas[:gt_h, :gt_w]

    return reconstructed, gt, {
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


def save_preview(scene_id: str, pred: np.ndarray, gt: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    diff = (pred != gt).astype(np.uint8)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(pred, cmap="gray")
    plt.title("Reconstructed prediction")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(diff, cmap="gray")
    plt.title("Difference map")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / f"{scene_id}_preview.png", dpi=200)
    plt.close()


# =========================================================
# MAIN
# =========================================================

def main():
    if not PRED_PATCH_DIR.exists():
        raise FileNotFoundError(f"PRED_PATCH_DIR does not exist: {PRED_PATCH_DIR}")

    model_name = PRED_PATCH_DIR.name
    parent_name = PRED_PATCH_DIR.parent.name

    run_output_dir = OUTPUT_DIR / f"{parent_name}_{model_name}_{GRID_MODE}_{ORIGIN}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    scene_to_files = group_patch_files_by_scene(PRED_PATCH_DIR)

    print(f"Found {len(scene_to_files)} scene(s) in {PRED_PATCH_DIR}")
    print(f"GRID_MODE={GRID_MODE}, ORIGIN={ORIGIN}")

    metrics_rows = []

    for i, (scene_id, patch_files) in enumerate(sorted(scene_to_files.items()), start=1):
        print(f"\n[{i}/{len(scene_to_files)}] Reconstructing {scene_id}")
        print(f"  patch count: {len(patch_files)}")

        reconstructed, gt, info = reconstruct_scene(scene_id, patch_files)

        print(f"  patch size: {info['patch_h']} x {info['patch_w']}")
        print(f"  grid size:  {info['max_row']} rows x {info['max_col']} cols")
        print(f"  canvas:     {info['canvas_h']} x {info['canvas_w']}")
        print(f"  gt shape:   {info['gt_h']} x {info['gt_w']}")

        out_tif = run_output_dir / f"pred_{scene_id}.TIF"
        save_mask(reconstructed, out_tif)
        print(f"  saved tif:  {out_tif}")

        if SAVE_PREVIEW_PNGS:
            save_preview(scene_id, reconstructed, gt, run_output_dir)
            print(f"  saved png:  {run_output_dir / f'{scene_id}_preview.png'}")

        if COMPUTE_METRICS:
            m = compute_metrics(reconstructed, gt)

            metrics_rows.append({
                "scene_id": scene_id,
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "iou": m["iou"],
                "dice": m["dice"],
                "tp": m["tp"],
                "tn": m["tn"],
                "fp": m["fp"],
                "fn": m["fn"],
                "num_patches": info["num_patches"],
                "patch_h": info["patch_h"],
                "patch_w": info["patch_w"],
                "max_row": info["max_row"],
                "max_col": info["max_col"],
                "gt_h": info["gt_h"],
                "gt_w": info["gt_w"],
            })

            print(
                f"  metrics: Dice={m['dice']:.4f}, IoU={m['iou']:.4f}, "
                f"Precision={m['precision']:.4f}, Recall={m['recall']:.4f}"
            )

    if COMPUTE_METRICS and metrics_rows:
        csv_path = run_output_dir / "scene_metrics.csv"

        fieldnames = list(metrics_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_rows)

        mean_accuracy = np.mean([r["accuracy"] for r in metrics_rows])
        mean_precision = np.mean([r["precision"] for r in metrics_rows])
        mean_recall = np.mean([r["recall"] for r in metrics_rows])
        mean_iou = np.mean([r["iou"] for r in metrics_rows])
        mean_dice = np.mean([r["dice"] for r in metrics_rows])

        print("\n===== Average metrics =====")
        print(f"Accuracy : {mean_accuracy:.4f}")
        print(f"Precision: {mean_precision:.4f}")
        print(f"Recall   : {mean_recall:.4f}")
        print(f"IoU      : {mean_iou:.4f}")
        print(f"Dice     : {mean_dice:.4f}")
        print(f"\nSaved CSV: {csv_path}")


if __name__ == "__main__":
    main()