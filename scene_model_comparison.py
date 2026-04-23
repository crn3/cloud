from pathlib import Path
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from metrics import compute_metrics


# =========================================================
# CONFIG
# =========================================================

# Root folder from predict_all_batches.py when SCENE_FILTER=None
# Example:
# patch_predictions/20260417_123456
PRED_ROOT = Path(r"C:\Users\racha\OneDrive\Desktop\cloud\cloud\patch_predictions\20260417_103818")

UNET_DIR = PRED_ROOT / "unet"
UNETPP_DIR = PRED_ROOT / "unetplusplus"

SCENE_GT_DIR = Path(r"C:\Users\racha\Desktop\Dataset\38-Cloud_test\Entire_scene_gts")
OUTPUT_DIR = Path(r"C:\Users\racha\OneDrive\Desktop\cloud\cloud\model_comparisons")

GRID_MODE = "row_col"
ORIGIN = "top_left"

## all scenes
SCENE_IDS = None

# Resize previews so they are manageable
PREVIEW_MAX_WIDTH = 1600


# =========================================================
# HELPERS
# =========================================================

def parse_patch_name(filename: str):
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

    return reconstructed, gt


def resize_for_preview(arr: np.ndarray, max_width: int):
    h, w = arr.shape
    if w <= max_width:
        return arr

    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)

    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    img = img.resize((new_w, new_h), Image.NEAREST)
    return (np.array(img) > 0).astype(np.uint8)


def save_comparison_preview(scene_id: str, gt: np.ndarray, unet_pred: np.ndarray, unetpp_pred: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_small = resize_for_preview(gt, PREVIEW_MAX_WIDTH)
    unet_small = resize_for_preview(unet_pred, PREVIEW_MAX_WIDTH)
    unetpp_small = resize_for_preview(unetpp_pred, PREVIEW_MAX_WIDTH)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(gt_small, cmap="gray")
    plt.title("Ground truth")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(unet_small, cmap="gray")
    plt.title("U-Net")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(unetpp_small, cmap="gray")
    plt.title("U-Net++")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / f"{scene_id}_comparison.png", dpi=200)
    plt.close()


def print_results_table(rows):
    headers = [
        "scene_id",
        "unet_dice", "unetpp_dice",
        "unet_iou", "unetpp_iou",
        "unet_prec", "unetpp_prec",
        "unet_rec", "unetpp_rec",
    ]

    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row[h])))

    def fmt_row(row_dict):
        return " | ".join(str(row_dict[h]).ljust(widths[h]) for h in headers)

    print("\n" + fmt_row({h: h for h in headers}))
    print("-+-".join("-" * widths[h] for h in headers))

    for row in rows:
        print(fmt_row(row))


# =========================================================
# MAIN
# =========================================================

def main():
    if not UNET_DIR.exists():
        raise FileNotFoundError(f"UNET_DIR does not exist: {UNET_DIR}")
    if not UNETPP_DIR.exists():
        raise FileNotFoundError(f"UNETPP_DIR does not exist: {UNETPP_DIR}")

    run_output_dir = OUTPUT_DIR / f"{PRED_ROOT.name}_{GRID_MODE}_{ORIGIN}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    unet_scene_files = group_patch_files_by_scene(UNET_DIR)
    unetpp_scene_files = group_patch_files_by_scene(UNETPP_DIR)

    scene_ids = sorted(set(unet_scene_files.keys()) & set(unetpp_scene_files.keys()))

    if SCENE_IDS is not None:
        scene_ids = [sid for sid in scene_ids if sid in SCENE_IDS]

    if not scene_ids:
        raise ValueError("No overlapping scenes found for U-Net and U-Net++.")

    rows_for_csv = []
    rows_for_print = []

    for i, scene_id in enumerate(scene_ids, start=1):
        print(f"\n[{i}/{len(scene_ids)}] Processing {scene_id}")

        unet_pred, gt = reconstruct_scene(scene_id, unet_scene_files[scene_id])
        unetpp_pred, gt2 = reconstruct_scene(scene_id, unetpp_scene_files[scene_id])

        if gt.shape != gt2.shape:
            raise ValueError(f"GT shape mismatch for {scene_id}")

        m_unet = compute_metrics(unet_pred, gt)
        m_unetpp = compute_metrics(unetpp_pred, gt)

        save_mask(unet_pred, run_output_dir / "reconstructed_unet" / f"pred_{scene_id}.TIF")
        save_mask(unetpp_pred, run_output_dir / "reconstructed_unetplusplus" / f"pred_{scene_id}.TIF")

        save_comparison_preview(
            scene_id=scene_id,
            gt=gt,
            unet_pred=unet_pred,
            unetpp_pred=unetpp_pred,
            out_dir=run_output_dir / "previews"
        )

        rows_for_csv.append({
            "scene_id": scene_id,
            "unet_accuracy": m_unet["accuracy"],
            "unet_precision": m_unet["precision"],
            "unet_recall": m_unet["recall"],
            "unet_iou": m_unet["iou"],
            "unet_dice": m_unet["dice"],
            "unetpp_accuracy": m_unetpp["accuracy"],
            "unetpp_precision": m_unetpp["precision"],
            "unetpp_recall": m_unetpp["recall"],
            "unetpp_iou": m_unetpp["iou"],
            "unetpp_dice": m_unetpp["dice"],
            "dice_diff_unetpp_minus_unet": m_unetpp["dice"] - m_unet["dice"],
            "iou_diff_unetpp_minus_unet": m_unetpp["iou"] - m_unet["iou"],
        })

        rows_for_print.append({
            "scene_id": scene_id,
            "unet_dice": f"{m_unet['dice']:.4f}",
            "unetpp_dice": f"{m_unetpp['dice']:.4f}",
            "unet_iou": f"{m_unet['iou']:.4f}",
            "unetpp_iou": f"{m_unetpp['iou']:.4f}",
            "unet_prec": f"{m_unet['precision']:.4f}",
            "unetpp_prec": f"{m_unetpp['precision']:.4f}",
            "unet_rec": f"{m_unet['recall']:.4f}",
            "unetpp_rec": f"{m_unetpp['recall']:.4f}",
        })

    print_results_table(rows_for_print)

    csv_path = run_output_dir / "model_comparison_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_for_csv[0].keys()))
        writer.writeheader()
        writer.writerows(rows_for_csv)

    mean_unet_accuracy = np.mean([r["unet_accuracy"] for r in rows_for_csv])
    mean_unet_precision = np.mean([r["unet_precision"] for r in rows_for_csv])
    mean_unet_recall = np.mean([r["unet_recall"] for r in rows_for_csv])
    mean_unet_iou = np.mean([r["unet_iou"] for r in rows_for_csv])
    mean_unet_dice = np.mean([r["unet_dice"] for r in rows_for_csv])

    mean_unetpp_accuracy = np.mean([r["unetpp_accuracy"] for r in rows_for_csv])
    mean_unetpp_precision = np.mean([r["unetpp_precision"] for r in rows_for_csv])
    mean_unetpp_recall = np.mean([r["unetpp_recall"] for r in rows_for_csv])
    mean_unetpp_iou = np.mean([r["unetpp_iou"] for r in rows_for_csv])
    mean_unetpp_dice = np.mean([r["unetpp_dice"] for r in rows_for_csv])

    print("\n===== Average metrics across selected scenes =====")
    print(f"{'Metric':<12} | {'U-Net':<10} | {'U-Net++':<10}")
    print("-" * 40)
    print(f"{'Accuracy':<12} | {mean_unet_accuracy:<10.4f} | {mean_unetpp_accuracy:<10.4f}")
    print(f"{'Precision':<12} | {mean_unet_precision:<10.4f} | {mean_unetpp_precision:<10.4f}")
    print(f"{'Recall':<12} | {mean_unet_recall:<10.4f} | {mean_unetpp_recall:<10.4f}")
    print(f"{'IoU':<12} | {mean_unet_iou:<10.4f} | {mean_unetpp_iou:<10.4f}")
    print(f"{'Dice':<12} | {mean_unet_dice:<10.4f} | {mean_unetpp_dice:<10.4f}")

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved previews in: {run_output_dir / 'previews'}")


if __name__ == "__main__":
    main()