## Figuring out how patches map to full scenes

from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from datasets.test_dataset import TestPatchDataset, parse_patch_name
from utils.load_model import load_checkpoint
from utils.metrics import compute_metrics


def predict_mask(model, x, device):
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
    return pred.cpu().numpy()


def crop_gt(full_gt, patch_h, patch_w, row, col, mode):
    h, w = full_gt.shape

    if mode == "top_left_row_col":
        r0 = (row - 1) * patch_h
        c0 = (col - 1) * patch_w

    elif mode == "top_left_col_row":
        r0 = (col - 1) * patch_h
        c0 = (row - 1) * patch_w

    elif mode == "bottom_left_row_col":
        r0 = h - row * patch_h
        c0 = (col - 1) * patch_w

    elif mode == "bottom_left_col_row":
        r0 = h - col * patch_h
        c0 = (row - 1) * patch_w

    else:
        raise ValueError(f"Unknown crop mode: {mode}")

    r1 = r0 + patch_h
    c1 = c0 + patch_w

    if r0 < 0 or c0 < 0 or r1 > h or c1 > w:
        return None

    return full_gt[r0:r1, c0:c1]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = TestPatchDataset(
        r_dir=r"C:\Users\racha\Desktop\Dataset\38-Cloud_test\test_red",
        g_dir=r"C:\Users\racha\Desktop\Dataset\38-Cloud_test\test_green",
        b_dir=r"C:\Users\racha\Desktop\Dataset\38-Cloud_test\test_blue",
        nir_dir=r"C:\Users\racha\Desktop\Dataset\38-Cloud_test\test_nir",
        include_nir=True
    )

    idx = 3 ## choose a random index
    x, name = dataset[idx]
    x_batch = x.unsqueeze(0)

    patch_idx, row, col, scene_id = parse_patch_name(name)
    print("Patch file:", name)
    print("Patch idx:", patch_idx)
    print("Row:", row, "Col:", col)
    print("Scene ID:", scene_id)

    scene_gt_path = Path(
        r"C:\Users\racha\Desktop\Dataset\38-Cloud_test\Entire_scene_gts"
    ) / f"edited_corrected_gts_{scene_id}.TIF"

    full_gt = np.array(Image.open(scene_gt_path))
    full_gt = np.where(full_gt == 255, 1, 0).astype(np.uint8)

    print("Full GT shape:", full_gt.shape)
    print("Patch tensor shape:", x.shape)

    patch_h, patch_w = x.shape[1], x.shape[2]

    unet = load_checkpoint(
        model_path=r"C:\Users\racha\OneDrive\Desktop\cloud\cloud\models\Unet_20260413_004720\Unet.pth",
        model_type="unet",
        device=device
    )

    unetpp = load_checkpoint(
        model_path=r"C:\Users\racha\OneDrive\Desktop\cloud\cloud\models\UnetPlusPlus_20260413_150345\UnetPlusPlus.pth",
        model_type="unetplusplus",
        device=device
    )

    pred_unet = predict_mask(unet, x_batch, device)[0]
    pred_unetpp = predict_mask(unetpp, x_batch, device)[0]

    rgb = x[:3].cpu().numpy().transpose(1, 2, 0)

    modes = [
        "top_left_row_col",
        "top_left_col_row",
        "bottom_left_row_col",
        "bottom_left_col_row",
    ]

    gt_crops = {}
    for mode in modes:
        crop = crop_gt(full_gt, patch_h, patch_w, row, col, mode)
        gt_crops[mode] = crop

    plt.figure(figsize=(20, 10))

    plt.subplot(2, 4, 1)
    plt.imshow(rgb)
    plt.title("Input")
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.imshow(pred_unet, cmap="gray")
    plt.title("U-Net")
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.imshow(pred_unetpp, cmap="gray")
    plt.title("U-Net++")
    plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.imshow(full_gt, cmap="gray")
    plt.title("Full GT")
    plt.axis("off")

    for i, mode in enumerate(modes, start=5):
        plt.subplot(2, 4, i)
        crop = gt_crops[mode]
        if crop is None:
            plt.text(0.5, 0.5, "Out of bounds", ha="center", va="center")
            plt.title(mode)
            plt.axis("off")
            continue

        metrics = compute_metrics(pred_unet, crop)
        plt.imshow(crop, cmap="gray")
        plt.title(f"{mode}\nDice={metrics['dice']:.3f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()