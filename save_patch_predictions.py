from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from PIL import Image

from datasets.test_dataset import TestPatchDataset
from load_model import load_checkpoint


# =========================================================
# CONFIG
# =========================================================

# To predict just one scene:
# SCENE_FILTER = "LC08_L1TP_029041_20160720_20170222_01_T1"

# no filter, the whole set
SCENE_FILTER = None

# Optionally limit number of selected patches after filtering.
# Use None for all selected patches.
MAX_PATCHES = None

BATCH_SIZE = 16


###############
### HELPERS ###
############### 

def patch_id_from_filename(filename: str) -> str:
    name = Path(filename).name
    if name.startswith("red_"):
        return name[len("red_"):]
    return name


def predict_batch(model, x_batch, device):
    x_batch = x_batch.to(device)

    with torch.no_grad():
        logits = model(x_batch)            # [B, 2, H, W]
        pred = torch.argmax(logits, dim=1) # [B, H, W]

    return pred.cpu().numpy().astype(np.uint8)


def save_mask_tif(mask: np.ndarray, save_path: Path):
    mask_to_save = (mask * 255).astype(np.uint8)
    img = Image.fromarray(mask_to_save, mode="L")
    img.save(save_path)


def filename_matches_scene(filename: str, scene_id: str) -> bool:
    return scene_id in filename


def build_subset_indices(dataset, scene_filter=None, max_patches=None):
    indices = []

    for i, red_file in enumerate(dataset.red_files):
        name = red_file.name

        if scene_filter is not None and not filename_matches_scene(name, scene_filter):
            continue

        indices.append(i)

    if max_patches is not None:
        indices = indices[:max_patches]

    return indices


# =========================================================
# MAIN
# =========================================================

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

    selected_indices = build_subset_indices(
        dataset,
        scene_filter=SCENE_FILTER,
        max_patches=MAX_PATCHES
    )

    if not selected_indices:
        raise ValueError("No patches matched the requested filter.")

    subset = Subset(dataset, selected_indices)

    dataloader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

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

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root_out = Path("patch_predictions") / timestamp

    if SCENE_FILTER is not None:
        root_out = root_out / SCENE_FILTER

    unet_dir = root_out / "unet"
    unetpp_dir = root_out / "unetplusplus"

    unet_dir.mkdir(parents=True, exist_ok=True)
    unetpp_dir.mkdir(parents=True, exist_ok=True)

    total = len(subset)
    done = 0

    print(f"Saving predictions to: {root_out}")
    print(f"Selected patches: {total}")
    print(f"Scene filter: {SCENE_FILTER}")
    print(f"Max patches: {MAX_PATCHES}")

    for batch_idx, (x_batch, names) in enumerate(dataloader, start=1):
        pred_unet = predict_batch(unet, x_batch, device)
        pred_unetpp = predict_batch(unetpp, x_batch, device)

        for i, name in enumerate(names):
            patch_id = patch_id_from_filename(name)
            patch_id_path = Path(patch_id)
            out_name = patch_id_path.stem + ".TIF"

            save_mask_tif(pred_unet[i], unet_dir / out_name)
            save_mask_tif(pred_unetpp[i], unetpp_dir / out_name)

        done += len(names)
        print(f"[{batch_idx}/{len(dataloader)}] Saved {done}/{total} patches")

    print("\nFinished saving selected patch predictions.")
    print("U-Net output folder:   ", unet_dir)
    print("U-Net++ output folder: ", unetpp_dir)


if __name__ == "__main__":
    main()