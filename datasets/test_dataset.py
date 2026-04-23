from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class TestPatchDataset(Dataset):
    def __init__(self, r_dir, g_dir, b_dir, nir_dir=None, include_nir=True):
        self.r_dir = Path(r_dir)
        self.g_dir = Path(g_dir)
        self.b_dir = Path(b_dir)
        self.nir_dir = Path(nir_dir) if nir_dir is not None else None
        self.include_nir = include_nir

        self.red_files = sorted([f for f in self.r_dir.iterdir() if f.is_file()])

    def __len__(self):
        return len(self.red_files)

    def __getitem__(self, idx):
        r_file = self.red_files[idx]
        name = r_file.name

        g_file = self.g_dir / name.replace("red", "green")
        b_file = self.b_dir / name.replace("red", "blue")

        r = np.array(Image.open(r_file))
        g = np.array(Image.open(g_file))
        b = np.array(Image.open(b_file))

        rgb = np.stack([r, g, b], axis=2)

        if self.include_nir and self.nir_dir is not None:
            nir_file = self.nir_dir / name.replace("red", "nir")
            nir = np.array(Image.open(nir_file))
            nir = np.expand_dims(nir, axis=2)
            arr = np.concatenate([rgb, nir], axis=2)
        else:
            arr = rgb

        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
        else:
            arr = arr.astype(np.float32)

        # change from image friendly HWC -> pytorch CHW
        arr = arr.transpose((2, 0, 1))

        x = torch.from_numpy(arr).float()
        return x, name
    

def parse_patch_name(filename):
    stem = Path(filename).stem
    parts = stem.split("_")

    if parts[0] in {"red", "green", "blue", "nir"}:
        parts = parts[1:]

    if parts[0] != "patch":
        raise ValueError(f"Unexpected patch filename format: {filename}")

    patch_idx = int(parts[1])
    row = int(parts[2])

    if parts[3] != "by":
        raise ValueError(f"Unexpected patch filename format: {filename}")

    col = int(parts[4])
    scene_id = "_".join(parts[5:])

    return patch_idx, row, col, scene_id


class TestPatchDatasetWithSceneGT(Dataset):
    def __init__(self, r_dir, g_dir, b_dir, nir_dir, scene_gt_dir, include_nir=True):
        self.r_dir = Path(r_dir)
        self.g_dir = Path(g_dir)
        self.b_dir = Path(b_dir)
        self.nir_dir = Path(nir_dir)
        self.scene_gt_dir = Path(scene_gt_dir)
        self.include_nir = include_nir

        self.red_files = sorted([f for f in self.r_dir.iterdir() if f.is_file()])

        # cache full-scene GTs so they don't get reloaded every sample
        self.gt_cache = {}

    def __len__(self):
        return len(self.red_files)

    def load_scene_gt(self, scene_id):
        if scene_id not in self.gt_cache:
            gt_path = self.scene_gt_dir / f"edited_corrected_gts_{scene_id}.TIF"
            gt = np.array(Image.open(gt_path))
            gt = np.where(gt == 255, 1, 0).astype(np.uint8)
            self.gt_cache[scene_id] = gt
        return self.gt_cache[scene_id]

    def __getitem__(self, idx):
        r_file = self.red_files[idx]
        name = r_file.name

        g_file = self.g_dir / name.replace("red", "green")
        b_file = self.b_dir / name.replace("red", "blue")
        nir_file = self.nir_dir / name.replace("red", "nir")

        r = np.array(Image.open(r_file))
        g = np.array(Image.open(g_file))
        b = np.array(Image.open(b_file))
        rgb = np.stack([r, g, b], axis=2)

        if self.include_nir:
            nir = np.array(Image.open(nir_file))
            nir = np.expand_dims(nir, axis=2)
            arr = np.concatenate([rgb, nir], axis=2)
        else:
            arr = rgb

        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
        else:
            arr = arr.astype(np.float32)

        patch_h, patch_w = arr.shape[:2]

        # HWC -> CHW
        x = torch.from_numpy(arr.transpose((2, 0, 1))).float()

        _, row, col, scene_id = parse_patch_name(name)

        full_gt = self.load_scene_gt(scene_id)

        row_start = (row - 1) * patch_h
        row_end = row_start + patch_h
        col_start = (col - 1) * patch_w
        col_end = col_start + patch_w

        gt_patch = full_gt[row_start:row_end, col_start:col_end]

        # check
        if gt_patch.shape != (patch_h, patch_w):
            raise ValueError(
                f"GT crop shape {gt_patch.shape} does not match patch shape {(patch_h, patch_w)} "
                f"for file {name}. Check patch-grid assumptions."
            )

        y = torch.from_numpy(gt_patch).long()

        return x, y, name