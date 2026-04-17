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