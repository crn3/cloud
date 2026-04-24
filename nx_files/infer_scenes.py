from pathlib import Path
import math
import time
import numpy as np
from PIL import Image

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

PATCH_SIZE = 384
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_scene(scene_dir: Path):
    b2 = np.array(Image.open(next(scene_dir.glob("*_B2.TIF"))))
    b3 = np.array(Image.open(next(scene_dir.glob("*_B3.TIF"))))
    b4 = np.array(Image.open(next(scene_dir.glob("*_B4.TIF"))))
    b5 = np.array(Image.open(next(scene_dir.glob("*_B5.TIF"))))

    arr = np.stack([b4, b3, b2, b5], axis=2)  # (red green blue nir)

    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
    else:
        arr = arr.astype(np.float32)

    return arr  # (HWC)


def iter_patches(arr, patch_size=384):
    h, w, c = arr.shape
    n_rows = math.ceil(h / patch_size)
    n_cols = math.ceil(w / patch_size)

    for row in range(n_rows):
        for col in range(n_cols):
            r0 = row * patch_size
            c0 = col * patch_size
            r1 = min(r0 + patch_size, h)
            c1 = min(c0 + patch_size, w)

            patch = arr[r0:r1, c0:c1, :]
            valid_h = r1 - r0
            valid_w = c1 - c0

            if valid_h < patch_size or valid_w < patch_size:
                padded = np.zeros((patch_size, patch_size, c), dtype=np.float32)
                padded[:valid_h, :valid_w, :] = patch
                patch = padded

            yield patch, (r0, c0, valid_h, valid_w)


def chw_batch(patch_hwc):
    x = patch_hwc.transpose(2, 0, 1)  # HWC -> CHW
    x = np.expand_dims(x, axis=0)  # CHW -> BCHW
    return np.ascontiguousarray(x, dtype=np.float32)


class TRTInference:
    def __init__(self, engine_path: str):
        self.engine_path = str(engine_path)

        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

        print("Loaded engine:", self.engine_path)
        print("Input tensor:", self.input_name, self.input_shape)
        print("Output tensor:", self.output_name, self.output_shape)

        self.host_input = np.empty(self.input_shape, dtype=np.float32)
        self.host_output = np.empty(self.output_shape, dtype=np.float32)

        self.device_input = cuda.mem_alloc(self.host_input.nbytes)
        self.device_output = cuda.mem_alloc(self.host_output.nbytes)

        self.stream = cuda.Stream()

        self.context.set_tensor_address(self.input_name, int(self.device_input))
        self.context.set_tensor_address(self.output_name, int(self.device_output))

    def predict_logits(self, x):
        if x.shape != self.input_shape:
            raise ValueError(f"Expected input shape {self.input_shape}, got {x.shape}")

        np.copyto(self.host_input, x)

        cuda.memcpy_htod_async(self.device_input, self.host_input, self.stream)

        ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execution failed")

        cuda.memcpy_dtoh_async(self.host_output, self.device_output, self.stream)
        self.stream.synchronize()

        return self.host_output.copy()


def infer_scene(scene_dir: Path, trt_model: TRTInference):
    arr = load_scene(scene_dir)
    h, w, _ = arr.shape

    print(f"Scene: {scene_dir.name}")
    print(f"Scene shape: {h} x {w}")

    cloud_pixels = 0
    total_pixels = 0
    patch_count = 0

    start = time.time()

    for patch_hwc, (_, _, valid_h, valid_w) in iter_patches(arr, PATCH_SIZE):
        x = chw_batch(patch_hwc)  # [1, 4, 384, 384]
        logits = trt_model.predict_logits(x)  # [1, 2, 384, 384]

        pred = np.argmax(logits, axis=1)[0].astype(np.uint8)
        pred_valid = pred[:valid_h, :valid_w]

        cloud_pixels += int(pred_valid.sum())
        total_pixels += pred_valid.size
        patch_count += 1

    elapsed = time.time() - start
    cloud_fraction = cloud_pixels / total_pixels if total_pixels > 0 else 0.0

    print(f"Patches processed: {patch_count}")
    print(f"Cloud pixels: {cloud_pixels}")
    print(f"Total pixels: {total_pixels}")
    print(f"Cloud fraction: {cloud_fraction:.6f}")
    print(f"Elapsed time: {elapsed:.2f} s")

    return {
        "scene_id": scene_dir.name,
        "patches": patch_count,
        "cloud_pixels": cloud_pixels,
        "total_pixels": total_pixels,
        "cloud_fraction": cloud_fraction,
        "elapsed_sec": elapsed,
        "error": "",
    }
