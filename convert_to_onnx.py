from pathlib import Path
import torch
import segmentation_models_pytorch as smp


def build_model(model_type: str):
    if model_type == "unet":
        return smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=4,
            classes=2,
        )
    elif model_type == "unetplusplus":
        return smp.UnetPlusPlus(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=4,
            classes=2,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def load_checkpoint_weights(model, checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def export_onnx(model, out_path: str, patch_h: int, patch_w: int):
    dummy = torch.randn(1, 4, patch_h, patch_w)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamo=False,
    )


def main():
    device = torch.device("cpu")

    PATCH_H = 384  # make static. better for tensorrt rather than potentially slightly changing each time
    PATCH_W = 384

    exports = [
        {
            "model_type": "unet",
            "checkpoint": r"C:\Users\racha\OneDrive\Desktop\cloud\cloud\models\Unet_20260413_004720\Unet.pth",
            "onnx_out": r"C:\Users\racha\OneDrive\Desktop\cloud\cloud\onnx_models\Unet.onnx",
        },
        {
            "model_type": "unetplusplus",
            "checkpoint": r"C:\Users\racha\OneDrive\Desktop\cloud\cloud\models\UnetPlusPlus_20260413_150345\UnetPlusPlus.pth",
            "onnx_out": r"C:\Users\racha\OneDrive\Desktop\cloud\cloud\onnx_models\UnetPlusPlus.onnx",
        },
    ]

    for item in exports:
        Path(item["onnx_out"]).parent.mkdir(parents=True, exist_ok=True)

        model = build_model(item["model_type"])
        model = load_checkpoint_weights(model, item["checkpoint"], device)
        export_onnx(model, item["onnx_out"], PATCH_H, PATCH_W)
        print(f"Exported: {item['onnx_out']}")


if __name__ == "__main__":
    main()
