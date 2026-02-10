#!/usr/bin/env python3
"""Download or export ONNX models for M2 milestone testing.

Models:
  - ResNet-50 (from ONNX Model Zoo via HuggingFace)
  - ConvNeXt-Tiny (export from torchvision)
  - X3D-M (export from PyTorchVideo / torch hub)
  - RT-DETRv2 (from HuggingFace)
  - YOLOv11n (from Ultralytics export)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from urllib.request import urlretrieve

import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Print download progress."""
    if total_size > 0:
        pct = min(100, block_num * block_size * 100 // total_size)
        print(f"\r  Downloading... {pct}%", end="", flush=True)


# ---------------------------------------------------------------------------
# 1. ResNet-50
# ---------------------------------------------------------------------------
def download_resnet50(output_dir: Path) -> Path:
    """Download ResNet-50 ONNX from ONNX Model Zoo (HuggingFace mirror)."""
    dest = output_dir / "resnet50.onnx"
    if dest.exists():
        logger.info("ResNet-50 already exists at %s", dest)
        return dest

    url = (
        "https://huggingface.co/onnx-models/resnet50-v1-12/resolve/main/"
        "resnet50-v1-12.onnx"
    )
    logger.info("Downloading ResNet-50 from %s", url)
    try:
        urlretrieve(url, str(dest), reporthook=progress_hook)
        print()  # newline after progress
    except Exception:
        logger.warning("HuggingFace download failed, trying ONNX Model Zoo...")
        url_alt = (
            "https://github.com/onnx/models/raw/main/validated/vision/"
            "classification/resnet/model/resnet50-v1-12.onnx"
        )
        urlretrieve(url_alt, str(dest), reporthook=progress_hook)
        print()

    logger.info("ResNet-50 saved to %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


# ---------------------------------------------------------------------------
# 2. ConvNeXt-Tiny
# ---------------------------------------------------------------------------
def export_convnext_tiny(output_dir: Path) -> Path:
    """Export ConvNeXt-Tiny from torchvision to ONNX."""
    dest = output_dir / "convnext_tiny.onnx"
    if dest.exists():
        logger.info("ConvNeXt-Tiny already exists at %s", dest)
        return dest

    logger.info("Exporting ConvNeXt-Tiny from torchvision...")
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        str(dest),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    logger.info("ConvNeXt-Tiny saved to %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


# ---------------------------------------------------------------------------
# 3. X3D-M (video model)
# ---------------------------------------------------------------------------
def export_x3dm(output_dir: Path) -> Path:
    """Export X3D-M from PyTorchVideo via torch hub to ONNX."""
    dest = output_dir / "x3dm.onnx"
    if dest.exists():
        logger.info("X3D-M already exists at %s", dest)
        return dest

    logger.info("Loading X3D-M from torch hub (facebookresearch/pytorchvideo)...")
    model = torch.hub.load(
        "facebookresearch/pytorchvideo", "x3d_m", pretrained=True
    )
    model.eval()

    # X3D-M expects: batch, channels, temporal(16), height(224), width(224)
    dummy = torch.randn(1, 3, 16, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        str(dest),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    logger.info("X3D-M saved to %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


# ---------------------------------------------------------------------------
# 4. RT-DETRv2
# ---------------------------------------------------------------------------
def download_rtdetrv2(output_dir: Path) -> Path:
    """Download RT-DETRv2-S ONNX from HuggingFace."""
    dest = output_dir / "rtdetrv2.onnx"
    if dest.exists():
        logger.info("RT-DETRv2 already exists at %s", dest)
        return dest

    url = (
        "https://huggingface.co/xnorpx/rt-detr2-onnx/resolve/main/"
        "rt-detrv2-s.onnx"
    )
    logger.info("Downloading RT-DETRv2-S from %s", url)
    urlretrieve(url, str(dest), reporthook=progress_hook)
    print()
    logger.info("RT-DETRv2 saved to %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


# ---------------------------------------------------------------------------
# 5. YOLOv11n
# ---------------------------------------------------------------------------
def export_yolov11n(output_dir: Path) -> Path:
    """Export YOLOv11n using Ultralytics to ONNX."""
    dest = output_dir / "yolov11n.onnx"
    if dest.exists():
        logger.info("YOLOv11n already exists at %s", dest)
        return dest

    logger.info("Exporting YOLOv11n via Ultralytics...")
    try:
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        model.export(format="onnx", imgsz=640, opset=17)
        # Ultralytics exports to same directory as .pt file
        exported = Path("yolo11n.onnx")
        if exported.exists():
            exported.rename(dest)
        else:
            # Check in current dir
            for p in Path(".").glob("yolo11n*.onnx"):
                p.rename(dest)
                break
    except ImportError:
        logger.error(
            "ultralytics not installed. Install with: pip install ultralytics"
        )
        raise

    logger.info("YOLOv11n saved to %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
MODEL_EXPORTERS = {
    "resnet50": download_resnet50,
    "convnext_tiny": export_convnext_tiny,
    "x3dm": export_x3dm,
    "rtdetrv2": download_rtdetrv2,
    "yolov11n": export_yolov11n,
}


def main(models: list[str] | None = None):
    """Download/export specified models (or all if None)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    targets = models or list(MODEL_EXPORTERS.keys())

    results = {}
    for name in targets:
        if name not in MODEL_EXPORTERS:
            logger.warning("Unknown model: %s (available: %s)", name, list(MODEL_EXPORTERS.keys()))
            continue
        try:
            path = MODEL_EXPORTERS[name](MODELS_DIR)
            results[name] = {"status": "ok", "path": str(path)}
            logger.info("✅ %s ready", name)
        except Exception as e:
            results[name] = {"status": "error", "error": str(e)}
            logger.error("❌ %s failed: %s", name, e)

    # Summary
    print("\n" + "=" * 60)
    print("Model Download Summary")
    print("=" * 60)
    for name, info in results.items():
        status = "✅" if info["status"] == "ok" else "❌"
        detail = info.get("path", info.get("error", ""))
        print(f"  {status} {name:20s} {detail}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    # Allow passing model names as CLI args
    model_args = sys.argv[1:] if len(sys.argv) > 1 else None
    main(model_args)
