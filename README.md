# ONNX2FX — ONNX to FX-Traceable PyTorch Converter

A reusable, extensible pipeline for converting ONNX models to FX-traceable PyTorch models, with integrated Post-Training Quantization (PTQ) and Quantization-Aware Fine-Tuning (QFT) workflows.

## Overview

This tool addresses the SOW requirement of building an ONNX-to-PyTorch model conversion pipeline that:

- Converts customer-provided ONNX models into **Torch FX modules**
- Validates accuracy of converted models against the ONNX reference (< 1% tolerance)
- Provides **operator gap analysis** with supported/unsupported operator reports
- Executes **PTQ** end-to-end, exporting quantization encodings and quantized ONNX models
- Executes **QFT** with 1-epoch fine-tuning, exporting encodings and QFT-optimized ONNX models
- Generates **per-model summary reports** covering conversion, accuracy, and quantization outcomes

## Target Models

| Milestone | Models | Timeline |
|-----------|--------|----------|
| M2 | X3DM, ConvNeXt-Tiny, RT-DETRv2, YOLOv11, CLIP ViT-Large | Start + 8 weeks |
| M3 | EfficientViT, YOLOv8n, RTMDet-Large, ResNet-50, Real-ESRGAN x2 | Start + 13 weeks |

## Installation

```bash
pip install -e .
```

### Dependencies

- `onnx >= 1.14.0`
- `onnx2torch >= 1.5.0`
- `onnxruntime >= 1.15.0`
- `torch >= 2.0.0`
- `numpy`, `typer`, `rich`

## CLI Usage

### Convert a single model

```bash
onnx2fx convert model.onnx -o output/
```

### Validate accuracy (ONNX vs PyTorch)

```bash
onnx2fx validate model.onnx output/model_full.pt --tolerance 1.0
```

### Analyze operator support

```bash
onnx2fx analyze model.onnx -o operator_report.md
```

### Post-Training Quantization

```bash
onnx2fx ptq output/model_full.pt -o quantized/ --shape 1,3,224,224
```

**Outputs:** quantized model (`.pt`), encodings (`.json`), quantized ONNX (`.onnx`)

### Quantization-Aware Fine-Tuning (1 epoch)

```bash
onnx2fx qft output/model_full.pt -o qft_output/ --epochs 1
```

**Outputs:** QFT model (`.pt`), QFT encodings (`.json`), QFT ONNX (`.onnx`)

### Batch processing (all models in a directory)

```bash
onnx2fx batch ./models/ -o ./output/ --ptq --qft
```

## Python API

```python
from onnx2fx import OnnxToFxConverter, AccuracyValidator

# Convert ONNX → FX-traceable PyTorch
converter = OnnxToFxConverter("model.onnx")
pytorch_model = converter.convert()
converter.save("output/")

# Validate accuracy
validator = AccuracyValidator("model.onnx", pytorch_model, tolerance=1.0)
metrics = validator.validate()
print(f"Relative error: {metrics.relative_error:.4f}%")
print(f"Passed: {metrics.passed}")

# Operator analysis
from onnx2fx import scan_onnx_operators
report = scan_onnx_operators("model.onnx")
print(f"Coverage: {report.coverage_percent:.1f}%")
```

## Architecture

```
src/onnx2fx/
├── __init__.py          # Public API
├── converter.py         # ONNX → PyTorch FX conversion (onnx2torch-based)
├── validator.py         # Accuracy validation (ONNX Runtime vs PyTorch)
├── fx_tracer.py         # FX traceability verification & utilities
├── operators.py         # Operator gap analysis & custom converter registry
├── cli.py               # Typer-based CLI
├── quantization/
│   ├── ptq.py           # Post-Training Quantization pipeline
│   └── qft.py           # Quantization-Aware Fine-Tuning pipeline
└── reports/
    └── __init__.py      # Per-model JSON/Markdown report generator
```

## Deliverables (per SOW)

| Deliverable | Module |
|-------------|--------|
| ONNX → Torch FX conversion module | `converter.py`, `fx_tracer.py` |
| Accuracy validation reports | `validator.py`, `reports/` |
| Operator gap analysis | `operators.py` |
| PTQ encodings + quantized ONNX | `quantization/ptq.py` |
| QFT encodings + QFT ONNX | `quantization/qft.py` |
| Per-model summary reports | `reports/` |
| CLI for batch processing | `cli.py` |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=onnx2fx
```

## Known Limitations

- **ONNX export of quantized models** may fail on certain PyTorch versions due to `Conv2dPackedParamsBase` lacking `__obj_flatten__`. The pipeline handles this gracefully and still produces the `.pt` model and encodings.
- **ReduceMean opset 18** and **Reshape allowzero=1** required custom workarounds, included in `converter.py`.

## License

Apache-2.0
