#!/usr/bin/env python3
"""Run M2 models through the full onnx2fx pipeline.

For each ONNX model in models/:
  1. Convert ONNX → FX-traceable PyTorch
  2. Validate accuracy (ONNX vs PyTorch, <1% relative error)
  3. Operator gap analysis
  4. Post-Training Quantization (PTQ)
  5. Quantization-Aware Fine-Tuning (QFT, 1 epoch)
  6. Generate per-model summary report
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_models")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"


# ── Model definitions ──────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    name: str
    onnx_file: str
    input_shape: tuple[int, ...]
    description: str = ""
    extra_inputs: list[tuple[tuple[int, ...], str]] | None = None
    """Additional inputs as list of (shape, dtype) tuples, e.g. for multi-input models."""


M2_MODELS = [
    ModelConfig(
        name="resnet50",
        onnx_file="resnet50.onnx",
        input_shape=(1, 3, 224, 224),
        description="ResNet-50 (ImageNet classification)",
    ),
    ModelConfig(
        name="convnext_tiny",
        onnx_file="convnext_tiny.onnx",
        input_shape=(1, 3, 224, 224),
        description="ConvNeXt-Tiny (ImageNet classification)",
    ),
    ModelConfig(
        name="x3dm",
        onnx_file="x3dm.onnx",
        input_shape=(1, 3, 16, 224, 224),
        description="X3D-M (Video classification, 16 frames)",
    ),
    ModelConfig(
        name="rtdetrv2",
        onnx_file="rtdetrv2.onnx",
        input_shape=(1, 3, 640, 640),
        description="RT-DETRv2-S (Real-time object detection)",
        extra_inputs=[((1, 2), "int64")],  # orig_target_sizes
    ),
    ModelConfig(
        name="yolov11n",
        onnx_file="yolov11n.onnx",
        input_shape=(1, 3, 640, 640),
        description="YOLOv11n (Object detection)",
    ),
]


# ── Pipeline steps ──────────────────────────────────────────────────────────
def step_convert(model_cfg: ModelConfig, out_dir: Path) -> dict[str, Any]:
    """Step 1: ONNX → FX-traceable PyTorch conversion.

    First tries with FX tracing enabled. If FX tracing fails but ONNX→PyTorch
    conversion succeeded, retries without FX verification so the rest of the
    pipeline (accuracy, PTQ, QFT) can still proceed.
    """
    from onnx2fx.converter import OnnxToFxConverter

    onnx_path = MODELS_DIR / model_cfg.onnx_file
    result: dict[str, Any] = {"status": "not_started"}

    # Try with FX verification first
    logger.info("[%s] Converting ONNX → PyTorch FX...", model_cfg.name)
    try:
        converter = OnnxToFxConverter(
            onnx_path,
            verify_fx_trace=True,
            sample_input_shape=model_cfg.input_shape,
        )
        input_info = converter.get_input_info()
        result["input_info"] = input_info
        logger.info("[%s] Input info: %s", model_cfg.name, input_info)

        pytorch_model = converter.convert()
        saved = converter.save(out_dir)
        result["status"] = "success"
        result["saved_paths"] = {k: str(v) for k, v in saved.items()}
        result["fx_traced"] = True
    except Exception as fx_err:
        fx_error_msg = str(fx_err)
        logger.warning("[%s] FX-traced conversion failed: %s", model_cfg.name, fx_error_msg)
        logger.info("[%s] Retrying conversion without FX verification...", model_cfg.name)

        # Retry without FX verification
        converter = OnnxToFxConverter(
            onnx_path,
            verify_fx_trace=False,
            sample_input_shape=model_cfg.input_shape,
        )
        if "input_info" not in result:
            result["input_info"] = converter.get_input_info()

        pytorch_model = converter.convert()
        saved = converter.save(out_dir)
        result["status"] = "success"
        result["saved_paths"] = {k: str(v) for k, v in saved.items()}
        result["fx_traced"] = False
        result["fx_trace_error"] = fx_error_msg

    # Quick sanity forward pass
    dummy = torch.randn(*model_cfg.input_shape)
    extra_args = []
    if model_cfg.extra_inputs:
        for shape, dtype in model_cfg.extra_inputs:
            if dtype == "int64":
                extra_args.append(torch.tensor([[640, 640]], dtype=torch.int64))
            else:
                extra_args.append(torch.randn(*shape))
    with torch.no_grad():
        out = pytorch_model(dummy, *extra_args)
    if isinstance(out, torch.Tensor):
        result["output_shape"] = list(out.shape)
    elif isinstance(out, (tuple, list)):
        result["output_shape"] = [list(o.shape) for o in out if isinstance(o, torch.Tensor)]
    logger.info("[%s] Conversion succeeded. Output shape: %s (FX traced: %s)",
                model_cfg.name, result.get("output_shape"), result["fx_traced"])

    return result, pytorch_model


def step_validate(model_cfg: ModelConfig, pytorch_model, out_dir: Path) -> dict[str, Any]:
    """Step 2: Accuracy validation — ONNX vs PyTorch."""
    from onnx2fx.validator import AccuracyValidator

    onnx_path = MODELS_DIR / model_cfg.onnx_file
    result: dict[str, Any] = {"status": "not_started"}

    logger.info("[%s] Validating accuracy...", model_cfg.name)
    # Detection models have outputs dominated by near-zero values so
    # relative error is misleading; use absolute tolerance instead.
    is_detection = model_cfg.name in ("rtdetrv2", "yolov11n")
    validator = AccuracyValidator(
        onnx_path,
        pytorch_model,
        tolerance=1.0,
        atol=0.01 if is_detection else None,
    )
    metrics = validator.validate(num_samples=3)
    result["status"] = "success"
    result["metrics"] = metrics.to_dict()
    result["passed"] = metrics.passed
    logger.info(
        "[%s] Accuracy: relative_error=%.4f%%, passed=%s",
        model_cfg.name, metrics.relative_error, metrics.passed,
    )

    # Save accuracy report
    report_path = out_dir / "accuracy_report.md"
    validator.generate_report(report_path)
    result["report_path"] = str(report_path)

    return result


def step_operator_analysis(model_cfg: ModelConfig, out_dir: Path) -> dict[str, Any]:
    """Step 3: Operator gap analysis."""
    from onnx2fx.operators import generate_operator_report

    onnx_path = MODELS_DIR / model_cfg.onnx_file
    result: dict[str, Any] = {"status": "not_started"}

    logger.info("[%s] Analyzing operators...", model_cfg.name)
    report = generate_operator_report(
        onnx_path,
        output_path=out_dir / "operator_report.md",
        format="markdown",
    )
    result["status"] = "success"
    result["coverage_percent"] = report.coverage_percent
    result["total_ops"] = len(report.operators)
    result["supported"] = len(report.supported_ops)
    result["unsupported"] = len(report.unsupported_ops)
    result["unknown"] = len(report.unknown_ops)
    logger.info(
        "[%s] Operator coverage: %.1f%% (%d supported, %d unsupported, %d unknown)",
        model_cfg.name, report.coverage_percent,
        len(report.supported_ops), len(report.unsupported_ops), len(report.unknown_ops),
    )

    # Also save JSON
    json_path = out_dir / "operator_report.json"
    json_path.write_text(json.dumps(report.to_dict(), indent=2))
    result["report_path"] = str(out_dir / "operator_report.md")

    return result


def step_ptq(model_cfg: ModelConfig, pytorch_model, out_dir: Path) -> dict[str, Any]:
    """Step 4: Post-Training Quantization."""
    from onnx2fx.quantization.ptq import run_ptq_pipeline, PTQConfig

    result: dict[str, Any] = {"status": "not_started"}
    ptq_dir = out_dir / "ptq"
    ptq_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[%s] Running PTQ...", model_cfg.name)

    # Multi-input models need special calibration data; skip for now
    if model_cfg.extra_inputs:
        result["status"] = "skipped"
        result["reason"] = "PTQ not yet supported for multi-input models"
        logger.info("[%s] PTQ skipped (multi-input model)", model_cfg.name)
        return result

    # Generate synthetic calibration data
    calib_data = [torch.randn(*model_cfg.input_shape) for _ in range(20)]

    config = PTQConfig(calibration_samples=20)
    _, ptq_outputs = run_ptq_pipeline(
        model=pytorch_model,
        calibration_data=calib_data,
        output_dir=ptq_dir,
        input_shape=model_cfg.input_shape,
        config=config,
        model_name=model_cfg.name,
    )
    result["status"] = "success"
    result["artifacts"] = {k: str(v) for k, v in ptq_outputs.items() if isinstance(v, Path)}
    logger.info("[%s] PTQ completed", model_cfg.name)

    return result


def step_qft(model_cfg: ModelConfig, pytorch_model, out_dir: Path) -> dict[str, Any]:
    """Step 5: Quantization-Aware Fine-Tuning (1 epoch)."""
    from onnx2fx.quantization.qft import run_qft_pipeline, QFTConfig
    from torch.utils.data import DataLoader, TensorDataset

    result: dict[str, Any] = {"status": "not_started"}
    qft_dir = out_dir / "qft"
    qft_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[%s] Running QFT (1 epoch)...", model_cfg.name)

    # Multi-input models need special training data; skip for now
    if model_cfg.extra_inputs:
        result["status"] = "skipped"
        result["reason"] = "QFT not yet supported for multi-input models"
        logger.info("[%s] QFT skipped (multi-input model)", model_cfg.name)
        return result

    # Create synthetic training data
    num_samples = 32
    inputs = torch.randn(num_samples, *model_cfg.input_shape[1:])

    # Determine output size for random labels
    with torch.no_grad():
        sample_out = pytorch_model(torch.randn(*model_cfg.input_shape))
    if isinstance(sample_out, torch.Tensor) and sample_out.dim() == 2:
        num_classes = sample_out.shape[1]
    else:
        num_classes = 10  # fallback

    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(inputs, labels)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    config = QFTConfig(epochs=1, learning_rate=1e-4)
    _, qft_outputs = run_qft_pipeline(
        model=pytorch_model,
        train_loader=train_loader,
        output_dir=qft_dir,
        input_shape=model_cfg.input_shape,
        config=config,
        model_name=model_cfg.name,
    )
    result["status"] = "success"
    result["artifacts"] = {k: str(v) for k, v in qft_outputs.items() if isinstance(v, Path)}
    logger.info("[%s] QFT completed", model_cfg.name)

    return result


def step_report(model_cfg: ModelConfig, results: dict, out_dir: Path) -> dict[str, Any]:
    """Step 6: Generate per-model summary report."""
    from onnx2fx.reports import ModelReport

    logger.info("[%s] Generating summary report...", model_cfg.name)

    report = ModelReport(model_name=model_cfg.name)
    report.description = model_cfg.description
    report.onnx_path = str(MODELS_DIR / model_cfg.onnx_file)

    # Populate from results
    conv = results.get("conversion", {})
    report.conversion_status = conv.get("status", "not_started")
    report.fx_traceable = conv.get("fx_traced", False)
    report.input_info = conv.get("input_info", [])
    report.output_shape = conv.get("output_shape")

    val = results.get("validation", {})
    report.accuracy_status = val.get("status", "not_started")
    report.accuracy_metrics = val.get("metrics", {})
    report.accuracy_passed = val.get("passed", False)

    ops = results.get("operators", {})
    report.operator_status = ops.get("status", "not_started")
    report.operator_coverage = ops.get("coverage_percent", 0)

    ptq = results.get("ptq", {})
    report.ptq_status = ptq.get("status", "not_started")

    qft = results.get("qft", {})
    report.qft_status = qft.get("status", "not_started")

    saved = report.save(out_dir / "report", format="both")
    logger.info("[%s] Report saved: %s", model_cfg.name, saved)

    return {"status": "success", "paths": {k: str(v) for k, v in saved.items()}}


# ── Main orchestrator ───────────────────────────────────────────────────────
def test_model(model_cfg: ModelConfig) -> dict[str, Any]:
    """Run full pipeline for a single model."""
    out_dir = OUTPUT_DIR / model_cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "model": model_cfg.name,
        "description": model_cfg.description,
        "input_shape": list(model_cfg.input_shape),
    }

    # Check ONNX file exists
    onnx_path = MODELS_DIR / model_cfg.onnx_file
    if not onnx_path.exists():
        results["error"] = f"ONNX file not found: {onnx_path}"
        logger.error("[%s] %s", model_cfg.name, results["error"])
        return results

    pytorch_model = None

    # Step 1: Convert
    try:
        conv_result, pytorch_model = step_convert(model_cfg, out_dir)
        results["conversion"] = conv_result
    except Exception as e:
        results["conversion"] = {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
        logger.error("[%s] Conversion failed: %s", model_cfg.name, e)

    # Step 2: Validate (needs successful conversion)
    if pytorch_model is not None:
        try:
            results["validation"] = step_validate(model_cfg, pytorch_model, out_dir)
        except Exception as e:
            results["validation"] = {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
            logger.error("[%s] Validation failed: %s", model_cfg.name, e)
    else:
        results["validation"] = {"status": "skipped", "reason": "conversion failed"}

    # Step 3: Operator analysis (independent of conversion)
    try:
        results["operators"] = step_operator_analysis(model_cfg, out_dir)
    except Exception as e:
        results["operators"] = {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
        logger.error("[%s] Operator analysis failed: %s", model_cfg.name, e)

    # Step 4: PTQ (needs successful conversion)
    if pytorch_model is not None:
        try:
            results["ptq"] = step_ptq(model_cfg, pytorch_model, out_dir)
        except Exception as e:
            results["ptq"] = {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
            logger.error("[%s] PTQ failed: %s", model_cfg.name, e)
    else:
        results["ptq"] = {"status": "skipped", "reason": "conversion failed"}

    # Step 5: QFT (needs successful conversion — use fresh copy of model)
    if pytorch_model is not None:
        try:
            import copy
            model_copy = copy.deepcopy(pytorch_model)
            results["qft"] = step_qft(model_cfg, model_copy, out_dir)
        except Exception as e:
            results["qft"] = {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
            logger.error("[%s] QFT failed: %s", model_cfg.name, e)
    else:
        results["qft"] = {"status": "skipped", "reason": "conversion failed"}

    # Step 6: Summary report
    try:
        results["report"] = step_report(model_cfg, results, out_dir)
    except Exception as e:
        results["report"] = {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
        logger.error("[%s] Report generation failed: %s", model_cfg.name, e)

    # Save raw results JSON
    results_json = out_dir / "results.json"
    results_json.write_text(json.dumps(results, indent=2, default=str))

    return results


def generate_summary(all_results: list[dict]) -> str:
    """Generate a summary markdown report across all models."""
    lines = [
        "# M2 Model Testing Summary",
        "",
        f"**Date**: {__import__('datetime').datetime.now().isoformat()}",
        f"**Models tested**: {len(all_results)}",
        "",
        "| Model | Conversion | FX Trace | Accuracy | Op Coverage | PTQ | QFT |",
        "|-------|-----------|----------|----------|-------------|-----|-----|",
    ]

    for r in all_results:
        name = r.get("model", "?")
        conv = r.get("conversion", {})
        val = r.get("validation", {})
        ops = r.get("operators", {})
        ptq = r.get("ptq", {})
        qft = r.get("qft", {})

        conv_s = "✅" if conv.get("status") == "success" else "❌"
        fx_s = "✅" if conv.get("fx_traced") else "❌"

        if val.get("status") == "success":
            err = val.get("metrics", {}).get("relative_error", "?")
            passed = val.get("passed", False)
            val_s = f"{'✅' if passed else '⚠️'} {err:.2f}%" if isinstance(err, (int, float)) else "❓"
        else:
            val_s = "❌" if val.get("status") == "error" else "⏭️"

        ops_s = f"{ops.get('coverage_percent', 0):.0f}%" if ops.get("status") == "success" else "❌"
        ptq_s = "✅" if ptq.get("status") == "success" else ("❌" if ptq.get("status") == "error" else "⏭️")
        qft_s = "✅" if qft.get("status") == "success" else ("❌" if qft.get("status") == "error" else "⏭️")

        lines.append(f"| {name} | {conv_s} | {fx_s} | {val_s} | {ops_s} | {ptq_s} | {qft_s} |")

    lines.extend(["", "## Per-Model Details", ""])
    for r in all_results:
        name = r.get("model", "?")
        desc = r.get("description", "")
        lines.append(f"### {name}")
        lines.append(f"_{desc}_")
        lines.append(f"- Input shape: `{r.get('input_shape')}`")

        conv = r.get("conversion", {})
        lines.append(f"- Conversion: **{conv.get('status', '?')}**")
        if conv.get("error"):
            lines.append(f"  - Error: `{conv['error'][:200]}`")

        val = r.get("validation", {})
        lines.append(f"- Validation: **{val.get('status', '?')}**")
        if val.get("metrics"):
            m = val["metrics"]
            lines.append(f"  - Relative error: {m.get('relative_error', '?')}%")
            lines.append(f"  - MSE: {m.get('mse', '?')}")
            lines.append(f"  - Passed: {val.get('passed', '?')}")

        ops = r.get("operators", {})
        if ops.get("status") == "success":
            lines.append(f"- Operators: {ops.get('coverage_percent', 0):.1f}% coverage ({ops.get('supported', 0)} supported, {ops.get('unsupported', 0)} unsupported)")

        for step_name in ("ptq", "qft"):
            step = r.get(step_name, {})
            lines.append(f"- {step_name.upper()}: **{step.get('status', '?')}**")
            if step.get("error"):
                lines.append(f"  - Error: `{step['error'][:200]}`")

        lines.append("")

    return "\n".join(lines)


def main(model_names: list[str] | None = None):
    """Run pipeline for specified models (or all M2 models)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if model_names:
        models = [m for m in M2_MODELS if m.name in model_names]
        if not models:
            logger.error("No matching models found. Available: %s", [m.name for m in M2_MODELS])
            return
    else:
        models = M2_MODELS

    all_results = []
    for model_cfg in models:
        print(f"\n{'='*60}")
        print(f"  Testing: {model_cfg.name} ({model_cfg.description})")
        print(f"{'='*60}")
        result = test_model(model_cfg)
        all_results.append(result)

    # Generate summary
    summary = generate_summary(all_results)
    summary_path = OUTPUT_DIR / "summary.md"
    summary_path.write_text(summary)
    print(f"\n{'='*60}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")
    print(summary)


if __name__ == "__main__":
    model_args = sys.argv[1:] if len(sys.argv) > 1 else None
    main(model_args)
