"""Post-Training Quantization (PTQ) module using PyTorch native quantization."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Union

import numpy as np
import torch
import torch.nn as nn
from torch.ao.quantization import (
    get_default_qconfig,
    prepare,
    convert,
    QConfigMapping,
)
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

logger = logging.getLogger(__name__)


@dataclass
class PTQConfig:
    """Configuration for Post-Training Quantization."""
    
    backend: str = "x86"
    """Quantization backend: 'x86', 'qnnpack', or 'onednn'."""
    
    calibration_samples: int = 100
    """Number of calibration samples to use."""
    
    per_channel: bool = True
    """Whether to use per-channel quantization for weights."""
    
    symmetric: bool = True
    """Whether to use symmetric quantization."""
    
    dtype: str = "qint8"
    """Quantization data type: 'qint8' or 'quint8'."""
    
    def get_qconfig(self):
        """Get PyTorch QConfig based on settings."""
        return get_default_qconfig(self.backend)


@dataclass
class QuantizationEncodings:
    """Container for quantization encodings/parameters."""
    
    layer_encodings: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per-layer quantization parameters."""
    
    global_config: dict[str, Any] = field(default_factory=dict)
    """Global quantization configuration."""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "version": "1.0",
            "global_config": self.global_config,
            "layer_encodings": self.layer_encodings,
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save encodings to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info(f"Saved encodings to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "QuantizationEncodings":
        """Load encodings from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())
        return cls(
            layer_encodings=data.get("layer_encodings", {}),
            global_config=data.get("global_config", {}),
        )


def calibrate_model(
    model: nn.Module,
    calibration_data: Iterator[torch.Tensor] | list[torch.Tensor],
    config: PTQConfig | None = None,
) -> nn.Module:
    """
    Prepare model for quantization and run calibration.
    
    This inserts observers to collect activation statistics.
    
    Args:
        model: FX-traced PyTorch model to calibrate.
        calibration_data: Iterator or list of input tensors.
        config: PTQ configuration.
        
    Returns:
        Model with calibration complete (ready for conversion).
    """
    config = config or PTQConfig()
    
    logger.info(f"Preparing model for calibration (backend={config.backend})")
    
    # Set quantization backend
    torch.backends.quantized.engine = config.backend
    
    # Create QConfig mapping
    qconfig_mapping = QConfigMapping().set_global(config.get_qconfig())
    
    # Prepare model with FX
    model.eval()
    
    # Get example input for tracing
    if isinstance(calibration_data, list):
        example_input = calibration_data[0]
        data_iter = iter(calibration_data)
    else:
        calibration_data = list(calibration_data)
        example_input = calibration_data[0]
        data_iter = iter(calibration_data)
    
    try:
        prepared_model = prepare_fx(
            model,
            qconfig_mapping,
            example_inputs=(example_input,),
        )
    except Exception as e:
        logger.warning(f"FX preparation failed, falling back to eager mode: {e}")
        # Fallback to eager mode
        model.qconfig = config.get_qconfig()
        prepared_model = prepare(model)
    
    # Run calibration
    logger.info(f"Running calibration with {config.calibration_samples} samples")
    prepared_model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(data_iter):
            if i >= config.calibration_samples:
                break
            if isinstance(data, (tuple, list)):
                prepared_model(*data)
            else:
                prepared_model(data)
            
            if (i + 1) % 20 == 0:
                logger.debug(f"Calibrated {i + 1} samples")
    
    logger.info("Calibration complete")
    return prepared_model


def quantize_model(
    prepared_model: nn.Module,
    use_fx: bool = True,
) -> nn.Module:
    """
    Convert a calibrated model to quantized form.
    
    Args:
        prepared_model: Model after calibration.
        use_fx: Whether to use FX-based conversion.
        
    Returns:
        Quantized model.
    """
    logger.info("Converting to quantized model")
    
    if use_fx:
        try:
            quantized_model = convert_fx(prepared_model)
        except Exception as e:
            logger.warning(f"FX conversion failed, falling back to eager: {e}")
            quantized_model = convert(prepared_model)
    else:
        quantized_model = convert(prepared_model)
    
    logger.info("Quantization complete")
    return quantized_model


def export_encodings(
    quantized_model: nn.Module,
    output_path: Union[str, Path],
) -> QuantizationEncodings:
    """
    Export quantization encodings from a quantized model.
    
    Args:
        quantized_model: Quantized PyTorch model.
        output_path: Path to save encodings JSON.
        
    Returns:
        QuantizationEncodings object.
    """
    encodings = QuantizationEncodings()
    
    # Extract quantization parameters from each quantized module
    for name, module in quantized_model.named_modules():
        layer_enc = {}
        
        # Check for weight quantization — handle both native quantized
        # modules (weight is callable) and onnx2torch modules (weight
        # is a plain Parameter)
        try:
            w = module.weight
            if callable(w):
                w = w()
            if hasattr(w, 'q_scale'):
                layer_enc['weight'] = {
                    'scale': float(w.q_scale()),
                    'zero_point': int(w.q_zero_point()),
                    'dtype': str(w.dtype),
                }
        except Exception:
            pass
        
        # Check for activation quantization (scale/zero_point attributes)
        if hasattr(module, 'scale') and hasattr(module, 'zero_point'):
            try:
                layer_enc['activation'] = {
                    'scale': float(module.scale),
                    'zero_point': int(module.zero_point),
                }
            except (AttributeError, TypeError):
                pass
        
        if layer_enc:
            encodings.layer_encodings[name] = layer_enc
    
    encodings.global_config = {
        'backend': torch.backends.quantized.engine,
        'num_layers_quantized': len(encodings.layer_encodings),
    }
    
    encodings.save(output_path)
    return encodings


def export_quantized_onnx(
    quantized_model: nn.Module,
    output_path: Union[str, Path],
    input_shape: tuple[int, ...],
    opset_version: int = 13,
) -> Path:
    """
    Export quantized model to ONNX format.
    
    Args:
        quantized_model: Quantized PyTorch model.
        output_path: Path for output ONNX file.
        input_shape: Input tensor shape.
        opset_version: ONNX opset version.
        
    Returns:
        Path to exported ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting quantized ONNX to {output_path}")
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Export to ONNX
    quantized_model.eval()
    torch.onnx.export(
        quantized_model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
    )
    
    logger.info(f"Exported quantized ONNX model to {output_path}")
    return output_path


def run_ptq_pipeline(
    model: nn.Module,
    calibration_data: Iterator[torch.Tensor] | list[torch.Tensor],
    output_dir: Union[str, Path],
    input_shape: tuple[int, ...],
    config: PTQConfig | None = None,
    model_name: str = "model",
) -> tuple[nn.Module, dict[str, Path]]:
    """
    Run the complete PTQ pipeline.
    
    Args:
        model: PyTorch model to quantize.
        calibration_data: Calibration data.
        output_dir: Output directory for artifacts.
        input_shape: Input tensor shape.
        config: PTQ configuration.
        model_name: Name for output files.
        
    Returns:
        Dict mapping artifact type to path.
    """
    config = config or PTQConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting PTQ pipeline for {model_name}")
    
    outputs = {}
    
    # Step 1: Calibrate
    prepared = calibrate_model(model, calibration_data, config)
    
    # Step 2: Quantize
    quantized = quantize_model(prepared)
    
    # Step 3: Save quantized model
    model_path = output_dir / f"{model_name}_quantized.pt"
    torch.save(quantized.state_dict(), model_path)
    outputs["quantized_model"] = model_path
    
    # Step 4: Export encodings
    encodings_path = output_dir / f"{model_name}_encodings.json"
    export_encodings(quantized, encodings_path)
    outputs["encodings"] = encodings_path
    
    # Step 5: Export to ONNX
    onnx_path = output_dir / f"{model_name}_quantized.onnx"
    try:
        export_quantized_onnx(quantized, onnx_path, input_shape)
        outputs["quantized_onnx"] = onnx_path
    except Exception as e:
        logger.warning(f"ONNX export failed: {e}")
    
    logger.info(f"PTQ pipeline complete. Outputs: {list(outputs.keys())}")
    return quantized, outputs
