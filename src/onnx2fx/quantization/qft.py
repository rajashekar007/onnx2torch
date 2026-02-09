"""Quantization-Aware Fine-Tuning (QFT/QAT) module using PyTorch native quantization."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.quantization import (
    get_default_qat_qconfig,
    prepare_qat,
    convert,
    QConfigMapping,
)
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class QFTConfig:
    """Configuration for Quantization-Aware Fine-Tuning."""
    
    backend: str = "x86"
    """Quantization backend: 'x86', 'qnnpack', or 'onednn'."""
    
    epochs: int = 1
    """Number of fine-tuning epochs (SOW specifies 1)."""
    
    learning_rate: float = 1e-4
    """Learning rate for fine-tuning."""
    
    batch_size: int = 32
    """Batch size for training."""
    
    per_channel: bool = True
    """Whether to use per-channel weight quantization."""
    
    freeze_bn: bool = False
    """Whether to freeze batch norm during fine-tuning."""
    
    def get_qat_qconfig(self):
        """Get QAT QConfig based on settings."""
        return get_default_qat_qconfig(self.backend)


@dataclass
class QFTEncodings:
    """Container for QFT quantization encodings."""
    
    layer_encodings: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per-layer quantization parameters."""
    
    training_config: dict[str, Any] = field(default_factory=dict)
    """Training configuration used."""
    
    training_metrics: dict[str, Any] = field(default_factory=dict)
    """Training metrics (loss, etc.)."""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "version": "1.0",
            "type": "qft",
            "training_config": self.training_config,
            "training_metrics": self.training_metrics,
            "layer_encodings": self.layer_encodings,
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info(f"Saved QFT encodings to {path}")


def prepare_qat(
    model: nn.Module,
    config: QFTConfig | None = None,
    example_input: torch.Tensor | None = None,
) -> nn.Module:
    """
    Prepare a model for Quantization-Aware Training.
    
    This inserts fake quantization modules that simulate quantization
    during training.
    
    Args:
        model: PyTorch model to prepare.
        config: QFT configuration.
        example_input: Example input for FX tracing.
        
    Returns:
        Model prepared for QAT.
    """
    config = config or QFTConfig()
    
    logger.info(f"Preparing model for QAT (backend={config.backend})")
    
    # Set backend
    torch.backends.quantized.engine = config.backend
    
    # QConfig for QAT
    qconfig_mapping = QConfigMapping().set_global(config.get_qat_qconfig())
    
    model.train()
    
    if example_input is not None:
        try:
            prepared_model = prepare_qat_fx(
                model,
                qconfig_mapping,
                example_inputs=(example_input,),
            )
            logger.info("Using FX-based QAT preparation")
            return prepared_model
        except Exception as e:
            logger.warning(f"FX QAT prep failed, using eager mode: {e}")
    
    # Fallback to eager mode
    model.qconfig = config.get_qat_qconfig()
    prepared_model = prepare_qat(model, inplace=False)
    
    return prepared_model


def train_qft(
    prepared_model: nn.Module,
    train_loader: DataLoader,
    loss_fn: Callable | None = None,
    config: QFTConfig | None = None,
    device: str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """
    Run Quantization-Aware Fine-Tuning for specified epochs.
    
    Args:
        prepared_model: Model prepared for QAT.
        train_loader: Training data loader.
        loss_fn: Loss function (default: CrossEntropyLoss).
        config: QFT configuration.
        device: Device to train on.
        
    Returns:
        Tuple of (trained model, training metrics).
    """
    config = config or QFTConfig()
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    
    logger.info(f"Starting QFT for {config.epochs} epoch(s)")
    
    prepared_model = prepared_model.to(device)
    prepared_model.train()
    
    # Freeze batch norm if configured
    if config.freeze_bn:
        for module in prepared_model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()
    
    optimizer = optim.Adam(
        prepared_model.parameters(),
        lr=config.learning_rate,
    )
    
    metrics = {
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "losses": [],
        "final_loss": None,
    }
    
    for epoch in range(config.epochs):
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Handle different batch formats
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            else:
                inputs = batch
                targets = None
            
            inputs = inputs.to(device)
            if targets is not None:
                targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = prepared_model(inputs)
            
            if targets is not None:
                loss = loss_fn(outputs, targets)
            else:
                # Self-supervised: minimize output variance (placeholder)
                loss = outputs.var()
            
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            if (batch_idx + 1) % 10 == 0:
                logger.debug(
                    f"Epoch {epoch+1}/{config.epochs}, "
                    f"Batch {batch_idx+1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        metrics["losses"].append(avg_loss)
        logger.info(f"Epoch {epoch+1}/{config.epochs} complete, Avg Loss: {avg_loss:.4f}")
    
    metrics["final_loss"] = metrics["losses"][-1] if metrics["losses"] else None
    
    # Switch to eval mode
    prepared_model.eval()
    
    logger.info("QFT training complete")
    return prepared_model, metrics


def export_qft_encodings(
    quantized_model: nn.Module,
    output_path: Union[str, Path],
    training_metrics: dict[str, Any] | None = None,
    config: QFTConfig | None = None,
) -> QFTEncodings:
    """
    Export quantization encodings from QFT model.
    
    Args:
        quantized_model: Quantized model after QFT.
        output_path: Path to save encodings.
        training_metrics: Optional training metrics to include.
        config: QFT configuration used.
        
    Returns:
        QFTEncodings object.
    """
    config = config or QFTConfig()
    
    encodings = QFTEncodings()
    encodings.training_config = {
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "backend": config.backend,
    }
    encodings.training_metrics = training_metrics or {}
    
    # Extract quantization parameters
    for name, module in quantized_model.named_modules():
        layer_enc = {}
        
        if hasattr(module, 'weight') and hasattr(module.weight, 'q_scale'):
            try:
                layer_enc['weight'] = {
                    'scale': float(module.weight().q_scale()),
                    'zero_point': int(module.weight().q_zero_point()),
                    'dtype': str(module.weight().dtype),
                }
            except Exception:
                pass
        
        if hasattr(module, 'scale') and hasattr(module, 'zero_point'):
            try:
                layer_enc['activation'] = {
                    'scale': float(module.scale),
                    'zero_point': int(module.zero_point),
                }
            except Exception:
                pass
        
        if layer_enc:
            encodings.layer_encodings[name] = layer_enc
    
    encodings.save(output_path)
    return encodings


def export_qft_onnx(
    quantized_model: nn.Module,
    output_path: Union[str, Path],
    input_shape: tuple[int, ...],
    opset_version: int = 13,
) -> Path:
    """
    Export QFT-optimized model to ONNX format.
    
    Args:
        quantized_model: Quantized model after QFT.
        output_path: Path for output ONNX file.
        input_shape: Input tensor shape.
        opset_version: ONNX opset version.
        
    Returns:
        Path to exported ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting QFT ONNX to {output_path}")
    
    dummy_input = torch.randn(*input_shape)
    
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
    
    logger.info(f"Exported QFT ONNX to {output_path}")
    return output_path


def run_qft_pipeline(
    model: nn.Module,
    train_loader: DataLoader,
    output_dir: Union[str, Path],
    input_shape: tuple[int, ...],
    loss_fn: Callable | None = None,
    config: QFTConfig | None = None,
    model_name: str = "model",
    device: str = "cpu",
) -> tuple[nn.Module, dict[str, Path]]:
    """
    Run the complete QFT pipeline.
    
    Args:
        model: PyTorch model to quantize.
        train_loader: Training data loader.
        output_dir: Output directory.
        input_shape: Input tensor shape.
        loss_fn: Loss function.
        config: QFT configuration.
        model_name: Name for output files.
        device: Device to train on.
        
    Returns:
        Tuple of (Quantized PyTorch model, Dict mapping artifact type to path).
    """
    config = config or QFTConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting QFT pipeline for {model_name}")
    
    outputs = {}
    
    # Get example input from loader
    example_batch = next(iter(train_loader))
    if isinstance(example_batch, (tuple, list)):
        example_input = example_batch[0][:1]  # First sample
    else:
        example_input = example_batch[:1]
    
    # Step 1: Prepare for QAT
    prepared = prepare_qat(model, config, example_input)
    
    # Step 2: Fine-tune
    trained, metrics = train_qft(
        prepared, train_loader, loss_fn, config, device
    )
    
    # Step 3: Convert to quantized
    try:
        quantized = convert_fx(trained)
    except Exception:
        quantized = convert(trained)
    
    # Step 4: Save quantized model
    model_path = output_dir / f"{model_name}_qft.pt"
    torch.save(quantized.state_dict(), model_path)
    outputs["qft_model"] = model_path
    
    # Step 5: Export encodings
    encodings_path = output_dir / f"{model_name}_qft_encodings.json"
    export_qft_encodings(quantized, encodings_path, metrics, config)
    outputs["qft_encodings"] = encodings_path
    
    # Step 6: Export ONNX
    onnx_path = output_dir / f"{model_name}_qft.onnx"
    try:
        export_qft_onnx(quantized, onnx_path, input_shape)
        outputs["qft_onnx"] = onnx_path
    except Exception as e:
        logger.warning(f"QFT ONNX export failed: {e}")
    
    logger.info(f"QFT pipeline complete. Outputs: {list(outputs.keys())}")
    return quantized, outputs
