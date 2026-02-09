"""Accuracy validation between ONNX and PyTorch models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class AccuracyMetrics:
    """Container for accuracy comparison metrics."""
    
    mse: float
    """Mean Squared Error between outputs."""
    
    max_diff: float
    """Maximum absolute difference."""
    
    mean_diff: float
    """Mean absolute difference."""
    
    correlation: float
    """Pearson correlation coefficient."""
    
    relative_error: float
    """Mean relative error (percentage)."""
    
    passed: bool
    """Whether accuracy is within tolerance."""
    
    tolerance: float
    """Tolerance threshold used for comparison."""
    
    def __str__(self) -> str:
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        return (
            f"Accuracy Metrics ({status}):\n"
            f"  MSE:            {self.mse:.6e}\n"
            f"  Max Diff:       {self.max_diff:.6e}\n"
            f"  Mean Diff:      {self.mean_diff:.6e}\n"
            f"  Correlation:    {self.correlation:.6f}\n"
            f"  Relative Error: {self.relative_error:.4f}%\n"
            f"  Tolerance:      {self.tolerance:.4f}%"
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "mse": float(self.mse),
            "max_diff": float(self.max_diff),
            "mean_diff": float(self.mean_diff),
            "correlation": float(self.correlation),
            "relative_error": float(self.relative_error),
            "passed": self.passed,
            "tolerance": float(self.tolerance),
        }


def compare_outputs(
    onnx_output: np.ndarray,
    torch_output: np.ndarray,
    tolerance: float = 1.0,
) -> AccuracyMetrics:
    """
    Compare outputs from ONNX and PyTorch models.
    
    Args:
        onnx_output: Output array from ONNX Runtime.
        torch_output: Output array from PyTorch model.
        tolerance: Maximum allowed relative error (percentage).
        
    Returns:
        AccuracyMetrics with comparison results.
    """
    # Flatten for comparison
    onnx_flat = onnx_output.flatten().astype(np.float64)
    torch_flat = torch_output.flatten().astype(np.float64)
    
    # Compute metrics
    diff = np.abs(onnx_flat - torch_flat)
    mse = float(np.mean((onnx_flat - torch_flat) ** 2))
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    
    # Correlation
    if np.std(onnx_flat) > 0 and np.std(torch_flat) > 0:
        correlation = float(np.corrcoef(onnx_flat, torch_flat)[0, 1])
    else:
        correlation = 1.0 if np.allclose(onnx_flat, torch_flat) else 0.0
    
    # Relative error (avoid division by zero)
    onnx_abs = np.abs(onnx_flat)
    mask = onnx_abs > 1e-8
    if mask.any():
        relative_errors = diff[mask] / onnx_abs[mask]
        relative_error = float(np.mean(relative_errors) * 100)
    else:
        relative_error = 0.0 if mean_diff < 1e-8 else 100.0
    
    passed = relative_error <= tolerance
    
    return AccuracyMetrics(
        mse=mse,
        max_diff=max_diff,
        mean_diff=mean_diff,
        correlation=correlation,
        relative_error=relative_error,
        passed=passed,
        tolerance=tolerance,
    )


class AccuracyValidator:
    """
    Validates accuracy between ONNX and converted PyTorch models.
    
    This class runs both models with the same input and compares outputs
    to ensure the conversion maintains accuracy within the specified tolerance.
    
    Example:
        >>> validator = AccuracyValidator("model.onnx", pytorch_model)
        >>> result = validator.validate(num_samples=10)
        >>> if not result.passed:
        ...     print(f"Accuracy drop: {result.relative_error}%")
    """
    
    def __init__(
        self,
        onnx_model_or_path: Union[str, Path, onnx.ModelProto],
        pytorch_model: nn.Module,
        tolerance: float = 1.0,
        device: str = "cpu",
    ):
        """
        Initialize the validator.
        
        Args:
            onnx_model_or_path: ONNX model path or loaded ModelProto.
            pytorch_model: Converted PyTorch model.
            tolerance: Maximum allowed relative error (percentage).
            device: Device to run PyTorch model on.
        """
        self.tolerance = tolerance
        self.device = device
        
        # Load ONNX model
        if isinstance(onnx_model_or_path, (str, Path)):
            self.onnx_model = onnx.load(str(onnx_model_or_path))
            self.model_path = Path(onnx_model_or_path)
        else:
            self.onnx_model = onnx_model_or_path
            self.model_path = None
            
        # Setup ONNX Runtime session
        self.ort_session = ort.InferenceSession(
            self.onnx_model.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        
        # PyTorch model
        self.pytorch_model = pytorch_model.to(device)
        self.pytorch_model.eval()
        
        # Get input info
        self.input_info = self._get_input_info()
        
    def _get_input_info(self) -> list[dict]:
        """Extract input information from ONNX model."""
        inputs = []
        for inp in self.ort_session.get_inputs():
            inputs.append({
                "name": inp.name,
                "shape": inp.shape,
                "dtype": inp.type,
            })
        return inputs
    
    def _generate_sample_input(
        self,
        batch_size: int = 1,
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate random sample input matching model input spec."""
        if seed is not None:
            np.random.seed(seed)
            
        inputs = {}
        for inp_info in self.input_info:
            shape = list(inp_info["shape"])
            
            # Replace dynamic dimensions
            for i, dim in enumerate(shape):
                if isinstance(dim, str) or dim is None or dim < 0:
                    if i == 0:  # Batch dimension
                        shape[i] = batch_size
                    else:
                        shape[i] = 224  # Default spatial dimension
                        
            # Generate based on dtype
            if "float" in inp_info["dtype"].lower():
                data = np.random.randn(*shape).astype(np.float32)
            elif "int" in inp_info["dtype"].lower():
                data = np.random.randint(0, 100, size=shape).astype(np.int64)
            else:
                data = np.random.randn(*shape).astype(np.float32)
                
            inputs[inp_info["name"]] = data
            
        return inputs
    
    def run_onnx(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Run ONNX model with given inputs."""
        outputs = self.ort_session.run(None, inputs)
        return outputs
    
    def run_pytorch(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Run PyTorch model with given inputs."""
        # Convert to tensors
        torch_inputs = []
        for inp_info in self.input_info:
            data = inputs[inp_info["name"]]
            tensor = torch.from_numpy(data).to(self.device)
            torch_inputs.append(tensor)
        
        # Run model
        with torch.no_grad():
            if len(torch_inputs) == 1:
                outputs = self.pytorch_model(torch_inputs[0])
            else:
                outputs = self.pytorch_model(*torch_inputs)
        
        # Convert outputs to numpy
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        elif isinstance(outputs, tuple):
            outputs = list(outputs)
            
        return [o.cpu().numpy() for o in outputs]
    
    def validate(
        self,
        num_samples: int = 5,
        custom_inputs: list[dict[str, np.ndarray]] | None = None,
    ) -> AccuracyMetrics:
        """
        Validate accuracy across multiple samples.
        
        Args:
            num_samples: Number of random samples to test.
            custom_inputs: Optional list of custom input dicts to use instead.
            
        Returns:
            Aggregate AccuracyMetrics across all samples.
        """
        logger.info(f"Validating accuracy with {num_samples} samples...")
        
        all_metrics = []
        
        if custom_inputs is not None:
            samples = custom_inputs
        else:
            samples = [
                self._generate_sample_input(seed=i) 
                for i in range(num_samples)
            ]
        
        for i, inputs in enumerate(samples):
            # Run both models
            onnx_outputs = self.run_onnx(inputs)
            torch_outputs = self.run_pytorch(inputs)
            
            # Compare first output (primary output)
            metrics = compare_outputs(
                onnx_outputs[0],
                torch_outputs[0],
                tolerance=self.tolerance,
            )
            all_metrics.append(metrics)
            
            logger.debug(f"Sample {i+1}: MSE={metrics.mse:.6e}, RelErr={metrics.relative_error:.4f}%")
        
        # Aggregate metrics
        aggregated = AccuracyMetrics(
            mse=np.mean([m.mse for m in all_metrics]),
            max_diff=np.max([m.max_diff for m in all_metrics]),
            mean_diff=np.mean([m.mean_diff for m in all_metrics]),
            correlation=np.mean([m.correlation for m in all_metrics]),
            relative_error=np.mean([m.relative_error for m in all_metrics]),
            passed=all(m.passed for m in all_metrics),
            tolerance=self.tolerance,
        )
        
        logger.info(f"Validation complete: {'PASSED' if aggregated.passed else 'FAILED'}")
        return aggregated
    
    def generate_report(self, output_path: Union[str, Path] | None = None) -> str:
        """
        Generate a detailed accuracy validation report.
        
        Args:
            output_path: Optional path to save report as markdown.
            
        Returns:
            Report content as string.
        """
        metrics = self.validate()
        
        model_name = self.model_path.stem if self.model_path else "model"
        
        report = f"""# Accuracy Validation Report: {model_name}

## Summary

| Metric | Value | Status |
|--------|-------|--------|
| MSE | {metrics.mse:.6e} | - |
| Max Diff | {metrics.max_diff:.6e} | - |
| Mean Diff | {metrics.mean_diff:.6e} | - |
| Correlation | {metrics.correlation:.6f} | - |
| Relative Error | {metrics.relative_error:.4f}% | {'✓' if metrics.relative_error <= self.tolerance else '✗'} |
| **Overall** | - | **{'PASSED' if metrics.passed else 'FAILED'}** |

## Configuration

- Tolerance: {self.tolerance}%
- Device: {self.device}

## Input Specification

| Name | Shape | Type |
|------|-------|------|
"""
        for inp in self.input_info:
            report += f"| {inp['name']} | {inp['shape']} | {inp['dtype']} |\n"
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            logger.info(f"Saved report to {output_path}")
            
        return report
