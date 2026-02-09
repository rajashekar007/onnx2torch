"""Unit tests for the validator module."""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
from pathlib import Path


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def onnx_and_pytorch_models():
    """Create matching ONNX and PyTorch models."""
    model = SimpleModel()
    model.eval()
    
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        dummy_input = torch.randn(1, 10)
        torch.onnx.export(
            model,
            dummy_input,
            f.name,
            opset_version=13,
            input_names=["input"],
            output_names=["output"],
        )
        return Path(f.name), model


class TestAccuracyMetrics:
    """Tests for compare_outputs function."""
    
    def test_identical_outputs(self):
        """Test comparing identical outputs."""
        from onnx2fx.validator import compare_outputs
        
        output = np.random.randn(1, 10).astype(np.float32)
        
        metrics = compare_outputs(output, output.copy())
        
        assert metrics.mse < 1e-10
        assert metrics.max_diff < 1e-10
        assert metrics.passed
        assert metrics.correlation > 0.999
    
    def test_similar_outputs(self):
        """Test comparing similar outputs."""
        from onnx2fx.validator import compare_outputs
        
        output1 = np.random.randn(1, 10).astype(np.float32)
        noise = np.random.randn(1, 10).astype(np.float32) * 0.001
        output2 = output1 + noise
        
        metrics = compare_outputs(output1, output2, tolerance=1.0)
        
        assert metrics.mse < 0.01
        assert metrics.correlation > 0.99
    
    def test_different_outputs_fail(self):
        """Test that very different outputs fail."""
        from onnx2fx.validator import compare_outputs
        
        output1 = np.ones((1, 10), dtype=np.float32)
        output2 = np.ones((1, 10), dtype=np.float32) * 2  # 100% different
        
        metrics = compare_outputs(output1, output2, tolerance=1.0)
        
        # This should fail the tolerance check
        assert metrics.relative_error > 1.0
        assert not metrics.passed
    
    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        from onnx2fx.validator import compare_outputs
        
        output = np.random.randn(1, 10).astype(np.float32)
        metrics = compare_outputs(output, output.copy())
        
        d = metrics.to_dict()
        assert "mse" in d
        assert "max_diff" in d
        assert "passed" in d
        assert isinstance(d["mse"], float)


class TestAccuracyValidator:
    """Tests for AccuracyValidator class."""
    
    def test_validator_creation(self, onnx_and_pytorch_models):
        """Test validator initialization."""
        from onnx2fx.validator import AccuracyValidator
        
        onnx_path, pytorch_model = onnx_and_pytorch_models
        
        validator = AccuracyValidator(onnx_path, pytorch_model)
        
        assert validator.tolerance == 1.0
        assert len(validator.input_info) == 1
    
    def test_validate_matching_models(self, onnx_and_pytorch_models):
        """Test validation with matching models."""
        from onnx2fx.validator import AccuracyValidator
        
        onnx_path, pytorch_model = onnx_and_pytorch_models
        
        validator = AccuracyValidator(onnx_path, pytorch_model, tolerance=1.0)
        metrics = validator.validate(num_samples=3)
        
        # Same model should match closely
        assert metrics.passed
        assert metrics.relative_error < 0.1  # Very close
    
    def test_generate_report(self, onnx_and_pytorch_models, tmp_path):
        """Test report generation."""
        from onnx2fx.validator import AccuracyValidator
        
        onnx_path, pytorch_model = onnx_and_pytorch_models
        
        validator = AccuracyValidator(onnx_path, pytorch_model)
        
        report_path = tmp_path / "report.md"
        report_content = validator.generate_report(report_path)
        
        assert report_path.exists()
        assert "Accuracy Validation Report" in report_content
        assert "MSE" in report_content
