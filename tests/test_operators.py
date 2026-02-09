"""Unit tests for the operators module."""

import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.mean(dim=[2, 3])
        x = self.fc(x)
        return x


@pytest.fixture
def sample_onnx_path():
    """Create a sample ONNX model for testing."""
    model = SimpleModel()
    model.eval()
    
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        dummy_input = torch.randn(1, 3, 32, 32)
        torch.onnx.export(
            model,
            dummy_input,
            f.name,
            opset_version=13,
            input_names=["input"],
            output_names=["output"],
        )
        return Path(f.name)


class TestOperatorScanner:
    """Tests for operator scanning functionality."""
    
    def test_scan_operators(self, sample_onnx_path):
        """Test scanning ONNX operators."""
        from onnx2fx.operators import scan_onnx_operators
        
        report = scan_onnx_operators(sample_onnx_path)
        
        assert report.model_name == sample_onnx_path.stem
        assert report.total_nodes > 0
        assert len(report.operators) > 0
        
        # Should have common ops
        op_types = list(report.operators.keys())
        assert any(op in op_types for op in ["Conv", "Relu", "Gemm", "BatchNormalization"])
    
    def test_operator_support_check(self):
        """Test operator support checking."""
        from onnx2fx.operators import check_operator_support
        
        # Known supported
        assert check_operator_support("Relu") is True
        assert check_operator_support("Conv") is True
        assert check_operator_support("MatMul") is True
        
        # Known unsupported
        assert check_operator_support("Loop") is False
        assert check_operator_support("If") is False
        
        # Unknown
        assert check_operator_support("SomeCustomOp") is None
    
    def test_operator_report_properties(self, sample_onnx_path):
        """Test OperatorReport properties."""
        from onnx2fx.operators import scan_onnx_operators
        
        report = scan_onnx_operators(sample_onnx_path)
        
        # Check properties work
        supported = report.supported_ops
        unsupported = report.unsupported_ops
        coverage = report.coverage_percent
        
        assert isinstance(supported, list)
        assert isinstance(unsupported, list)
        assert 0 <= coverage <= 100
    
    def test_generate_report_markdown(self, sample_onnx_path, tmp_path):
        """Test generating markdown report."""
        from onnx2fx.operators import generate_operator_report
        
        output_path = tmp_path / "operators.md"
        report = generate_operator_report(sample_onnx_path, output_path, format="markdown")
        
        assert output_path.exists()
        content = output_path.read_text()
        
        assert "Operator Analysis" in content
        assert "Summary" in content
    
    def test_generate_report_json(self, sample_onnx_path, tmp_path):
        """Test generating JSON report."""
        from onnx2fx.operators import generate_operator_report
        import json
        
        output_path = tmp_path / "operators.json"
        report = generate_operator_report(sample_onnx_path, output_path, format="json")
        
        assert output_path.exists()
        
        # Should be valid JSON
        data = json.loads(output_path.read_text())
        assert "model_name" in data
        assert "operators" in data
    
    def test_report_to_dict(self, sample_onnx_path):
        """Test report serialization."""
        from onnx2fx.operators import scan_onnx_operators
        
        report = scan_onnx_operators(sample_onnx_path)
        d = report.to_dict()
        
        assert "model_name" in d
        assert "opset_version" in d
        assert "operators" in d
        assert "coverage_percent" in d


class TestCustomConverterRegistry:
    """Tests for custom converter registration."""
    
    def test_register_custom_converter(self):
        """Test registering a custom converter."""
        from onnx2fx.operators import register_custom_converter, get_custom_converters
        
        @register_custom_converter("TestCustomOp", versions=[1, 2])
        def my_converter(node, graph):
            return nn.Identity()
        
        converters = get_custom_converters()
        assert "TestCustomOp" in converters
        assert converters["TestCustomOp"]["versions"] == [1, 2]
