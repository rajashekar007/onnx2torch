"""Unit tests for the converter module."""

import pytest
import torch
import torch.nn as nn
import onnx
from pathlib import Path
import tempfile


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.flatten(1)
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


class TestOnnxToFxConverter:
    """Tests for OnnxToFxConverter class."""
    
    def test_load_onnx_model(self, sample_onnx_path):
        """Test loading ONNX model."""
        from onnx2fx.converter import load_onnx_model
        
        model = load_onnx_model(sample_onnx_path)
        assert model is not None
        assert isinstance(model, onnx.ModelProto)
    
    def test_load_nonexistent_model(self):
        """Test loading non-existent model raises error."""
        from onnx2fx.converter import load_onnx_model
        
        with pytest.raises(FileNotFoundError):
            load_onnx_model("/nonexistent/model.onnx")
    
    def test_convert_basic(self, sample_onnx_path):
        """Test basic conversion."""
        from onnx2fx.converter import convert
        
        pytorch_model = convert(sample_onnx_path, verify_fx_trace=True)
        
        assert pytorch_model is not None
        assert isinstance(pytorch_model, nn.Module)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 32, 32)
        output = pytorch_model(dummy_input)
        assert output.shape == (1, 10)
    
    def test_converter_class(self, sample_onnx_path):
        """Test OnnxToFxConverter class interface."""
        from onnx2fx.converter import OnnxToFxConverter
        
        converter = OnnxToFxConverter(sample_onnx_path)
        
        # Check model name
        assert converter.model_name == sample_onnx_path.stem
        
        # Get input info
        input_info = converter.get_input_info()
        assert len(input_info) == 1
        assert input_info[0]["name"] == "input"
        
        # Convert
        pytorch_model = converter.convert()
        assert pytorch_model is not None
        
        # Check FX graph
        fx_graph = converter.get_fx_graph()
        assert fx_graph is not None
    
    def test_save_converted_model(self, sample_onnx_path, tmp_path):
        """Test saving converted model."""
        from onnx2fx.converter import OnnxToFxConverter
        
        converter = OnnxToFxConverter(sample_onnx_path)
        converter.convert()
        
        saved_paths = converter.save(tmp_path)
        
        assert "pytorch" in saved_paths
        assert "pytorch_full" in saved_paths
        assert saved_paths["pytorch"].exists()
        assert saved_paths["pytorch_full"].exists()
    
    def test_skip_fx_verification(self, sample_onnx_path):
        """Test conversion without FX verification."""
        from onnx2fx.converter import OnnxToFxConverter
        
        converter = OnnxToFxConverter(sample_onnx_path, verify_fx_trace=False)
        pytorch_model = converter.convert()
        
        assert pytorch_model is not None
        assert converter.get_fx_graph() is None  # Not traced


class TestFxTraceability:
    """Tests for FX tracing functionality."""
    
    def test_trace_converted_model(self, sample_onnx_path):
        """Test FX tracing of converted model."""
        from onnx2fx.converter import convert
        from onnx2fx.fx_tracer import trace_model, validate_fx_graph
        
        pytorch_model = convert(sample_onnx_path, verify_fx_trace=False)
        traced = trace_model(pytorch_model)
        
        assert traced is not None
        
        # Validate graph
        results = validate_fx_graph(traced)
        assert results["valid"]
    
    def test_graph_stats(self, sample_onnx_path):
        """Test getting graph statistics."""
        from onnx2fx.converter import convert
        from onnx2fx.fx_tracer import trace_model, get_graph_stats
        
        pytorch_model = convert(sample_onnx_path, verify_fx_trace=False)
        traced = trace_model(pytorch_model)
        
        stats = get_graph_stats(traced)
        
        assert stats["total_nodes"] > 0
        assert "placeholder" in stats["operations_by_type"]
        assert "output" in stats["operations_by_type"]
