"""ONNX to PyTorch FX conversion module."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import onnx
import torch
import torch.nn as nn
from onnx2torch import convert as onnx2torch_convert
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import (
    OperationConverterResult, 
    onnx_mapping_from_node, 
    get_const_value
)
# Attempt to import OnnxReduceStaticAxes, handling potential path variations
try:
    from onnx2torch.node_converters.reduce import OnnxReduceStaticAxes
except ImportError:
    # Fallback or re-implementation if class is not public
    class OnnxReduceStaticAxes(nn.Module):
        def __init__(self, operation_type, axes, keepdims=1):
            super().__init__()
            self.operation_type = operation_type
            self.axes = sorted(axes) if axes else None
            self.keepdims = keepdims == 1
            
        def forward(self, x):
            if self.axes is None:
                return torch.mean(x, keepdim=self.keepdims) if self.operation_type == 'ReduceMean' else x
            return torch.mean(x, dim=self.axes, keepdim=self.keepdims)

logger = logging.getLogger(__name__)


# Workaround for ReduceMean opset 18
def _reduce_mean_18_converter(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    node_attributes = node.attributes
    axes = node_attributes.get("axes", None)
    keepdims = node_attributes.get("keepdims", 1)
    
    # In opset 18, axes is the second input
    if axes is None and len(node.input_values) > 1:
        try:
            val = get_const_value(node.input_values[1], graph)
            if isinstance(val, torch.Tensor):
                axes = val.tolist()
            else:
                axes = val
        except KeyError:
            pass

    from onnx2torch.utils.common import OnnxMapping

    return OperationConverterResult(
        torch_module=OnnxReduceStaticAxes(
            operation_type="ReduceMean",
            axes=axes,
            keepdims=keepdims,
        ),
        onnx_mapping=OnnxMapping(
            inputs=(node.input_values[0],),
            outputs=node.output_values,
        ),
    )

# Register ReduceMean safely
try:
    add_converter(operation_type="ReduceMean", version=18)(_reduce_mean_18_converter)
except ValueError:
    pass  # Already registered


# Workaround for Reshape allowzero=1 (opset 14+)
try:
    from onnx2torch.node_converters.reshape import OnnxReshape
except ImportError:
    class OnnxReshape(nn.Module):
        def forward(self, x, shape):
            shape_list = []
            for i, dim in enumerate(shape):
                if dim == 0:
                    shape_list.append(x.shape[i])
                else:
                    shape_list.append(dim)
            return torch.reshape(x, shape_list)

def _reshape_converter_allowzero(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    # We ignore allowzero=1 and assume standard behavior is intended or sufficient
    if node.attributes.get('allowzero', 0) == 1:
        pass 

    return OperationConverterResult(
        torch_module=OnnxReshape(),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )

# Register Reshape with overwrite for existing versions
from onnx2torch.node_converters.registry import _CONVERTER_REGISTRY, OperationDescription

for version in [5, 13, 14, 19]:
    description = OperationDescription(domain='', operation_type='Reshape', version=version)
    _CONVERTER_REGISTRY[description] = _reshape_converter_allowzero


class ConversionError(Exception):
    """Raised when ONNX to PyTorch conversion fails."""
    pass


class FxTraceError(Exception):
    """Raised when FX tracing fails on converted model."""
    pass


def load_onnx_model(path: Union[str, Path]) -> onnx.ModelProto:
    """
    Load and validate an ONNX model from file.
    
    Args:
        path: Path to the ONNX model file.
        
    Returns:
        Loaded ONNX ModelProto.
        
    Raises:
        FileNotFoundError: If model file doesn't exist.
        ConversionError: If model is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")
    
    logger.info(f"Loading ONNX model from {path}")
    model = onnx.load(str(path))
    
    try:
        onnx.checker.check_model(model)
        logger.info("ONNX model validation passed")
    except onnx.checker.ValidationError as e:
        raise ConversionError(f"Invalid ONNX model: {e}") from e
    
    return model


def convert(
    model_or_path: Union[str, Path, onnx.ModelProto],
    verify_fx_trace: bool = True,
) -> nn.Module:
    """
    Convert an ONNX model to PyTorch.
    
    This is the main entry point for conversion. It handles loading,
    conversion, and optional FX trace verification.
    
    Args:
        model_or_path: Either a path to ONNX file or loaded ModelProto.
        verify_fx_trace: If True, verify the model is FX-traceable.
        
    Returns:
        Converted PyTorch module.
        
    Raises:
        ConversionError: If conversion fails.
        FxTraceError: If FX tracing fails and verify_fx_trace is True.
    """
    converter = OnnxToFxConverter(model_or_path, verify_fx_trace=verify_fx_trace)
    return converter.convert()


class OnnxToFxConverter:
    """
    ONNX to FX-traceable PyTorch model converter.
    
    This class provides a configurable interface for converting ONNX models
    to PyTorch with FX traceability verification.
    
    Example:
        >>> converter = OnnxToFxConverter("model.onnx")
        >>> pytorch_model = converter.convert()
        >>> traced = torch.fx.symbolic_trace(pytorch_model)
    """
    
    def __init__(
        self,
        model_or_path: Union[str, Path, onnx.ModelProto],
        verify_fx_trace: bool = True,
        sample_input_shape: tuple[int, ...] | None = None,
    ):
        """
        Initialize the converter.
        
        Args:
            model_or_path: ONNX model path or loaded ModelProto.
            verify_fx_trace: Whether to verify FX traceability after conversion.
            sample_input_shape: Optional input shape for FX trace verification.
        """
        if isinstance(model_or_path, (str, Path)):
            self.onnx_model = load_onnx_model(model_or_path)
            self.model_path = Path(model_or_path)
        else:
            self.onnx_model = model_or_path
            self.model_path = None
            
        self.verify_fx_trace = verify_fx_trace
        self.sample_input_shape = sample_input_shape
        self._pytorch_model: nn.Module | None = None
        self._fx_graph: torch.fx.GraphModule | None = None
        
    @property
    def model_name(self) -> str:
        """Get model name from path or graph."""
        if self.model_path:
            return self.model_path.stem
        return self.onnx_model.graph.name or "unknown"
    
    def get_input_info(self) -> list[dict[str, Any]]:
        """
        Extract input tensor information from ONNX model.
        
        Returns:
            List of dicts with 'name', 'shape', and 'dtype' keys.
        """
        inputs = []
        for inp in self.onnx_model.graph.input:
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)  # Dynamic dimension
                else:
                    shape.append(-1)  # Unknown
                    
            dtype = inp.type.tensor_type.elem_type
            inputs.append({
                "name": inp.name,
                "shape": shape,
                "dtype": self._onnx_dtype_to_str(dtype),
            })
        return inputs
    
    def _onnx_dtype_to_str(self, dtype: int) -> str:
        """Convert ONNX dtype enum to string."""
        dtype_map = {
            1: "float32",
            2: "uint8", 
            3: "int8",
            4: "uint16",
            5: "int16",
            6: "int32",
            7: "int64",
            9: "bool",
            10: "float16",
            11: "float64",
        }
        return dtype_map.get(dtype, f"unknown_{dtype}")
    
    def convert(self) -> nn.Module:
        """
        Execute the conversion pipeline.
        
        Returns:
            Converted PyTorch module.
            
        Raises:
            ConversionError: If conversion fails.
            FxTraceError: If FX verification fails.
        """
        logger.info(f"Converting model: {self.model_name}")
        
        # Step 1: Convert using onnx2torch
        try:
            self._pytorch_model = onnx2torch_convert(self.onnx_model)
            self._pytorch_model.eval()
            logger.info("ONNX to PyTorch conversion successful")
        except Exception as e:
            raise ConversionError(f"Conversion failed: {e}") from e
        
        # Step 2: Verify FX traceability
        if self.verify_fx_trace:
            self._verify_fx_trace()
            
        return self._pytorch_model
    
    def _verify_fx_trace(self) -> None:
        """Verify the converted model is FX-traceable."""
        if self._pytorch_model is None:
            raise RuntimeError("Must call convert() first")
            
        logger.info("Verifying FX traceability...")
        try:
            # Try symbolic trace
            self._fx_graph = torch.fx.symbolic_trace(self._pytorch_model)
            logger.info("FX symbolic trace successful")
            
            # Validate graph structure
            self._fx_graph.graph.lint()
            logger.info("FX graph lint passed")
            
        except Exception as e:
            raise FxTraceError(
                f"Model is not FX-traceable: {e}. "
                "This may be due to dynamic control flow or unsupported operations."
            ) from e
    
    def get_fx_graph(self) -> torch.fx.GraphModule | None:
        """Get the FX graph if available."""
        return self._fx_graph
    
    def save(
        self,
        output_path: Union[str, Path],
        save_traced: bool = True,
    ) -> dict[str, Path]:
        """
        Save the converted model(s) to disk.
        
        Args:
            output_path: Directory to save models.
            save_traced: Whether to also save the FX-traced version.
            
        Returns:
            Dict mapping model type to saved path.
        """
        if self._pytorch_model is None:
            raise RuntimeError("Must call convert() first")
            
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        # Save PyTorch model
        model_path = output_path / f"{self.model_name}.pt"
        torch.save(self._pytorch_model.state_dict(), model_path)
        saved_paths["pytorch"] = model_path
        logger.info(f"Saved PyTorch model to {model_path}")
        
        # Save full model (for easier loading)
        full_model_path = output_path / f"{self.model_name}_full.pt"
        torch.save(self._pytorch_model, full_model_path)
        saved_paths["pytorch_full"] = full_model_path
        
        # Save FX traced if available
        if save_traced and self._fx_graph is not None:
            traced_path = output_path / f"{self.model_name}_traced.pt"
            torch.save(self._fx_graph, traced_path)
            saved_paths["fx_traced"] = traced_path
            logger.info(f"Saved FX-traced model to {traced_path}")
            
        return saved_paths
