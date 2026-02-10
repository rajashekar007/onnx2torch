"""FX traceability utilities for converted PyTorch models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Union

import torch
import torch.fx as fx
import torch.nn as nn

logger = logging.getLogger(__name__)


class FxTraceError(Exception):
    """Raised when FX tracing fails."""
    pass


def _collect_onnx2torch_leaf_types(model: nn.Module) -> set[type]:
    """Collect all onnx2torch module types used in a model.

    These modules contain dynamic control flow in their ``forward()``
    that breaks ``torch.fx.symbolic_trace``.  Treating them as leaf
    modules lets FX trace around them.
    """
    leaf_types: set[type] = set()
    for _name, mod in model.named_modules():
        if "onnx2torch" in type(mod).__module__:
            leaf_types.add(type(mod))
    return leaf_types


class Onnx2TorchTracer(fx.Tracer):
    """FX Tracer that treats onnx2torch converter modules as leaf nodes.

    onnx2torch generates custom ``nn.Module`` subclasses (e.g.
    ``OnnxResize``, ``OnnxSlice``, ``OnnxConstant``) whose ``forward``
    methods use Python control flow on tensor values, type checks, and
    other patterns that ``symbolic_trace`` cannot handle.

    This tracer automatically detects all such module types in the model
    and marks them as leaves so that FX builds graph edges *around* them
    rather than trying to inline their implementations.
    """

    def __init__(self, model: nn.Module | None = None, extra_leaf_types: set[type] | None = None):
        super().__init__()
        self._leaf_types: set[type] = set(extra_leaf_types or set())
        if model is not None:
            self._leaf_types |= _collect_onnx2torch_leaf_types(model)

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if type(m) in self._leaf_types:
            return True
        # Also treat any onnx2torch module as leaf even if not pre-collected
        if "onnx2torch" in type(m).__module__:
            return True
        return super().is_leaf_module(m, module_qualified_name)


def trace_model(
    model: nn.Module,
    concrete_args: dict[str, Any] | None = None,
    tracer_class: type[fx.Tracer] | None = None,
    auto_leaf: bool = True,
) -> fx.GraphModule:
    """
    Trace a PyTorch model using FX symbolic trace.
    
    Args:
        model: PyTorch model to trace.
        concrete_args: Optional dict of concrete values for some arguments.
        tracer_class: Optional custom Tracer class to use.
        auto_leaf: If True and no tracer_class is given, automatically use
            ``Onnx2TorchTracer`` when the model contains onnx2torch modules.
        
    Returns:
        FX GraphModule containing the traced graph.
        
    Raises:
        FxTraceError: If tracing fails.
    """
    model.eval()
    
    try:
        if tracer_class is not None:
            tracer = tracer_class()
            graph = tracer.trace(model, concrete_args)
            traced = fx.GraphModule(model, graph)
        elif auto_leaf and _collect_onnx2torch_leaf_types(model):
            tracer = Onnx2TorchTracer(model)
            graph = tracer.trace(model, concrete_args)
            traced = fx.GraphModule(model, graph)
            logger.info("FX trace successful (Onnx2TorchTracer with leaf modules)")
        else:
            traced = fx.symbolic_trace(model, concrete_args)
            
        logger.info("FX symbolic trace successful")
        return traced
        
    except Exception as e:
        raise FxTraceError(f"Failed to trace model: {e}") from e


def validate_fx_graph(
    traced_model: fx.GraphModule,
    sample_input: torch.Tensor | None = None,
) -> dict[str, Any]:
    """
    Validate an FX-traced graph structure and optionally run a forward pass.
    
    Args:
        traced_model: FX GraphModule to validate.
        sample_input: Optional sample input to test forward pass.
        
    Returns:
        Dict containing validation results.
    """
    results = {
        "valid": True,
        "num_nodes": 0,
        "node_types": {},
        "errors": [],
        "warnings": [],
    }
    
    # Lint the graph
    try:
        traced_model.graph.lint()
        logger.info("Graph lint passed")
    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Graph lint failed: {e}")
    
    # Count nodes by type
    for node in traced_model.graph.nodes:
        results["num_nodes"] += 1
        op = node.op
        results["node_types"][op] = results["node_types"].get(op, 0) + 1
    
    # Check for potentially problematic patterns
    for node in traced_model.graph.nodes:
        # Check for in-place operations (can cause issues)
        if hasattr(node, 'target') and isinstance(node.target, str):
            if node.target.endswith('_'):
                results["warnings"].append(
                    f"In-place operation detected: {node.target}"
                )
    
    # Test forward pass if input provided
    if sample_input is not None:
        try:
            traced_model.eval()
            with torch.no_grad():
                _ = traced_model(sample_input)
            logger.info("Forward pass test successful")
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Forward pass failed: {e}")
    
    return results


def export_fx_graph(
    traced_model: fx.GraphModule,
    output_path: Union[str, Path],
    format: str = "code",
) -> Path:
    """
    Export FX graph for inspection and debugging.
    
    Args:
        traced_model: FX GraphModule to export.
        output_path: Path to save the export.
        format: Export format - 'code', 'graph', or 'both'.
        
    Returns:
        Path to the primary exported file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format in ("code", "both"):
        code_path = output_path.with_suffix(".py")
        code = traced_model.code
        code_path.write_text(code)
        logger.info(f"Exported code to {code_path}")
    
    if format in ("graph", "both"):
        graph_path = output_path.with_suffix(".txt")
        graph_str = str(traced_model.graph)
        graph_path.write_text(graph_str)
        logger.info(f"Exported graph to {graph_path}")
    
    return output_path


def print_graph_tabular(traced_model: fx.GraphModule) -> str:
    """
    Generate a tabular representation of the FX graph.
    
    Args:
        traced_model: FX GraphModule to display.
        
    Returns:
        Formatted table string.
    """
    traced_model.graph.print_tabular()
    return ""


class CustomTracer(fx.Tracer):
    """
    Custom FX Tracer with configurable leaf modules.
    
    Use this when certain modules should not be traced into
    (treated as leaf/atomic operations).
    """
    
    def __init__(
        self,
        leaf_modules: list[type[nn.Module]] | None = None,
        leaf_functions: list[Callable] | None = None,
    ):
        """
        Initialize custom tracer.
        
        Args:
            leaf_modules: Module types that should not be traced into.
            leaf_functions: Functions that should not be traced into.
        """
        super().__init__()
        self.leaf_modules = set(leaf_modules or [])
        self.leaf_functions = set(leaf_functions or [])
    
    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        """Determine if module should be treated as a leaf."""
        if type(m) in self.leaf_modules:
            return True
        return super().is_leaf_module(m, module_qualified_name)


def make_fx_compatible(
    model: nn.Module,
    fix_inplace: bool = True,
    fix_dynamic_shapes: bool = True,
) -> nn.Module:
    """
    Attempt to make a model FX-compatible by fixing common issues.
    
    Args:
        model: Model to fix.
        fix_inplace: Replace in-place operations.
        fix_dynamic_shapes: Replace dynamic shape operations.
        
    Returns:
        Modified model (may be the same instance).
    """
    # This is a basic implementation - complex models may need
    # custom handling
    
    if fix_inplace:
        # Replace common in-place activations
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU) and module.inplace:
                setattr_nested(model, name, nn.ReLU(inplace=False))
            elif isinstance(module, nn.LeakyReLU) and module.inplace:
                setattr_nested(model, name, nn.LeakyReLU(
                    module.negative_slope, inplace=False
                ))
    
    return model


def setattr_nested(module: nn.Module, name: str, value: Any) -> None:
    """Set a nested attribute on a module by dotted name."""
    parts = name.split('.')
    for part in parts[:-1]:
        module = getattr(module, part)
    setattr(module, parts[-1], value)


def get_graph_stats(traced_model: fx.GraphModule) -> dict[str, Any]:
    """
    Get statistics about an FX graph.
    
    Args:
        traced_model: FX GraphModule to analyze.
        
    Returns:
        Dict with graph statistics.
    """
    stats = {
        "total_nodes": 0,
        "operations_by_type": {},
        "call_modules": [],
        "call_functions": [],
        "placeholders": [],
        "outputs": [],
    }
    
    for node in traced_model.graph.nodes:
        stats["total_nodes"] += 1
        
        op_count = stats["operations_by_type"]
        op_count[node.op] = op_count.get(node.op, 0) + 1
        
        if node.op == "call_module":
            stats["call_modules"].append(str(node.target))
        elif node.op == "call_function":
            func_name = getattr(node.target, "__name__", str(node.target))
            stats["call_functions"].append(func_name)
        elif node.op == "placeholder":
            stats["placeholders"].append(node.name)
        elif node.op == "output":
            stats["outputs"].append(node.name)
    
    return stats
