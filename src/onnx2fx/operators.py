"""ONNX operator analysis and gap detection."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Union

import onnx

logger = logging.getLogger(__name__)

# Known supported operators from onnx2torch
# This is a subset - full list at https://github.com/ENOT-AutoDL/onnx2torch/blob/main/operators.md
KNOWN_SUPPORTED_OPS = {
    # Activation functions
    "Relu", "LeakyRelu", "Sigmoid", "Tanh", "Softmax", "LogSoftmax",
    "Elu", "Selu", "PRelu", "Gelu", "HardSigmoid", "HardSwish", "Mish",
    
    # Convolution and Pooling
    "Conv", "ConvTranspose", "MaxPool", "AveragePool", "GlobalAveragePool",
    "GlobalMaxPool", "LpPool",
    
    # Normalization
    "BatchNormalization", "InstanceNormalization", "LayerNormalization",
    "GroupNormalization", "LRN",
    
    # Element-wise operations
    "Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "Exp", "Log", "Abs",
    "Neg", "Floor", "Ceil", "Round", "Sign", "Clip", "Min", "Max",
    "Sum", "Mean", "Erf",
    
    # Reduction operations
    "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd",
    "ReduceL1", "ReduceL2",
    
    # Shape operations
    "Reshape", "Transpose", "Flatten", "Squeeze", "Unsqueeze",
    "Concat", "Split", "Slice", "Gather", "Pad", "Tile", "Expand",
    
    # Matrix operations
    "MatMul", "Gemm", "Einsum",
    
    # Comparison
    "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "Equal", "Not",
    "And", "Or", "Where",
    
    # Other common ops
    "Dropout", "Identity", "Cast", "Constant", "Shape", "Size",
    "Resize", "Upsample", "ConstantOfShape", "Range",
    "Softplus", "Softsign", "ThresholdedRelu",
    
    # Attention / Transformer
    "Attention", "MultiHeadAttention",
}


@dataclass
class OperatorInfo:
    """Information about an ONNX operator usage."""
    
    op_type: str
    """ONNX operator type name."""
    
    count: int = 1
    """Number of occurrences in the model."""
    
    versions: set[int] = field(default_factory=set)
    """Opset versions used."""
    
    node_names: list[str] = field(default_factory=list)
    """Names of nodes using this operator."""
    
    supported: bool | None = None
    """Whether this operator is supported by onnx2torch."""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "op_type": self.op_type,
            "count": self.count,
            "versions": list(self.versions),
            "node_names": self.node_names[:5],  # Limit for readability
            "supported": self.supported,
        }


@dataclass
class OperatorReport:
    """Report of ONNX operators in a model."""
    
    model_name: str
    """Name of the analyzed model."""
    
    opset_version: int
    """Primary opset version of the model."""
    
    operators: dict[str, OperatorInfo] = field(default_factory=dict)
    """Map of operator type to info."""
    
    total_nodes: int = 0
    """Total number of nodes in the graph."""
    
    @property
    def supported_ops(self) -> list[str]:
        """List of supported operator types."""
        return [op for op, info in self.operators.items() if info.supported]
    
    @property
    def unsupported_ops(self) -> list[str]:
        """List of unsupported operator types."""
        return [op for op, info in self.operators.items() if info.supported is False]
    
    @property
    def unknown_ops(self) -> list[str]:
        """List of operators with unknown support status."""
        return [op for op, info in self.operators.items() if info.supported is None]
    
    @property
    def coverage_percent(self) -> float:
        """Percentage of operators that are supported."""
        total = len(self.operators)
        if total == 0:
            return 100.0
        supported = len(self.supported_ops)
        return (supported / total) * 100
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "opset_version": self.opset_version,
            "total_nodes": self.total_nodes,
            "total_unique_ops": len(self.operators),
            "supported_count": len(self.supported_ops),
            "unsupported_count": len(self.unsupported_ops),
            "unknown_count": len(self.unknown_ops),
            "coverage_percent": self.coverage_percent,
            "operators": {k: v.to_dict() for k, v in self.operators.items()},
        }
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Operator Analysis: {self.model_name}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Opset Version | {self.opset_version} |",
            f"| Total Nodes | {self.total_nodes} |",
            f"| Unique Operators | {len(self.operators)} |",
            f"| Supported | {len(self.supported_ops)} |",
            f"| Unsupported | {len(self.unsupported_ops)} |",
            f"| Unknown | {len(self.unknown_ops)} |",
            f"| Coverage | {self.coverage_percent:.1f}% |",
            "",
        ]
        
        if self.unsupported_ops:
            lines.extend([
                "## ⚠️ Unsupported Operators",
                "",
                "| Operator | Count | Nodes |",
                "|----------|-------|-------|",
            ])
            for op in sorted(self.unsupported_ops):
                info = self.operators[op]
                nodes = ", ".join(info.node_names[:3])
                if len(info.node_names) > 3:
                    nodes += "..."
                lines.append(f"| {op} | {info.count} | {nodes} |")
            lines.append("")
        
        lines.extend([
            "## ✓ Supported Operators",
            "",
            "| Operator | Count |",
            "|----------|-------|",
        ])
        for op in sorted(self.supported_ops):
            info = self.operators[op]
            lines.append(f"| {op} | {info.count} |")
        
        return "\n".join(lines)


def scan_onnx_operators(
    model_or_path: Union[str, Path, onnx.ModelProto],
) -> OperatorReport:
    """
    Scan an ONNX model and extract all operators used.
    
    Args:
        model_or_path: ONNX model path or loaded ModelProto.
        
    Returns:
        OperatorReport with operator information.
    """
    # Load model if path
    if isinstance(model_or_path, (str, Path)):
        model = onnx.load(str(model_or_path))
        model_name = Path(model_or_path).stem
    else:
        model = model_or_path
        model_name = model.graph.name or "unknown"
    
    # Get opset version
    opset_version = 0
    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":
            opset_version = opset.version
            break
    
    # Scan nodes
    operators: dict[str, OperatorInfo] = {}
    total_nodes = 0
    
    def scan_graph(graph):
        nonlocal total_nodes
        for node in graph.node:
            total_nodes += 1
            op_type = node.op_type
            
            if op_type not in operators:
                operators[op_type] = OperatorInfo(op_type=op_type)
            
            info = operators[op_type]
            info.count += 1
            info.versions.add(opset_version)
            if node.name:
                info.node_names.append(node.name)
            
            # Scan subgraphs (for control flow ops)
            for attr in node.attribute:
                if attr.g:  # Has a graph attribute
                    scan_graph(attr.g)
    
    scan_graph(model.graph)
    
    # Check support status
    for op_type, info in operators.items():
        info.supported = check_operator_support(op_type)
    
    return OperatorReport(
        model_name=model_name,
        opset_version=opset_version,
        operators=operators,
        total_nodes=total_nodes,
    )


def check_operator_support(op_type: str) -> bool | None:
    """
    Check if an ONNX operator is supported by onnx2torch.
    
    Args:
        op_type: ONNX operator type name.
        
    Returns:
        True if supported, False if known unsupported, None if unknown.
    """
    if op_type in KNOWN_SUPPORTED_OPS:
        return True
    
    # Some ops are definitely not supported
    KNOWN_UNSUPPORTED = {
        "Loop", "If", "Scan",  # Control flow
        "SequenceConstruct", "SequenceAt", "SequenceEmpty",  # Sequences
        "NonMaxSuppression",  # Detection-specific
    }
    
    if op_type in KNOWN_UNSUPPORTED:
        return False
    
    return None  # Unknown


def generate_operator_report(
    model_or_path: Union[str, Path, onnx.ModelProto],
    output_path: Union[str, Path] | None = None,
    format: str = "markdown",
) -> OperatorReport:
    """
    Generate a detailed operator analysis report.
    
    Args:
        model_or_path: ONNX model to analyze.
        output_path: Optional path to save report.
        format: Output format - 'markdown' or 'json'.
        
    Returns:
        OperatorReport with analysis results.
    """
    report = scan_onnx_operators(model_or_path)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            output_path.write_text(json.dumps(report.to_dict(), indent=2))
        else:
            output_path.write_text(report.to_markdown())
            
        logger.info(f"Saved operator report to {output_path}")
    
    return report


# Custom converter registration for extending onnx2torch
_CUSTOM_CONVERTERS: dict[str, Callable] = {}


def register_custom_converter(op_type: str, versions: list[int] | None = None):
    """
    Decorator to register a custom operator converter.
    
    This allows extending onnx2torch with custom operator implementations.
    
    Example:
        @register_custom_converter("CustomOp", versions=[1, 2])
        def convert_custom_op(node, graph):
            return MyCustomModule()
    
    Args:
        op_type: ONNX operator type to handle.
        versions: Optional list of opset versions to support.
    """
    def decorator(func: Callable):
        key = op_type
        _CUSTOM_CONVERTERS[key] = {
            "func": func,
            "versions": versions or [],
        }
        logger.info(f"Registered custom converter for {op_type}")
        return func
    return decorator


def get_custom_converters() -> dict[str, Any]:
    """Get all registered custom converters."""
    return _CUSTOM_CONVERTERS.copy()
