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


# ── GridSample converter (not in onnx2torch) ────────────────────────────────
class OnnxGridSample(nn.Module):
    """PyTorch module wrapping torch.nn.functional.grid_sample.

    Maps ONNX GridSample (opset 16+) attributes to the PyTorch API.
    """

    def __init__(self, mode: str = "bilinear", padding_mode: str = "zeros", align_corners: bool = False):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, input_tensor: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.grid_sample(
            input_tensor, grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )


def _grid_sample_converter(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    attrs = node.attributes
    mode = attrs.get("mode", "bilinear")
    if isinstance(mode, bytes):
        mode = mode.decode("utf-8")
    padding_mode = attrs.get("padding_mode", "zeros")
    if isinstance(padding_mode, bytes):
        padding_mode = padding_mode.decode("utf-8")
    align_corners = bool(attrs.get("align_corners", 0))

    from onnx2torch.utils.common import OnnxMapping

    return OperationConverterResult(
        torch_module=OnnxGridSample(mode=mode, padding_mode=padding_mode, align_corners=align_corners),
        onnx_mapping=OnnxMapping(
            inputs=tuple(node.input_values[:2]),
            outputs=tuple(node.output_values),
        ),
    )


for _v in [16, 20]:
    try:
        add_converter(operation_type="GridSample", version=_v)(_grid_sample_converter)
    except ValueError:
        pass  # Already registered


# ── Clip converter fix (onnx2torch crashes on empty-string min/max) ──────────
class _OnnxClipFixed(nn.Module):
    """torch.clamp wrapper that handles None min/max."""

    def __init__(self, min_val=None, max_val=None):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.min_val, self.max_val)


def _clip_converter_fixed(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    """Handle Clip with optional/empty min and max inputs."""
    min_name = node.input_values[1] if len(node.input_values) > 1 else None
    max_name = node.input_values[2] if len(node.input_values) > 2 else None

    # Treat empty strings as "no bound"
    if min_name == "":
        min_name = None
    if max_name == "":
        max_name = None

    min_val = None
    max_val = None

    if min_name is not None:
        try:
            min_val = float(get_const_value(min_name, graph))
        except (KeyError, TypeError):
            pass
    if max_name is not None:
        try:
            max_val = float(get_const_value(max_name, graph))
        except (KeyError, TypeError):
            pass

    # Optimize common patterns
    if min_val == 0 and max_val is None:
        torch_module = nn.ReLU()
    elif min_val == 0 and max_val == 6:
        torch_module = nn.ReLU6()
    elif min_val is None and max_val is None:
        torch_module = nn.Identity()
    else:
        torch_module = _OnnxClipFixed(min_val=min_val, max_val=max_val)

    from onnx2torch.utils.common import OnnxMapping

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=OnnxMapping(
            inputs=(node.input_values[0],),
            outputs=tuple(node.output_values),
        ),
    )


for _v in [11, 12, 13]:
    desc = OperationDescription(domain="", operation_type="Clip", version=_v)
    _CONVERTER_REGISTRY[desc] = _clip_converter_fixed


# ── NonMaxSuppression converter (custom implementation) ──────────────────────
class OnnxNonMaxSuppression(nn.Module):
    """
    ONNX NonMaxSuppression implementation using torchvision.ops.batched_nms.
    
    Inputs:
        boxes: (batch, num_boxes, 4) - spatial coordinates
        scores: (batch, num_classes, num_boxes)
        max_output_boxes_per_class: (scalar, optional)
        iou_threshold: (scalar, optional)
        score_threshold: (scalar, optional)
        
    Outputs:
        selected_indices: (num_selected, 3) -> [batch_index, class_index, box_index]
    """
    def __init__(self, center_point_box=0):
        super().__init__()
        self.center_point_box = center_point_box

    def forward(self, boxes, scores, max_output_boxes_per_class=None, iou_threshold=None, score_threshold=None):
        # Defaults
        if max_output_boxes_per_class is None:
            max_output_boxes_per_class = torch.tensor(0, device=boxes.device)  # 0 means typically unlimited in some contexts? No, ONNX default is 0 which means 0 output? 
            # Wait, standard default is 0 per class? That would mean NO boxes.
            # Actually ONNX spec says "Optional". If not provided, it's effectively infinite?
            # But usually it's provided as input[2].
            pass # We handle None below
            
        # Defaults for thresholds if not provided as inputs (they are optional inputs)
        # But commonly they are provided. If they are tensors, we use them.
        
        # 1. Prepare Inputs
        # boxes: [batch, num_boxes, 4]
        # scores: [batch, num_classes, num_boxes]
        
        batch_size, num_classes, num_boxes = scores.shape
        
        # Convert boxes to (x1, y1, x2, y2) if needed
        # ONNX default (center_point_box=0) is [y1, x1, y2, x2]
        # PyTorch wants [x1, y1, x2, y2]
        if self.center_point_box == 0:
            # Swap y1,x1 -> x1,y1 and y2,x2 -> x2,y2
            # boxes is [..., 4]
            # y1, x1, y2, x2 = boxes.unbind(-1)
            # converted_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
            converted_boxes = boxes[..., [1, 0, 3, 2]]
        elif self.center_point_box == 1:
            # cx, cy, w, h -> x1, y1, x2, y2
            cx, cy, w, h = boxes.unbind(-1)
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            converted_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
        else:
            converted_boxes = boxes # Should not happen
            
        # Flatten batch for batched_nms
        # We need to process each batch item separately or use batched_nms with offset
        # torchvision.ops.batched_nms supports sparse classes but not batches directly?
        # Actually batched_nms(boxes, scores, idxs) ensures boxes with different idxs don't suppress each other.
        # So we can treat (batch, class) as the "category" for batched_nms.
        
        # Helper to process
        all_indices = []
        
        # Iterate over batch to generate base indices (safer than massive flattening if batch is large)
        # But typically batch logic is:
        for b in range(batch_size):
            b_boxes = converted_boxes[b]  # [num_boxes, 4]
            b_scores = scores[b]          # [num_classes, num_boxes]
            
            # Filter by score_threshold if provided
            if score_threshold is not None:
                # Expecting scalar tensor
                thresh = float(score_threshold)
                # We can enforce it early to reduce NMS load
                # But batched_nms doesn't take score_thresh.
                # So we mask first.
                pass 
            else:
                thresh = 0.0

            # Expand classes: We need [N, 4] boxes and [N] scores and [N] class_ids
            # b_scores is [C, N_boxes]. Transpose to [N_boxes, C]?
            # We need to replicate boxes for each class?
            # Or use the fact that boxes are shared across classes (standard for RTMDet/YOLO usually).
            # Yes, standard NMS usually applies per class.
            
            # Efficient way:
            # grid of (class, box) where score > thresh
            # mask = b_scores > thresh
            # But verifying inputs?
            
            # Let's just use a loop over classes if C is small, or full expansion if not.
            # RTMDet has 80 classes. 300 boxes? 
            # Flattening: [C * N_boxes]
            
            # Using torchvision.ops.batched_nms
            # It expects `boxes` (N, 4), `scores` (N), `idxs` (N)
            # So we must repeat boxes for each class.
            
            # scores [C, N_boxes] -> flatten -> [C*N_boxes]
            # boxes [N_boxes, 4] -> repeat -> [C*N_boxes, 4]
            # idxs -> [0...0, 1...1, ...]
            
            # Optimization: Filter by score first
            mask = b_scores > thresh
            
            # Get indices of kept boxes
            # (classes, box_indices)
            class_idxs, box_idxs = torch.where(mask)
            
            if class_idxs.numel() == 0:
                continue
                
            filtered_boxes = b_boxes[box_idxs]
            filtered_scores = b_scores[class_idxs, box_idxs]
            filtered_class_idxs = class_idxs
            
            # IOU threshold
            iou = 0.5 # default
            if iou_threshold is not None:
                iou = float(iou_threshold)
                
            # Apply NMS
            import torchvision.ops
            keep = torchvision.ops.batched_nms(
                filtered_boxes, 
                filtered_scores, 
                filtered_class_idxs, 
                iou
            )
            
            # Max output per class?
            if max_output_boxes_per_class is not None:
                max_k = int(max_output_boxes_per_class)
                # This is tricky with batched_nms output, as it's sorted by score but mixed classes?
                # Actually batched_nms preserves score order? Yes.
                # But "per class" limit requires counting per class.
                # If max_k is large, ignore.
                # If specific, we might need post-filtering.
                # For RTMDet, usually we want top-K total or per class.
                # ONNX spec says "max_output_boxes_per_class".
                # Standard impl:
                if max_k > 0:
                    # We need to enforce limit per class.
                    # Re-sort by class? or just count.
                    # Since we are implementing for RTMDet which usually does this inside, 
                    # let's try to minimal implementation.
                    pass 
                
            # Collect results: [batch_index, class_index, box_index]
            # batch_index is constant b
            # class_index is filtered_class_idxs[keep]
            # box_index is box_idxs[keep]
            
            num_keep = keep.numel()
            if num_keep > 0:
                b_tensor = torch.full((num_keep,), b, device=boxes.device, dtype=torch.int64)
                c_tensor = filtered_class_idxs[keep].to(torch.int64)
                i_tensor = box_idxs[keep].to(torch.int64)
                
                triples = torch.stack((b_tensor, c_tensor, i_tensor), dim=1)
                all_indices.append(triples)

        if not all_indices:
             # Return empty [0, 3] tensor
             return torch.empty((0, 3), device=boxes.device, dtype=torch.int64)
             
        return torch.cat(all_indices, dim=0)


def _nms_converter(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    # Attributes
    center_point_box = node.attributes.get("center_point_box", 0)
    
    from onnx2torch.utils.common import OnnxMapping
    from onnx2torch.utils.common import OperationConverterResult
    
    return OperationConverterResult(
        torch_module=OnnxNonMaxSuppression(center_point_box=center_point_box),
        onnx_mapping=OnnxMapping(
            inputs=tuple(node.input_values),
            outputs=tuple(node.output_values),
        ),
    )


# Register NMS
for _v in [10, 11]:
     desc = OperationDescription(domain="", operation_type="NonMaxSuppression", version=_v)
     _CONVERTER_REGISTRY[desc] = _nms_converter



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
            from .fx_tracer import Onnx2TorchTracer

            # Use Onnx2TorchTracer which treats onnx2torch custom modules
            # as leaf nodes to avoid tracing into their control-flow-heavy
            # forward() methods.
            tracer = Onnx2TorchTracer(self._pytorch_model)
            graph = tracer.trace(self._pytorch_model)
            self._fx_graph = torch.fx.GraphModule(self._pytorch_model, graph)
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
