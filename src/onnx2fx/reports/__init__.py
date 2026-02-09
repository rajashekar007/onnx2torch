"""Report generation for conversion, accuracy, and quantization results."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Union

logger = logging.getLogger(__name__)


@dataclass
class ModelReport:
    """Comprehensive report for a model conversion and quantization run."""
    
    model_name: str
    """Name of the model."""
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    """Timestamp of report generation."""
    
    # Conversion results
    conversion_status: str = "not_started"
    conversion_errors: list[str] = field(default_factory=list)
    fx_traceable: bool = False
    
    # Input/output info
    input_info: list[dict[str, Any]] = field(default_factory=list)
    output_info: list[dict[str, Any]] = field(default_factory=list)
    
    # Accuracy validation
    accuracy_metrics: dict[str, Any] = field(default_factory=dict)
    accuracy_passed: bool = False
    
    # Operator analysis
    operator_coverage: float = 0.0
    unsupported_operators: list[str] = field(default_factory=list)
    
    # PTQ results
    ptq_status: str = "not_started"
    ptq_outputs: dict[str, str] = field(default_factory=dict)
    ptq_errors: list[str] = field(default_factory=list)
    
    # QFT results
    qft_status: str = "not_started"
    qft_outputs: dict[str, str] = field(default_factory=dict)
    qft_training_metrics: dict[str, Any] = field(default_factory=dict)
    qft_errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "conversion": {
                "status": self.conversion_status,
                "fx_traceable": self.fx_traceable,
                "errors": self.conversion_errors,
            },
            "io_info": {
                "inputs": self.input_info,
                "outputs": self.output_info,
            },
            "accuracy": {
                "metrics": self.accuracy_metrics,
                "passed": self.accuracy_passed,
            },
            "operators": {
                "coverage_percent": self.operator_coverage,
                "unsupported": self.unsupported_operators,
            },
            "ptq": {
                "status": self.ptq_status,
                "outputs": self.ptq_outputs,
                "errors": self.ptq_errors,
            },
            "qft": {
                "status": self.qft_status,
                "outputs": self.qft_outputs,
                "training_metrics": self.qft_training_metrics,
                "errors": self.qft_errors,
            },
        }
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Model Report: {self.model_name}",
            "",
            f"*Generated: {self.timestamp}*",
            "",
            "---",
            "",
            "## Summary",
            "",
            "| Stage | Status |",
            "|-------|--------|",
            f"| Conversion | {self._status_emoji(self.conversion_status)} {self.conversion_status} |",
            f"| FX Traceable | {'✓' if self.fx_traceable else '✗'} |",
            f"| Accuracy | {'✓ PASSED' if self.accuracy_passed else '✗ FAILED'} |",
            f"| PTQ | {self._status_emoji(self.ptq_status)} {self.ptq_status} |",
            f"| QFT | {self._status_emoji(self.qft_status)} {self.qft_status} |",
            "",
        ]
        
        # Conversion details
        lines.extend([
            "## Conversion",
            "",
        ])
        
        if self.conversion_errors:
            lines.append("**Errors:**")
            for err in self.conversion_errors:
                lines.append(f"- {err}")
            lines.append("")
        
        # Accuracy details
        if self.accuracy_metrics:
            lines.extend([
                "## Accuracy Validation",
                "",
                "| Metric | Value |",
                "|--------|-------|",
            ])
            for key, value in self.accuracy_metrics.items():
                if isinstance(value, float):
                    lines.append(f"| {key} | {value:.6f} |")
                else:
                    lines.append(f"| {key} | {value} |")
            lines.append("")
        
        # Operator coverage
        if self.unsupported_operators:
            lines.extend([
                "## Operator Coverage",
                "",
                f"Coverage: {self.operator_coverage:.1f}%",
                "",
                "**Unsupported Operators:**",
            ])
            for op in self.unsupported_operators:
                lines.append(f"- `{op}`")
            lines.append("")
        
        # PTQ outputs
        if self.ptq_outputs:
            lines.extend([
                "## PTQ Outputs",
                "",
            ])
            for name, path in self.ptq_outputs.items():
                lines.append(f"- **{name}**: `{path}`")
            lines.append("")
        
        # QFT outputs
        if self.qft_outputs:
            lines.extend([
                "## QFT Outputs",
                "",
            ])
            for name, path in self.qft_outputs.items():
                lines.append(f"- **{name}**: `{path}`")
            
            if self.qft_training_metrics:
                lines.extend(["", "**Training Metrics:**"])
                for key, value in self.qft_training_metrics.items():
                    lines.append(f"- {key}: {value}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _status_emoji(self, status: str) -> str:
        """Get emoji for status."""
        if status == "success":
            return "✓"
        elif status == "failed":
            return "✗"
        elif status == "in_progress":
            return "⏳"
        return "○"
    
    def save(
        self,
        output_path: Union[str, Path],
        format: str = "both",
    ) -> dict[str, Path]:
        """
        Save report to file(s).
        
        Args:
            output_path: Base path (without extension).
            format: 'json', 'markdown', or 'both'.
            
        Returns:
            Dict of format to path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        saved = {}
        
        if format in ("json", "both"):
            json_path = output_path.with_suffix(".json")
            json_path.write_text(json.dumps(self.to_dict(), indent=2))
            saved["json"] = json_path
            logger.info(f"Saved JSON report to {json_path}")
        
        if format in ("markdown", "both"):
            md_path = output_path.with_suffix(".md")
            md_path.write_text(self.to_markdown())
            saved["markdown"] = md_path
            logger.info(f"Saved markdown report to {md_path}")
        
        return saved


class ReportGenerator:
    """Generator for model conversion and quantization reports."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports: dict[str, ModelReport] = {}
    
    def create_report(self, model_name: str) -> ModelReport:
        """Create a new report for a model."""
        report = ModelReport(model_name=model_name)
        self.reports[model_name] = report
        return report
    
    def get_report(self, model_name: str) -> ModelReport | None:
        """Get existing report for a model."""
        return self.reports.get(model_name)
    
    def save_all(self, format: str = "both") -> dict[str, dict[str, Path]]:
        """
        Save all reports.
        
        Args:
            format: Output format.
            
        Returns:
            Dict mapping model name to saved paths.
        """
        saved = {}
        for name, report in self.reports.items():
            output_path = self.output_dir / f"{name}_report"
            saved[name] = report.save(output_path, format)
        return saved
    
    def generate_summary(self) -> str:
        """Generate a summary report of all models."""
        lines = [
            "# Batch Conversion Summary",
            "",
            f"*Generated: {datetime.now().isoformat()}*",
            "",
            f"**Total Models:** {len(self.reports)}",
            "",
            "## Results Overview",
            "",
            "| Model | Conversion | FX Trace | Accuracy | PTQ | QFT |",
            "|-------|------------|----------|----------|-----|-----|",
        ]
        
        for name, report in self.reports.items():
            conv = "✓" if report.conversion_status == "success" else "✗"
            fx = "✓" if report.fx_traceable else "✗"
            acc = "✓" if report.accuracy_passed else "✗"
            ptq = "✓" if report.ptq_status == "success" else ("✗" if report.ptq_status == "failed" else "-")
            qft = "✓" if report.qft_status == "success" else ("✗" if report.qft_status == "failed" else "-")
            lines.append(f"| {name} | {conv} | {fx} | {acc} | {ptq} | {qft} |")
        
        lines.extend([
            "",
            "---",
            "",
            "## Statistics",
            "",
        ])
        
        # Calculate stats
        n = len(self.reports)
        if n > 0:
            conv_success = sum(1 for r in self.reports.values() if r.conversion_status == "success")
            fx_success = sum(1 for r in self.reports.values() if r.fx_traceable)
            acc_passed = sum(1 for r in self.reports.values() if r.accuracy_passed)
            
            lines.extend([
                f"- Conversion Success: {conv_success}/{n} ({100*conv_success/n:.1f}%)",
                f"- FX Traceable: {fx_success}/{n} ({100*fx_success/n:.1f}%)",
                f"- Accuracy Passed: {acc_passed}/{n} ({100*acc_passed/n:.1f}%)",
            ])
        
        return "\n".join(lines)
    
    def save_summary(self, filename: str = "summary") -> Path:
        """Save the summary report."""
        summary_path = self.output_dir / f"{filename}.md"
        summary_path.write_text(self.generate_summary())
        logger.info(f"Saved summary to {summary_path}")
        return summary_path
