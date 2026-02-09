"""Command-line interface for ONNX to FX converter."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(
    name="onnx2fx",
    help="Convert ONNX models to FX-traceable PyTorch models with PTQ/QFT support.",
    add_completion=False,
)
console = Console()


def setup_logging(verbose: bool = False):
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.command()
def convert(
    model_path: Path = typer.Argument(..., help="Path to ONNX model file"),
    output_dir: Path = typer.Option(
        Path("./output"), "-o", "--output", help="Output directory"
    ),
    skip_fx_verify: bool = typer.Option(
        False, "--skip-fx-verify", help="Skip FX traceability verification"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
):
    """
    Convert an ONNX model to FX-traceable PyTorch.
    
    Example:
        onnx2fx convert model.onnx -o output/
    """
    setup_logging(verbose)
    
    from onnx2fx.converter import OnnxToFxConverter, ConversionError, FxTraceError
    
    console.print(Panel(f"[bold blue]Converting: {model_path.name}[/bold blue]"))
    
    if not model_path.exists():
        console.print(f"[red]Error: Model not found: {model_path}[/red]")
        raise typer.Exit(1)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading model...", total=None)
            
            converter = OnnxToFxConverter(
                model_path,
                verify_fx_trace=not skip_fx_verify,
            )
            
            progress.update(task, description="Converting to PyTorch...")
            pytorch_model = converter.convert()
            
            progress.update(task, description="Saving model...")
            saved = converter.save(output_dir)
        
        # Print results
        table = Table(title="Conversion Results")
        table.add_column("Output", style="cyan")
        table.add_column("Path", style="green")
        
        for name, path in saved.items():
            table.add_row(name, str(path))
        
        console.print(table)
        console.print("[bold green]✓ Conversion successful![/bold green]")
        
    except ConversionError as e:
        console.print(f"[red]Conversion failed: {e}[/red]")
        raise typer.Exit(1)
    except FxTraceError as e:
        console.print(f"[yellow]Warning: FX trace failed: {e}[/yellow]")
        console.print("Model was converted but is not FX-traceable.")
        raise typer.Exit(2)


@app.command()
def validate(
    onnx_path: Path = typer.Argument(..., help="Path to ONNX model"),
    pytorch_path: Path = typer.Argument(..., help="Path to converted PyTorch model"),
    tolerance: float = typer.Option(
        1.0, "-t", "--tolerance", help="Max relative error tolerance (%)"
    ),
    samples: int = typer.Option(
        5, "-n", "--samples", help="Number of validation samples"
    ),
    report: Optional[Path] = typer.Option(
        None, "-r", "--report", help="Path to save validation report"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
):
    """
    Validate accuracy between ONNX and converted PyTorch model.
    
    Example:
        onnx2fx validate model.onnx converted.pt --tolerance 1.0
    """
    setup_logging(verbose)
    
    import torch
    from onnx2fx.validator import AccuracyValidator
    
    console.print(Panel("[bold blue]Validating Model Accuracy[/bold blue]"))
    
    # Load PyTorch model
    try:
        pytorch_model = torch.load(pytorch_path)
    except Exception as e:
        console.print(f"[red]Failed to load PyTorch model: {e}[/red]")
        raise typer.Exit(1)
    
    # Run validation
    validator = AccuracyValidator(onnx_path, pytorch_model, tolerance=tolerance)
    metrics = validator.validate(num_samples=samples)
    
    # Display results
    table = Table(title="Accuracy Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("MSE", f"{metrics.mse:.6e}")
    table.add_row("Max Diff", f"{metrics.max_diff:.6e}")
    table.add_row("Mean Diff", f"{metrics.mean_diff:.6e}")
    table.add_row("Correlation", f"{metrics.correlation:.6f}")
    table.add_row("Relative Error", f"{metrics.relative_error:.4f}%")
    table.add_row("Tolerance", f"{metrics.tolerance:.4f}%")
    
    console.print(table)
    
    if metrics.passed:
        console.print("[bold green]✓ Validation PASSED[/bold green]")
    else:
        console.print("[bold red]✗ Validation FAILED[/bold red]")
    
    if report:
        report_content = validator.generate_report(report)
        console.print(f"Report saved to: {report}")
    
    raise typer.Exit(0 if metrics.passed else 1)


@app.command()
def analyze(
    model_path: Path = typer.Argument(..., help="Path to ONNX model"),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Path to save operator report"
    ),
    format: str = typer.Option(
        "markdown", "-f", "--format", help="Output format (markdown/json)"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
):
    """
    Analyze ONNX operators and check conversion support.
    
    Example:
        onnx2fx analyze model.onnx --output report.md
    """
    setup_logging(verbose)
    
    from onnx2fx.operators import generate_operator_report
    
    console.print(Panel("[bold blue]Analyzing ONNX Operators[/bold blue]"))
    
    report = generate_operator_report(model_path, output, format)
    
    # Display summary
    table = Table(title="Operator Analysis")
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="white")
    
    table.add_row("Total Nodes", str(report.total_nodes))
    table.add_row("Unique Operators", str(len(report.operators)))
    table.add_row("Supported", str(len(report.supported_ops)))
    table.add_row("Unsupported", str(len(report.unsupported_ops)))
    table.add_row("Unknown", str(len(report.unknown_ops)))
    table.add_row("Coverage", f"{report.coverage_percent:.1f}%")
    
    console.print(table)
    
    if report.unsupported_ops:
        console.print("\n[yellow]Unsupported Operators:[/yellow]")
        for op in report.unsupported_ops:
            console.print(f"  • {op}")
    
    if output:
        console.print(f"\nReport saved to: {output}")


@app.command()
def ptq(
    model_path: Path = typer.Argument(..., help="Path to PyTorch model (.pt)"),
    output_dir: Path = typer.Option(
        Path("./ptq_output"), "-o", "--output", help="Output directory"
    ),
    calibration_dir: Optional[Path] = typer.Option(
        None, "--calibration", "-c", help="Directory with calibration data (numpy files)"
    ),
    samples: int = typer.Option(
        100, "-n", "--samples", help="Number of calibration samples"
    ),
    backend: str = typer.Option(
        "x86", "-b", "--backend", help="Quantization backend (x86/qnnpack)"
    ),
    input_shape: str = typer.Option(
        "1,3,224,224", "-s", "--shape", help="Input shape (comma-separated)"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
):
    """
    Run Post-Training Quantization on a PyTorch model.
    
    Example:
        onnx2fx ptq model.pt -o quantized/ --shape 1,3,224,224
    """
    setup_logging(verbose)
    
    import torch
    import numpy as np
    from onnx2fx.quantization.ptq import PTQConfig, run_ptq_pipeline
    
    console.print(Panel("[bold blue]Post-Training Quantization[/bold blue]"))
    
    # Parse input shape
    shape = tuple(int(x) for x in input_shape.split(","))
    
    # Load model
    try:
        model = torch.load(model_path)
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        raise typer.Exit(1)
    
    # Prepare calibration data
    if calibration_dir and calibration_dir.exists():
        # Load numpy files from directory
        calib_data = []
        for f in sorted(calibration_dir.glob("*.npy"))[:samples]:
            data = torch.from_numpy(np.load(f))
            calib_data.append(data)
        console.print(f"Loaded {len(calib_data)} calibration samples from {calibration_dir}")
    else:
        # Generate random calibration data
        console.print(f"Using random calibration data (shape={shape}, samples={samples})")
        calib_data = [torch.randn(shape) for _ in range(samples)]
    
    # Run PTQ
    config = PTQConfig(
        backend=backend,
        calibration_samples=samples,
    )
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running PTQ pipeline...", total=None)
        
        outputs = run_ptq_pipeline(
            model=model,
            calibration_data=calib_data,
            output_dir=output_dir,
            input_shape=shape,
            config=config,
            model_name=model_path.stem,
        )
    
    # Display results
    table = Table(title="PTQ Outputs")
    table.add_column("Artifact", style="cyan")
    table.add_column("Path", style="green")
    
    for name, path in outputs.items():
        table.add_row(name, str(path))
    
    console.print(table)
    console.print("[bold green]✓ PTQ complete![/bold green]")


@app.command()
def qft(
    model_path: Path = typer.Argument(..., help="Path to PyTorch model (.pt)"),
    output_dir: Path = typer.Option(
        Path("./qft_output"), "-o", "--output", help="Output directory"
    ),
    train_dir: Optional[Path] = typer.Option(
        None, "--train", "-t", help="Directory with training data (numpy files)"
    ),
    epochs: int = typer.Option(1, "-e", "--epochs", help="Number of training epochs"),
    learning_rate: float = typer.Option(
        1e-4, "-lr", "--learning-rate", help="Learning rate"
    ),
    batch_size: int = typer.Option(32, "-bs", "--batch-size", help="Batch size"),
    input_shape: str = typer.Option(
        "1,3,224,224", "-s", "--shape", help="Input shape (comma-separated)"
    ),
    device: str = typer.Option("cpu", "-d", "--device", help="Device (cpu/cuda)"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
):
    """
    Run Quantization-Aware Fine-Tuning (1 epoch as per SOW).
    
    Example:
        onnx2fx qft model.pt -o qft_output/ --train train_data/
    """
    setup_logging(verbose)
    
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset
    from onnx2fx.quantization.qft import QFTConfig, run_qft_pipeline
    
    console.print(Panel("[bold blue]Quantization-Aware Fine-Tuning[/bold blue]"))
    
    # Parse input shape
    shape = tuple(int(x) for x in input_shape.split(","))
    
    # Load model
    try:
        model = torch.load(model_path)
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        raise typer.Exit(1)
    
    # Prepare training data
    if train_dir and train_dir.exists():
        # Load from directory
        tensors = []
        for f in sorted(train_dir.glob("*.npy")):
            data = torch.from_numpy(np.load(f))
            tensors.append(data)
        if tensors:
            train_data = torch.stack(tensors)
        else:
            train_data = torch.randn(100, *shape[1:])
        console.print(f"Loaded {len(train_data)} samples from {train_dir}")
    else:
        # Generate random training data
        console.print(f"Using random training data (shape={shape})")
        train_data = torch.randn(100, *shape[1:])
    
    # Create DataLoader
    dataset = TensorDataset(train_data)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Run QFT
    config = QFTConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Running QFT ({epochs} epoch)...", total=None)
        
        outputs = run_qft_pipeline(
            model=model,
            train_loader=train_loader,
            output_dir=output_dir,
            input_shape=shape,
            config=config,
            model_name=model_path.stem,
            device=device,
        )
    
    # Display results
    table = Table(title="QFT Outputs")
    table.add_column("Artifact", style="cyan")
    table.add_column("Path", style="green")
    
    for name, path in outputs.items():
        table.add_row(name, str(path))
    
    console.print(table)
    console.print("[bold green]✓ QFT complete![/bold green]")


@app.command()
def batch(
    models_dir: Path = typer.Argument(..., help="Directory containing ONNX models"),
    output_dir: Path = typer.Option(
        Path("./batch_output"), "-o", "--output", help="Output directory"
    ),
    run_ptq: bool = typer.Option(False, "--ptq", help="Run PTQ on converted models"),
    run_qft: bool = typer.Option(False, "--qft", help="Run QFT on converted models"),
    input_shape: str = typer.Option(
        "1,3,224,224", "-s", "--shape", help="Default input shape"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
):
    """
    Batch convert multiple ONNX models.
    
    Example:
        onnx2fx batch ./models/ -o ./output/ --ptq
    """
    setup_logging(verbose)
    
    import torch
    import numpy as np
    from onnx2fx.converter import OnnxToFxConverter
    from onnx2fx.reports import ReportGenerator
    
    console.print(Panel("[bold blue]Batch Model Conversion[/bold blue]"))
    
    # Parse input shape
    shape = tuple(int(x) for x in input_shape.split(","))
    
    # Find ONNX models
    onnx_files = list(models_dir.glob("*.onnx"))
    if not onnx_files:
        console.print(f"[yellow]No ONNX models found in {models_dir}[/yellow]")
        raise typer.Exit(0)
    
    console.print(f"Found {len(onnx_files)} ONNX models")
    if run_ptq:
        console.print("[cyan]PTQ enabled[/cyan]")
    if run_qft:
        console.print("[cyan]QFT enabled[/cyan]")
    
    # Initialize report generator
    reporter = ReportGenerator(output_dir / "reports")
    
    # Process each model
    results = {"success": 0, "failed": 0, "ptq": 0, "qft": 0}
    
    for model_file in onnx_files:
        model_name = model_file.stem
        console.print(f"\n[cyan]Processing: {model_name}[/cyan]")
        
        report = reporter.create_report(model_name)
        model_output = output_dir / model_name
        
        try:
            # Convert
            converter = OnnxToFxConverter(model_file)
            pytorch_model = converter.convert()
            converter.save(model_output)
            
            report.conversion_status = "success"
            report.fx_traceable = converter.get_fx_graph() is not None
            results["success"] += 1
            
            console.print(f"  [green]✓ Converted successfully[/green]")
            
            # Run PTQ if requested
            if run_ptq:
                try:
                    from onnx2fx.quantization.ptq import PTQConfig, run_ptq_pipeline
                    
                    console.print(f"  [cyan]Running PTQ...[/cyan]")
                    ptq_config = PTQConfig(backend="qnnpack")
                    calib_data = [torch.randn(shape) for _ in range(100)]
                    
                    _, ptq_outputs = run_ptq_pipeline(
                        model=pytorch_model,
                        calibration_data=calib_data,
                        output_dir=model_output / "ptq",
                        input_shape=shape,
                        config=ptq_config,
                        model_name=model_name,
                    )
                    
                    report.ptq_status = "success"
                    results["ptq"] += 1
                    console.print(f"  [green]✓ PTQ complete[/green]")
                except Exception as e:
                    report.ptq_status = "failed"
                    console.print(f"  [yellow]⚠ PTQ failed: {e}[/yellow]")
            
            # Run QFT if requested
            if run_qft:
                try:
                    from torch.utils.data import DataLoader, TensorDataset
                    from onnx2fx.quantization.qft import QFTConfig, run_qft_pipeline
                    
                    console.print(f"  [cyan]Running QFT (1 epoch)...[/cyan]")
                    qft_config = QFTConfig(backend="qnnpack", epochs=1)
                    train_data = torch.randn(100, *shape[1:])
                    dataset = TensorDataset(train_data)
                    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
                    
                    _, qft_outputs = run_qft_pipeline(
                        model=pytorch_model,
                        train_loader=train_loader,
                        output_dir=model_output / "qft",
                        input_shape=shape,
                        config=qft_config,
                        model_name=model_name,
                    )
                    
                    report.qft_status = "success"
                    results["qft"] += 1
                    console.print(f"  [green]✓ QFT complete[/green]")
                except Exception as e:
                    report.qft_status = "failed"
                    console.print(f"  [yellow]⚠ QFT failed: {e}[/yellow]")
            
        except Exception as e:
            report.conversion_status = "failed"
            report.conversion_errors.append(str(e))
            results["failed"] += 1
            console.print(f"  [red]✗ Failed: {e}[/red]")
    
    # Save reports
    reporter.save_all()
    reporter.save_summary()
    
    # Final summary
    summary_parts = [f"{results['success']} converted, {results['failed']} failed"]
    if run_ptq:
        summary_parts.append(f"{results['ptq']} PTQ")
    if run_qft:
        summary_parts.append(f"{results['qft']} QFT")
    console.print(f"\n[bold]Results: {', '.join(summary_parts)}[/bold]")
    console.print(f"Reports saved to: {output_dir / 'reports'}")


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", help="Show version and exit"
    ),
):
    """ONNX to FX-traceable PyTorch converter with PTQ/QFT support."""
    if version:
        from onnx2fx import __version__
        console.print(f"onnx2fx version {__version__}")
        raise typer.Exit(0)


if __name__ == "__main__":
    app()
