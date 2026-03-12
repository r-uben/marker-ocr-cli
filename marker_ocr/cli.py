"""Command-line interface for Marker OCR."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from marker_ocr import __version__
from marker_ocr.config import Config
from marker_ocr.processor import OCRProcessor
from marker_ocr.processor import console as proc_console
from marker_ocr.utils import (
    format_file_size,
    get_pdf_page_count,
    get_supported_files,
    is_pdf_file,
    setup_logging,
)

console = Console()


@click.command()
@click.argument("input_path", type=click.Path(path_type=Path))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Output directory for results",
)
@click.option(
    "--pages",
    type=str,
    default=None,
    help="Page range to process (e.g., '0-5' or '1,3,5')",
)
@click.option(
    "--force-ocr",
    is_flag=True,
    help="Force OCR on all pages regardless of embedded text",
)
@click.option(
    "--reprocess",
    is_flag=True,
    help="Reprocess files even if already done",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="List files that would be processed without loading models",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    help="Suppress output except file paths (for scripting)",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--info",
    is_flag=True,
    help="Show system and device information",
)
@click.version_option(version=__version__, prog_name="marker-ocr")
def cli(
    input_path: Path,
    output_dir: Path | None,
    pages: str | None,
    force_ocr: bool,
    reprocess: bool,
    dry_run: bool,
    quiet: bool,
    verbose: bool,
    info: bool,
) -> None:
    """Marker OCR - Extract text from PDFs using Marker's layout-aware pipeline.

    \b
    Examples:
        marker-ocr paper.pdf
        marker-ocr ./papers/ -o ./results/
        marker-ocr paper.pdf --pages 0-5
        marker-ocr paper.pdf --force-ocr
        marker-ocr --info
    """
    setup_logging(verbose=verbose)

    # Handle --info flag
    if info:
        _show_info()
        return

    # Validate input
    if not input_path.exists():
        console.print(f"[red]Error:[/red] Input path does not exist: {input_path}")
        sys.exit(1)

    # Set quiet mode on processor console too
    if quiet:
        proc_console.quiet = True
        console.quiet = True

    # Handle --dry-run (no model loading needed)
    if dry_run:
        _dry_run(input_path)
        return

    try:
        config = Config(
            pages=pages,
            force_ocr=force_ocr,
            verbose=verbose,
            quiet=quiet,
            output_dir=output_dir,
        )

        if not quiet:
            console.print(f"[bold blue]Marker OCR[/bold blue] [dim]v{__version__}[/dim]")
            console.print("[dim]Loading models (Surya + Texify)...[/dim]\n")

        processor = OCRProcessor(config)
        processor.process(
            input_path,
            output_path=output_dir,
            reprocess=reprocess,
        )

        if not quiet:
            console.print("\n[bold green]Done![/bold green]\n")

    except ValueError as e:
        console.print(f"\n[red]Error:[/red] {e}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]\n")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}\n")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def _dry_run(input_path: Path) -> None:
    """List files that would be processed without loading models."""
    files = [input_path] if input_path.is_file() else get_supported_files(input_path)

    if not files:
        console.print("[yellow]No supported files found[/yellow]")
        return

    table = Table(title="Files to process (dry run)")
    table.add_column("File", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Pages", justify="right")

    total_size = 0
    for f in files:
        size = f.stat().st_size
        total_size += size
        page_count = str(get_pdf_page_count(f)) if is_pdf_file(f) else "-"
        table.add_row(f.name, format_file_size(size), page_count)

    console.print(table)
    console.print(f"\n[dim]Total: {len(files)} file(s), {format_file_size(total_size)}[/dim]")


def _show_info() -> None:
    """Show system and device information."""
    console.print(f"[bold blue]Marker OCR[/bold blue] [dim]v{__version__}[/dim]\n")

    sys_table = Table(title="System Information")
    sys_table.add_column("Component", style="cyan")
    sys_table.add_column("Value", style="green")
    sys_table.add_row(
        "Python",
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )
    sys_table.add_row("Platform", sys.platform)
    console.print(sys_table)
    console.print()

    try:
        import torch

        device_table = Table(title="Device Information")
        device_table.add_column("Feature", style="cyan")
        device_table.add_column("Status", style="yellow")
        device_table.add_row("PyTorch", torch.__version__)
        device_table.add_row(
            "CUDA",
            f"Available ({torch.cuda.get_device_name(0)})"
            if torch.cuda.is_available()
            else "Not available",
        )
        device_table.add_row(
            "MPS (Apple Metal)",
            "Available" if torch.backends.mps.is_available() else "Not available",
        )
        console.print(device_table)
    except ImportError:
        console.print("[red]PyTorch not installed[/red]")

    console.print()

    try:
        import marker

        console.print(f"[bold]Marker version:[/bold] {marker.__version__}")
    except (ImportError, AttributeError):
        console.print("[yellow]marker-pdf not installed[/yellow]")

    console.print()
    console.print("[bold]Supported Formats:[/bold] PDF")
    console.print("[bold]Models:[/bold] Surya (layout/OCR), Texify (equations)")
    console.print()


def main() -> None:
    """Entry point -- handles bare invocations and delegates to cli()."""
    argv = sys.argv[1:]

    # If no args at all, show help
    if not argv:
        cli(["--help"])
        return

    cli()


if __name__ == "__main__":
    main()
