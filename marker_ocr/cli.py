"""Command-line interface for Marker OCR."""

import os
import sys
from pathlib import Path

import click
from ocr_output_contract import iter_input_files, resolve_output_root
from rich.console import Console
from rich.table import Table

from marker_ocr import __version__
from marker_ocr.config import Config
from marker_ocr.processor import OCRProcessor
from marker_ocr.processor import console as proc_console
from marker_ocr.utils import (
    SUPPORTED_EXTENSIONS,
    format_file_size,
    get_pdf_page_count,
    is_pdf_file,
    setup_logging,
)

console = Console()


@click.command()
@click.argument("input_path", type=click.Path(path_type=Path), required=False)
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
    "--device",
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    default=None,
    help="Device for inference (default: cpu on Apple Silicon, auto elsewhere)",
)
@click.option(
    "--info",
    is_flag=True,
    help="Show system and device information",
)
@click.version_option(version=__version__, prog_name="marker-ocr")
def cli(
    input_path: Path | None,
    output_dir: Path | None,
    pages: str | None,
    force_ocr: bool,
    reprocess: bool,
    dry_run: bool,
    quiet: bool,
    verbose: bool,
    device: str | None,
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

    # Handle --info flag (no INPUT_PATH required)
    if info:
        _show_info()
        return

    # INPUT_PATH is optional only so --info/--version work without it. A bare
    # invocation with no INPUT_PATH (and no --info) is a usage error, not a
    # successful help path: emit the usage message and exit nonzero (click's
    # standard exit code 2) so scripts can detect the missing argument.
    if input_path is None:
        ctx = click.get_current_context()
        raise click.UsageError("Missing argument 'INPUT_PATH'.", ctx=ctx)

    # Validate input
    if not input_path.exists():
        console.print(f"[red]Error:[/red] Input path does not exist: {input_path}")
        sys.exit(1)

    # A single file passed directly must be a PDF (directory inputs are filtered
    # by extension downstream; a bare non-PDF would otherwise be handed to marker).
    if input_path.is_file() and not is_pdf_file(input_path):
        console.print(f"[red]Error:[/red] Unsupported file type (expected PDF): {input_path}")
        sys.exit(1)

    # Set quiet mode on processor console too
    if quiet:
        proc_console.quiet = True
        console.quiet = True

    # Handle --dry-run (no model loading needed)
    if dry_run:
        _dry_run(input_path, output_dir)
        return

    try:
        config = Config(
            pages=pages,
            force_ocr=force_ocr,
            device=device if device != "auto" else None,
            verbose=verbose,
            quiet=quiet,
            output_dir=output_dir,
        )

        # Must set TORCH_DEVICE before any marker imports
        config.apply_device()

        if not quiet:
            console.print(f"[bold blue]Marker OCR[/bold blue] [dim]v{__version__}[/dim]")
            device_info = os.environ.get("TORCH_DEVICE", "auto")
            console.print(f"[dim]Loading models (Surya + Texify) on {device_info}...[/dim]\n")

        processor = OCRProcessor(config)
        outcome = processor.process(
            input_path,
            output_path=output_dir,
            reprocess=reprocess,
        )

        if quiet:
            # Scripting contract: emit one written output .md path per line.
            for path in outcome.outputs:
                click.echo(path)
        else:
            console.print("\n[bold green]Done![/bold green]\n")

        # Uniform exit policy (canon SYS-02): nonzero if any file failed,
        # across both single-file and batch runs.
        if outcome.exit_code != 0:
            sys.exit(outcome.exit_code)

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


def _dry_run(input_path: Path, output_dir: Path | None = None) -> None:
    """List files that would be processed without loading models.

    Uses the SAME discovery as the real run (``iter_input_files`` with the
    resolved output root excluded) so the dry run never over-reports inputs the
    real run would skip (e.g. the engine's own ``ocr/`` output subtree).
    """
    output_root = resolve_output_root(input_path, output_dir)
    files = list(iter_input_files(input_path, output_root, SUPPORTED_EXTENSIONS))

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

    # Probe the installed distribution version, NOT marker.__version__: marker-pdf
    # 1.10.2 exposes no module-level __version__, so the old import-and-read probe
    # falsely reported "not installed" even when OCR worked. importlib.metadata
    # reads the actual installed package metadata.
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        console.print(f"[bold]Marker version:[/bold] {_pkg_version('marker-pdf')}")
    except PackageNotFoundError:
        console.print("[yellow]marker-pdf not installed[/yellow]")

    console.print()
    console.print("[bold]Supported Formats:[/bold] PDF")
    console.print("[bold]Models:[/bold] Surya (layout/OCR), Texify (equations)")
    console.print()


def main() -> None:
    """Console-script entry point — delegates straight to the Click command.

    The packaged ``marker-ocr`` binary MUST honour the same exit-code policy the
    Click layer enforces. A previous shim intercepted the bare no-arg case and
    rewrote it to ``cli(['--help'])``, which exits 0 — so the shipped binary
    returned success on missing required input, breaking the scripting contract
    (a no-input run that exits 0 is indistinguishable from a successful run).

    Now we delegate directly: ``cli`` raises a Click ``UsageError`` (exit 2) for a
    bare invocation with no ``INPUT_PATH`` (and no ``--info``), still printing the
    usage message and a ``Try '... --help'`` hint, so missing input is a nonzero
    exit while ``--help``/``--info``/``--version`` continue to exit 0.
    """
    cli()


if __name__ == "__main__":
    main()
