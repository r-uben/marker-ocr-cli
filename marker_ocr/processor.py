"""Core OCR processing module using Marker's layout-aware pipeline."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from marker_ocr.config import Config
from marker_ocr.metadata import MetadataManager
from marker_ocr.utils import (
    determine_output_path,
    format_file_size,
    get_supported_files,
    sanitize_filename,
)

logger = logging.getLogger(__name__)

# Shared console instance — CLI sets .quiet on this directly
console = Console()


@dataclass
class OCRResult:
    """Result from processing a document."""

    file_path: Path
    text: str
    success: bool
    pages: int = 0
    error: str | None = None
    processing_time: float = 0.0
    images: dict[str, Any] = field(default_factory=dict)


class OCRProcessor:
    """OCR processor using Marker's layout-aware pipeline."""

    def __init__(self, config: Config):
        """Initialize the OCR processor and load models."""
        self.config = config
        self._converter = None
        self._load_models()

    def _load_models(self) -> None:
        """Load Marker models (Surya layout, reading order, OCR, Texify)."""
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        self._artifact_dict = create_model_dict()
        self._converter = PdfConverter(artifact_dict=self._artifact_dict)
        logger.info("Loaded Marker models")

    def process(
        self,
        input_path: Path,
        output_path: Path | None = None,
        reprocess: bool = False,
    ) -> None:
        """Process input path (file or directory)."""
        if input_path.is_file():
            self._process_single_file(input_path, output_path, reprocess)
        elif input_path.is_dir():
            self._process_directory(input_path, output_path, reprocess)
        else:
            raise ValueError(f"Input path does not exist: {input_path}")

    def _process_single_file(
        self,
        file_path: Path,
        output_path: Path | None,
        reprocess: bool,
    ) -> None:
        """Process a single file."""
        output_dir = determine_output_path(file_path, output_path)
        meta = MetadataManager(output_dir)

        if meta.is_processed(file_path) and not reprocess:
            console.print(f"[yellow]Already processed:[/yellow] {file_path.name}")
            console.print("[dim]Use --reprocess to force reprocessing[/dim]")
            return

        console.print(f"[blue]Processing:[/blue] {file_path}")
        console.print(f"[blue]Output:[/blue] {output_dir}\n")

        result = self.process_file(file_path)

        if result.success:
            output_file = self.save_results(result, output_dir)
            meta.record(
                file_path,
                processing_time=result.processing_time,
                output_path=str(output_file.relative_to(output_dir)),
                pages=result.pages,
            )
            console.print("\n[green]Success[/green]")
            console.print(f"[dim]Time: {result.processing_time:.2f}s | Pages: {result.pages}[/dim]")
        else:
            console.print(f"\n[red]Failed to process file: {result.error}[/red]")

    def _process_directory(
        self,
        dir_path: Path,
        output_path: Path | None,
        reprocess: bool,
    ) -> None:
        """Process all files in a directory (sequential — GPU is the bottleneck)."""
        files = get_supported_files(dir_path)
        if not files:
            console.print("[yellow]No supported files found[/yellow]")
            return

        output_dir = determine_output_path(dir_path, output_path)
        meta = MetadataManager(output_dir)

        # Filter files
        files_to_process = []
        for f in files:
            if meta.is_processed(f) and not reprocess:
                if self.config.verbose:
                    console.print(f"[dim]Skipping: {f.name}[/dim]")
            else:
                files_to_process.append(f)

        if not files_to_process:
            console.print("[green]All files already processed[/green]")
            console.print("[dim]Use --reprocess to force reprocessing[/dim]")
            return

        console.print(f"[blue]Processing {len(files_to_process)} file(s)...[/blue]")
        console.print(f"[blue]Output:[/blue] {output_dir}\n")

        start_time = time.time()
        success_count = 0

        for file_path in files_to_process:
            file_size = format_file_size(file_path.stat().st_size)
            console.print(f"[cyan]{file_path.name}[/cyan] ({file_size})")

            result = self.process_file(file_path)

            if result.success:
                output_file = self.save_results(result, output_dir)
                meta.record(
                    file_path,
                    processing_time=result.processing_time,
                    output_path=str(output_file.relative_to(output_dir)),
                    pages=result.pages,
                )
                success_count += 1
                console.print(f"  [green]OK[/green] ({result.processing_time:.1f}s)\n")
            else:
                console.print(f"  [red]FAILED: {result.error}[/red]\n")

        total_time = time.time() - start_time
        console.print(f"\n[green]Completed:[/green] {success_count}/{len(files_to_process)} files")
        console.print(f"[dim]Total time: {total_time:.2f}s[/dim]")

    def process_file(self, file_path: Path, show_progress: bool = True) -> OCRResult:
        """Process a single PDF file."""
        start_time = time.time()
        self.config.validate_file_size(file_path)

        try:
            if show_progress and not self.config.quiet:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True,
                ) as progress:
                    progress.add_task("Running Marker pipeline...", total=None)
                    rendered = self._converter(str(file_path))
            else:
                rendered = self._converter(str(file_path))

            # Extract page count from metadata
            page_count = 0
            if hasattr(rendered, "metadata") and rendered.metadata:
                page_count = rendered.metadata.get("pages_processed", 0)

            # Extract images
            images = {}
            if hasattr(rendered, "images") and rendered.images:
                images = rendered.images

            return OCRResult(
                file_path=file_path,
                text=rendered.markdown,
                success=True,
                pages=page_count,
                processing_time=time.time() - start_time,
                images=images,
            )
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return OCRResult(
                file_path=file_path,
                text="",
                success=False,
                error=str(e),
                processing_time=time.time() - start_time,
            )

    def save_results(self, result: OCRResult, output_dir: Path) -> Path:
        """Save OCR results to per-document folder.

        Output structure:
            output_dir/doc_name/doc_name.md
            output_dir/doc_name/figures/
        """
        base_name = sanitize_filename(result.file_path.stem)
        doc_dir = output_dir / base_name
        doc_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = doc_dir / f"{base_name}.md"

        # Write clean markdown — just the OCR text, no headers
        markdown_path.write_text(
            result.text if result.success else f"*[OCR Failed: {result.error}]*",
            encoding="utf-8",
        )

        # Save extracted images/figures
        if result.images:
            figures_dir = doc_dir / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            for img_name, img in result.images.items():
                img_path = figures_dir / img_name
                if hasattr(img, "save"):
                    img.save(img_path)

        if self.config.verbose:
            console.print(f"[green]Saved:[/green] {markdown_path}")

        return markdown_path
