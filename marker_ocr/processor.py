"""Core OCR processing module using Marker's layout-aware pipeline.

Marker emits whole-document markdown plus an ``images`` dict in a single call.
This module owns *how OCR happens* (the marker conversion); the shared
``ocr-output-contract`` package owns *where bytes go* (paths, page assembly,
dual-level metadata, the uniform exit policy).

Page boundaries: marker is run with ``paginate_output=True`` so the rendered
markdown carries per-page boundary markers (``{page_id}`` followed by a rule).
We split on those markers to recover per-page text, then re-emit it under the
canonical ``## Page N`` headers via :func:`assemble_pages`. If a document yields
no boundary markers, the whole blob is kept as a single page so content is never
silently dropped.
"""

import io
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ocr_output_contract import (
    DocMetadata,
    RootIndex,
    RunOutcome,
    Status,
    assemble_pages,
    doc_dir_for,
    figure_filename,
    figure_markdown_link,
    figures_dir_for,
    markdown_path_for,
    relative_key,
    resolve_output_root,
    sha256_checksum,
    utc_timestamp,
    write_doc_metadata,
)
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from marker_ocr.config import Config
from marker_ocr.utils import (
    format_file_size,
    get_supported_files,
)

logger = logging.getLogger(__name__)

#: Backend identifier recorded in metadata. Marker is local layout-aware OCR.
BACKEND = "marker-pdf"
#: Model identifier recorded in metadata (marker's Surya + Texify stack).
MODEL = "marker"

# Shared console instance — CLI sets .quiet on this directly
console = Console()

# Marker's paginated output prefixes each page with ``{page_id}`` followed by a
# run of hyphens (default ``'-' * 48``). page_id is 0-indexed. We split on these
# markers to recover per-page text, dropping the marker itself.
_PAGE_MARKER_RE = re.compile(r"\{(\d+)\}-{3,}")

# Marker's image dict keys encode the source page, e.g. ``_page_1_Figure_0.jpeg``
# or ``page_2_Picture_3.png``. Capture the first integer that follows ``page``.
_IMG_PAGE_RE = re.compile(r"page[_\-]?(\d+)", re.IGNORECASE)


def split_marker_pages(markdown: str) -> list[str]:
    """Split marker's paginated markdown into per-page text blocks.

    Marker (run with ``paginate_output=True``) prefixes each page with a
    ``{page_id}`` + rule marker. We split on those, drop empty leading/trailing
    fragments, and return the per-page bodies in order. When no markers are
    present (pagination unavailable, or a single-page doc rendered without one),
    the whole blob is returned as a single page so content is never dropped.
    """
    text = markdown.strip()
    if not text:
        return []
    parts = _PAGE_MARKER_RE.split(text)
    # re.split with one capture group yields: [pre, id, body, id, body, ...].
    # If the doc starts with a marker, parts[0] is empty/whitespace.
    if len(parts) == 1:
        # No markers found — keep the whole document as a single page.
        return [text]

    pages: list[str] = []
    # parts[0] is any text before the first marker (usually empty).
    lead = parts[0].strip()
    if lead:
        pages.append(lead)
    # Remaining items come in (page_id, body) pairs.
    for i in range(1, len(parts), 2):
        body = parts[i + 1] if i + 1 < len(parts) else ""
        pages.append(body.strip())
    # A doc that was all markers (no content) collapses to nothing — guard it.
    return pages or [text]


@dataclass
class OCRResult:
    """Result from processing a document.

    ``pages`` holds the per-page markdown text in order. ``success`` is True when
    marker produced usable text; a conversion error yields an empty ``pages`` and
    an ``error`` string, mapped to ``Status.FAILED``.
    """

    file_path: Path
    pages: list[str]
    success: bool
    error: str | None = None
    processing_time: float = 0.0
    #: marker's image dict ({name: PIL.Image-like}); name encodes the page.
    images: dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def status(self) -> Status:
        if self.success and self.pages:
            return Status.COMPLETED
        return Status.FAILED

    @property
    def text(self) -> str:
        """Flat text view (pages joined with blank lines)."""
        return "\n\n".join(p for p in self.pages if p)


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

        marker_config = self.config.to_marker_config()
        self._artifact_dict = create_model_dict()
        self._converter = PdfConverter(artifact_dict=self._artifact_dict, config=marker_config)
        logger.info("Loaded Marker models")

    # ------------------------------------------------------------------
    # Conversion (marker boundary)
    # ------------------------------------------------------------------

    def process_file(self, file_path: Path, show_progress: bool = True) -> OCRResult:
        """Process a single PDF file through marker and recover per-page text."""
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

            images = {}
            if hasattr(rendered, "images") and rendered.images:
                images = rendered.images

            pages = split_marker_pages(rendered.markdown or "")
            if not pages:
                return OCRResult(
                    file_path=file_path,
                    pages=[],
                    success=False,
                    error="Marker returned empty output (no text)",
                    processing_time=time.time() - start_time,
                )

            return OCRResult(
                file_path=file_path,
                pages=pages,
                success=True,
                processing_time=time.time() - start_time,
                images=images,
            )
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return OCRResult(
                file_path=file_path,
                pages=[],
                success=False,
                error=str(e),
                processing_time=time.time() - start_time,
            )

    # ------------------------------------------------------------------
    # Output writing (all routed through the ocr-output-contract package)
    # ------------------------------------------------------------------

    def save_results(
        self,
        result: OCRResult,
        output_root: Path,
        rel_key: str,
    ) -> Path:
        """Write the aggregated markdown + figures for one document.

        Layout is determined entirely by the contract package:
        ``<output_root>/<rel/dir>/<stem>/<stem>.md`` plus a ``figures/`` folder.
        Pages are emitted under ``## Page N`` headers (no frontmatter); figures
        are normalised to PNG and named ``figure_<N>_page<P>.png`` with links
        that RESOLVE from the markdown file.
        """
        doc_dir = doc_dir_for(output_root, rel_key)
        doc_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = markdown_path_for(doc_dir, rel_key)

        body = assemble_pages(result.pages) if result.pages else "*[OCR Failed]*\n"

        figure_links = self._save_figures(result, doc_dir)
        if figure_links:
            body = body.rstrip("\n") + "\n\n## Figures\n\n" + "\n\n".join(figure_links) + "\n"

        markdown_path.write_text(body, encoding="utf-8")

        if self.config.verbose:
            console.print(f"[green]Saved:[/green] {markdown_path}")

        return markdown_path

    def _save_figures(self, result: OCRResult, doc_dir: Path) -> list[str]:
        """Persist marker's images as PNG and return resolving markdown links.

        Marker's image keys encode the source page (e.g. ``_page_1_Figure_0``).
        We parse that page number, save under ``figures/figure_<N>_page<P>.png``,
        and emit links via the contract's ``figure_markdown_link`` so they resolve
        relative to the ``.md`` (the prior dangling-link bug: links pointed at the
        marker-native key in the doc dir, not the file in ``figures/``).
        """
        if not result.images:
            return []

        figures_dir = figures_dir_for(doc_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        links: list[str] = []
        figure_counter = 0
        for img_name, img in sorted(result.images.items()):
            page_no = self._page_from_image_name(img_name)
            figure_counter += 1
            filename = figure_filename(figure_counter, page_no)
            img_path = figures_dir / filename
            try:
                pil_img = self._to_pil(img)
                if pil_img.mode not in ("RGB", "RGBA"):
                    pil_img = pil_img.convert("RGB")
                pil_img.save(img_path, format="PNG")
                links.append(figure_markdown_link(figure_counter, page_no))
            except Exception as e:
                logger.warning(f"Failed to save figure {img_name!r}: {e}")
                figure_counter -= 1
        return links

    @staticmethod
    def _page_from_image_name(name: str) -> int:
        """Extract the 1-indexed source page from a marker image key.

        Marker keys look like ``_page_1_Figure_0.jpeg``. The captured number is
        marker's page id (0-indexed in the markdown markers but 1-indexed in the
        image keys for current marker-pdf); we clamp to at least 1.
        """
        m = _IMG_PAGE_RE.search(name)
        if not m:
            return 1
        return max(1, int(m.group(1)))

    @staticmethod
    def _to_pil(img: Any) -> Image.Image:
        """Coerce a marker image value (PIL image or raw bytes) to a PIL image."""
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, (bytes, bytearray)):
            return Image.open(io.BytesIO(bytes(img)))
        if hasattr(img, "save"):  # already PIL-like
            return img  # type: ignore[return-value]
        raise TypeError(f"Unsupported image value type: {type(img)!r}")

    def _build_doc_metadata(
        self,
        result: OCRResult,
        file_path: Path,
        markdown_path: Path,
        output_root: Path,
    ) -> DocMetadata:
        """Assemble the per-document metadata record from a result."""
        status = result.status
        error = result.error if status is not Status.COMPLETED else None
        return DocMetadata(
            status=status,
            checksum=sha256_checksum(file_path),
            model=MODEL,
            backend=BACKEND,
            processing_time=result.processing_time,
            timestamp=utc_timestamp(),
            output_path=str(markdown_path.relative_to(output_root)),
            pages=result.page_count,
            error=error,
        )

    def _persist(
        self,
        result: OCRResult,
        file_path: Path,
        output_root: Path,
        rel_key: str,
        index: RootIndex,
    ) -> tuple[DocMetadata, Path]:
        """Write markdown, figures, and BOTH metadata levels for one document.

        Always writes output (markdown + per-doc + root metadata) regardless of
        success, so failures are recorded with ``status=failed`` per the canon.
        Returns ``(metadata, markdown_path)``.
        """
        markdown_path = self.save_results(result, output_root, rel_key)
        meta = self._build_doc_metadata(result, file_path, markdown_path, output_root)
        doc_dir = doc_dir_for(output_root, rel_key)
        write_doc_metadata(doc_dir, rel_key, meta)
        index.record(rel_key, meta)
        return meta, markdown_path

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def process(
        self,
        input_path: Path,
        output_path: Path | None = None,
        reprocess: bool = False,
    ) -> RunOutcome:
        """Process input path (file or directory). Returns a RunOutcome.

        The returned :class:`RunOutcome` carries the uniform exit policy: nonzero
        if any file failed, across both single-file and batch runs.
        """
        if input_path.is_file():
            return self._process_single_file(input_path, output_path, reprocess)
        elif input_path.is_dir():
            return self._process_directory(input_path, output_path, reprocess)
        else:
            raise ValueError(f"Input path does not exist: {input_path}")

    def _process_single_file(
        self,
        file_path: Path,
        output_path: Path | None,
        reprocess: bool,
    ) -> RunOutcome:
        """Process a single file. Scan root is the file's parent (rel key = name)."""
        outcome = RunOutcome()
        output_root = resolve_output_root(file_path, output_path)
        output_root.mkdir(parents=True, exist_ok=True)
        rel_key = relative_key(file_path, file_path.parent)
        index = RootIndex(output_root)

        if not reprocess and index.is_completed(rel_key, sha256_checksum(file_path)):
            console.print(f"[yellow]Already processed:[/yellow] {file_path.name}")
            console.print("[dim]Use --reprocess to force reprocessing[/dim]")
            outcome.add(
                Status.COMPLETED,
                output_path=str(markdown_path_for(doc_dir_for(output_root, rel_key), rel_key)),
            )
            return outcome

        console.print(f"[blue]Processing:[/blue] {file_path}")
        console.print(f"[blue]Output:[/blue] {output_root}\n")

        result = self.process_file(file_path)
        meta, markdown_path = self._persist(result, file_path, output_root, rel_key, index)
        outcome.add(
            meta.status,
            detail=None if meta.status is Status.COMPLETED else rel_key,
            output_path=str(markdown_path),
        )

        if meta.status is Status.COMPLETED:
            console.print("\n[green]Success[/green]")
            console.print(
                f"[dim]Time: {result.processing_time:.2f}s | Pages: {result.page_count}[/dim]"
            )
        else:
            console.print(f"\n[red]Failed ({meta.status.value}):[/red] {meta.error}")
        return outcome

    def _process_directory(
        self,
        dir_path: Path,
        output_path: Path | None,
        reprocess: bool,
    ) -> RunOutcome:
        """Process all files in a directory (sequential — GPU is the bottleneck)."""
        outcome = RunOutcome()
        files = get_supported_files(dir_path)
        if not files:
            console.print("[yellow]No supported files found[/yellow]")
            return outcome

        output_root = resolve_output_root(dir_path, output_path)
        output_root.mkdir(parents=True, exist_ok=True)
        index = RootIndex(output_root)

        files_to_process: list[tuple[Path, str]] = []
        for f in files:
            rel_key = relative_key(f, dir_path)
            if not reprocess and index.is_completed(rel_key, sha256_checksum(f)):
                if self.config.verbose:
                    console.print(f"[dim]Skipping: {rel_key}[/dim]")
                outcome.add(
                    Status.COMPLETED,
                    output_path=str(markdown_path_for(doc_dir_for(output_root, rel_key), rel_key)),
                )
            else:
                files_to_process.append((f, rel_key))

        if not files_to_process:
            console.print("[green]All files already processed[/green]")
            console.print("[dim]Use --reprocess to force reprocessing[/dim]")
            return outcome

        console.print(f"[blue]Processing {len(files_to_process)} file(s)...[/blue]")
        console.print(f"[blue]Output:[/blue] {output_root}\n")

        start_time = time.time()

        for file_path, rel_key in files_to_process:
            file_size = format_file_size(file_path.stat().st_size)
            console.print(f"[cyan]{rel_key}[/cyan] ({file_size})")

            result = self.process_file(file_path, show_progress=False)
            meta, markdown_path = self._persist(result, file_path, output_root, rel_key, index)
            outcome.add(
                meta.status,
                detail=None if meta.status is Status.COMPLETED else rel_key,
                output_path=str(markdown_path),
            )
            if meta.status is Status.COMPLETED:
                console.print(f"  [green]OK[/green] ({result.processing_time:.1f}s)\n")
            else:
                console.print(f"  [red]{meta.status.value.upper()}: {meta.error}[/red]\n")

        total_time = time.time() - start_time
        total = outcome.completed + outcome.failed + outcome.partial
        console.print(f"\n[green]Completed:[/green] {outcome.completed}/{total} files")
        if outcome.has_failures:
            console.print(
                f"[red]Failures:[/red] {outcome.failed} failed, {outcome.partial} partial"
            )
        console.print(f"[dim]Total time: {total_time:.2f}s[/dim]")
        return outcome
