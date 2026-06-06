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
    figures_dir_for,
    iter_input_files,
    markdown_path_for,
    relative_key,
    resolve_output_root,
    run_fingerprint,
    safe_checksum,
    sha256_checksum,
    utc_timestamp,
    write_doc_metadata,
)
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from marker_ocr.config import Config
from marker_ocr.utils import (
    SUPPORTED_EXTENSIONS,
    format_file_size,
)

logger = logging.getLogger(__name__)

#: Backend identifier recorded in metadata. Marker is local layout-aware OCR.
BACKEND = "marker-pdf"
#: Model identifier recorded in metadata (marker's Surya + Texify stack).
MODEL = "marker"

# Shared console instance — CLI sets .quiet on this directly
console = Console()

# Marker's paginated output prefixes each page with ``{page_id}`` followed by a
# run of hyphens (default ``'-' * 48``). page_id is 0-indexed. Marker always
# emits the marker at the start of its own line (preceded by a blank line). The
# pattern is FULL-LINE-anchored (``(?m)^[ \t]*...[ \t]*$``) so a literal
# ``{5}---`` appearing mid-line in body text (code, templates, a brace-number
# followed by a horizontal rule) is NOT mistaken for a page boundary — the
# unanchored form fragmented pages and shifted all subsequent ``## Page N``.
_PAGE_MARKER_RE = re.compile(r"(?m)^[ \t]*\{(\d+)\}-{3,}[ \t]*$")

# Marker's image dict keys encode the source page, e.g. ``_page_1_Figure_0.jpeg``
# or ``page_2_Picture_3.png``. Capture the first integer that follows ``page``.
# In marker-pdf 1.10.2 this number is the SAME raw 0-indexed ``page_id`` that the
# markdown ``{page_id}`` boundary marker carries: the markdown renderer builds the
# marker from ``data-page-id`` (renderers/markdown.py) and the image key from
# ``ref_block_id.to_path()`` -> ``_page_<page_id>_<Type>_<id>``
# (schema/blocks/base.py, renderers/html.py), both using the identical page_id.
_IMG_PAGE_RE = re.compile(r"page[_\-]?(\d+)", re.IGNORECASE)

# Marker image keys end with the block id, e.g. ``_page_1_Figure_10.jpeg``. Capture
# the trailing integer so figures sort numerically WITHIN a page (block 10 after
# block 2), not lexicographically ("_Figure_10" < "_Figure_2" as strings).
_IMG_BLOCK_RE = re.compile(r"_(\d+)(?:\.[A-Za-z0-9]+)?$")


def _marker_page_id_to_source_page(page_id: int) -> int:
    """Map marker's raw 0-indexed ``page_id`` to a 1-indexed source page number.

    This is the SINGLE source of truth shared by the body ``## Page N`` headers
    (via :func:`split_marker_pages`) and the figure ``page<P>`` tags (via
    :func:`OCRProcessor._page_from_image_name`). Because marker-pdf shares one raw
    ``page_id`` between the markdown boundary marker and the image-dict key, both
    MUST apply the identical offset or a figure's ``page<P>`` tag drifts off-by-one
    from the body header for the same physical page (the canon's page-number
    fidelity guarantee). Clamp to >=1 so a negative/garbage id can never produce a
    non-positive page label.
    """
    return max(1, page_id + 1)


def split_marker_pages(markdown: str) -> list[tuple[int, str]]:
    """Split marker's paginated markdown into ``(source_page, text)`` blocks.

    Marker (run with ``paginate_output=True``) prefixes each page with a
    full-line ``{page_id}`` + rule marker (``page_id`` is 0-indexed). We split on
    those, recover the captured source page id (offset to 1-index), and return
    the per-page bodies in order alongside their source page number.

    When no markers are present (pagination unavailable, or a single-page doc
    rendered without one) the whole blob is returned as a single page numbered 1,
    so content is never dropped. Any text before the first marker (rare preamble)
    is folded into the first page.
    """
    text = markdown.strip()
    if not text:
        return []

    matches = list(_PAGE_MARKER_RE.finditer(markdown))
    if not matches:
        # No markers found — keep the whole document as a single page.
        return [(1, text)]

    pages: list[tuple[int, str]] = []
    preamble = markdown[: matches[0].start()].strip()
    for i, m in enumerate(matches):
        page_id = int(m.group(1))
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[body_start:body_end].strip()
        if i == 0 and preamble:
            body = (preamble + "\n\n" + body).strip() if body else preamble
        # marker's page_id is 0-indexed; present a 1-indexed source page number
        # via the shared helper so the body header matches the figure page tag.
        pages.append((_marker_page_id_to_source_page(page_id), body))
    # A doc that was all markers (no content) collapses to nothing — guard it.
    return pages or [(1, text)]


@dataclass
class OCRResult:
    """Result from processing a document.

    ``pages`` holds ``(source_page, markdown_text)`` pairs in order: the source
    page number (1-indexed, ``marker page_id + 1``, applied via the SAME
    :func:`_marker_page_id_to_source_page` helper the figure tags use, so body
    ``## Page N`` headers and figure ``page<P>`` tags agree for the same source
    page — both whole-document and under ``--pages``) and that page's body.
    ``success`` is True when marker produced usable text; a conversion error
    yields empty ``pages`` and an ``error`` string -> ``Status.FAILED``.
    """

    file_path: Path
    pages: list[tuple[int, str]]
    success: bool
    error: str | None = None
    processing_time: float = 0.0
    #: marker's image dict ({name: PIL.Image-like}); name encodes the page.
    images: dict[str, Any] = field(default_factory=dict)
    #: True when SOME content was produced but the render is incomplete (page
    #: shortfall vs the real PDF, or a figure that could not be saved). Drives
    #: ``Status.PARTIAL`` so the run is non-silently degraded, not "completed".
    degraded: bool = False

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def page_numbers(self) -> list[int]:
        """Source page numbers, in body order (for explicit ``## Page N`` labels)."""
        return [n for n, _ in self.pages]

    @property
    def page_texts(self) -> list[str]:
        """Per-page body text, in order."""
        return [t for _, t in self.pages]

    @property
    def has_content(self) -> bool:
        """True if at least one page has non-whitespace body text.

        An all-empty/whitespace render is treated as a non-success: the page
        markers may exist but the document carries no recoverable content.
        """
        return any(t.strip() for _, t in self.pages)

    @property
    def status(self) -> Status:
        # No usable content at all -> hard failure.
        if not (self.pages and self.has_content):
            return Status.FAILED
        # Content produced but incomplete (page shortfall / dropped figure) ->
        # partial, recorded non-silently rather than masquerading as completed.
        if self.degraded:
            return Status.PARTIAL
        if self.success:
            return Status.COMPLETED
        return Status.FAILED

    @property
    def text(self) -> str:
        """Flat text view (pages joined with blank lines)."""
        return "\n\n".join(t for _, t in self.pages if t)


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

            result = OCRResult(
                file_path=file_path,
                pages=pages,
                success=True,
                processing_time=time.time() - start_time,
                images=images,
            )

            # Page-count validation: cross-check the recovered page count against
            # the actual PDF page count (constrained to the --pages subset when
            # given) so a missing/false split is not silently recorded as a
            # complete render with the wrong count. A shortfall is a partial.
            expected = self._expected_page_count(file_path)
            if expected is not None and result.page_count < expected:
                result.error = (
                    f"Recovered {result.page_count} page(s) but the PDF has "
                    f"{expected} (marker dropped or failed to split pages)"
                )
                result.degraded = True  # -> Status.PARTIAL (see OCRResult.status)
            return result
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return OCRResult(
                file_path=file_path,
                pages=[],
                success=False,
                error=str(e),
                processing_time=time.time() - start_time,
            )

    def _expected_page_count(self, file_path: Path) -> int | None:
        """Best-effort expected page count for validation (None if unknown).

        Returns the number of pages marker should have produced: the size of the
        requested ``--pages`` subset when one is given, otherwise the PDF's real
        page count. Returns ``None`` (validation disabled) when the count cannot
        be determined, so a probe failure never turns a good render into a
        spurious partial.
        """
        page_range = self.config.parse_page_range()
        if page_range is not None:
            return len(page_range) or None
        try:
            from marker_ocr.utils import get_pdf_page_count

            count = get_pdf_page_count(file_path)
            return count or None
        except Exception as e:  # pragma: no cover - defensive (corrupt/locked PDF)
            logger.debug(f"Could not determine page count for {file_path}: {e}")
            return None

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
        are normalised to PNG and named ``figure_<N>_page<P>.png``.

        Marker inlines figure references into its body as ``![](<key>)`` where
        ``<key>`` is byte-identical to the image-dict key. We save each figure and
        rewrite those inline ``src`` targets IN PLACE to the resolving
        ``./figures/figure_<N>_page<P>.png`` path. We do NOT append a separate
        ``## Figures`` section: the prior approach left marker's original inline
        link dangling (it pointed at a file that was never written under that
        name) while duplicating a correct link at the end. For a figure that
        FAILS to save we strip its inline reference entirely (rather than leaving
        a dangling ``![](<key>)``), so a PARTIAL document still has no broken
        local image links.
        """
        doc_dir = doc_dir_for(output_root, rel_key)
        doc_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = markdown_path_for(doc_dir, rel_key)

        if result.pages:
            body = assemble_pages(result.page_texts, page_numbers=result.page_numbers)
        else:
            body = "*[OCR Failed]*\n"

        # Save figures and rewrite the inline links the body already carries.
        link_map, failed_keys = self._save_figures(result, doc_dir)
        if link_map:
            body = self._rewrite_inline_figure_links(body, link_map)
        if failed_keys:
            body = self._strip_inline_figure_links(body, failed_keys)

        markdown_path.write_text(body, encoding="utf-8")

        if self.config.verbose:
            console.print(f"[green]Saved:[/green] {markdown_path}")

        return markdown_path

    def _save_figures(self, result: OCRResult, doc_dir: Path) -> tuple[dict[str, str], list[str]]:
        """Persist marker's images as PNG.

        Returns ``(link_map, failed_keys)`` where ``link_map`` is
        ``{marker_key: relative_link}`` for the figures that saved and
        ``failed_keys`` is the marker keys whose save raised.

        Marker's image keys encode the source page (e.g. ``_page_1_Figure_0``)
        AND are byte-identical to the inline ``src`` marker wrote into the body.
        We parse the source page, save under ``figures/figure_<N>_page<P>.png``,
        and return a mapping from the marker key to the resolving relative link so
        the caller can rewrite the inline references in place. The caller MUST also
        drop the inline references for ``failed_keys`` so a save failure does not
        leave a dangling ``![](<key>)`` link in the body.

        Figures are numbered in source-page order (numeric sort, so
        ``_page_10`` follows ``_page_2`` — lexicographic sort mis-numbered docs
        with >9 figure-bearing pages). A save failure flags the result as
        ``degraded`` (-> ``Status.PARTIAL``) and sets ``result.error`` so the
        partial carries a diagnostic instead of recording ``error=None``.
        """
        if not result.images:
            return {}, []

        figures_dir = figures_dir_for(doc_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        link_map: dict[str, str] = {}
        failed_keys: list[str] = []
        figure_counter = 0
        # Sort numerically by (source page, intra-page block id, key) so figure_N
        # tracks source/block order even for docs with >9 figure-bearing pages OR
        # >9 figures on one page (block 10 after block 2, not lexicographically).
        ordered = sorted(
            result.images.items(),
            key=lambda kv: (
                self._page_from_image_name(kv[0]),
                self._block_id_from_image_name(kv[0]),
                kv[0],
            ),
        )
        for img_name, img in ordered:
            page_no = self._page_from_image_name(img_name)
            figure_counter += 1
            filename = figure_filename(figure_counter, page_no)
            img_path = figures_dir / filename
            try:
                pil_img = self._to_pil(img)
                pil_img = self._normalize_image_mode(pil_img)
                pil_img.save(img_path, format="PNG")
                link_map[img_name] = f"./{figures_dir.name}/{filename}"
            except Exception as e:
                logger.warning(f"Failed to save figure {img_name!r}: {e}")
                figure_counter -= 1
                failed_keys.append(img_name)
                # A dropped figure loses content: record the doc as partial, not
                # completed. The caller strips its inline reference so no dangling
                # link survives in the body.
                result.degraded = True

        if failed_keys:
            # Record a diagnostic on the result so the PARTIAL is not stored with
            # error=None (an undiagnosed degraded document).
            detail = f"failed to save {len(failed_keys)} figure(s): " + ", ".join(
                repr(k) for k in failed_keys
            )
            result.error = f"{result.error}; {detail}" if result.error else detail
        return link_map, failed_keys

    @staticmethod
    def _rewrite_inline_figure_links(body: str, link_map: dict[str, str]) -> str:
        """Rewrite marker's inline ``![..](<key>)`` targets to resolving links.

        Marker's inline image ``src`` is byte-identical to the image-dict key. We
        replace each ``(<key>)`` occurrence with ``(<relative figures link>)`` so
        the inline reference resolves on disk. Longest keys are replaced first to
        avoid a shorter key being a substring of a longer one.
        """
        for key in sorted(link_map, key=len, reverse=True):
            body = body.replace(f"({key})", f"({link_map[key]})")
        return body

    @staticmethod
    def _strip_inline_figure_links(body: str, keys: list[str]) -> str:
        """Remove the whole inline image ``![..](<key>)`` for un-saved figures.

        A figure that failed to save has no file on disk; leaving marker's inline
        ``![](<key>)`` would be a dangling local image link (the exact failure the
        rewrite path eliminates). We strip the full inline image markup for each
        failed key so the body carries no broken reference; the loss is recorded
        as ``Status.PARTIAL`` with a diagnostic in metadata. Longest keys first so
        a shorter key is not matched inside a longer one.
        """
        for key in sorted(keys, key=len, reverse=True):
            # Match an optional alt-text image: ![ ... ](<key>)
            pattern = re.compile(r"!\[[^\]]*\]\(" + re.escape(key) + r"\)")
            body = pattern.sub("", body)
        return body

    @staticmethod
    def _page_from_image_name(name: str) -> int:
        """Extract the 1-indexed source page from a marker image key.

        Marker keys look like ``_page_1_Figure_0.jpeg`` where the captured number
        is marker's raw 0-indexed ``page_id`` -- the SAME id the body's
        ``{page_id}`` boundary marker carries. We route it through
        :func:`_marker_page_id_to_source_page`, the identical offset
        :func:`split_marker_pages` applies, so the figure ``page<P>`` tag matches
        the body ``## Page N`` header for the same source page (no off-by-one, and
        page_id 0 vs 1 no longer collide on the first two pages).
        """
        m = _IMG_PAGE_RE.search(name)
        if not m:
            return 1
        return _marker_page_id_to_source_page(int(m.group(1)))

    @staticmethod
    def _block_id_from_image_name(name: str) -> int:
        """Extract the trailing block id from a marker image key for sorting.

        Marker keys end ``..._<BlockType>_<block_id>.<ext>``; the block id is the
        intra-page source/reading order. Used as the secondary sort key so figures
        within one page number in block order (block 10 after block 2), not the
        lexicographic order of the raw string key. Falls back to 0 when absent.
        """
        m = _IMG_BLOCK_RE.search(name)
        return int(m.group(1)) if m else 0

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

    @staticmethod
    def _normalize_image_mode(img: Image.Image) -> Image.Image:
        """Coerce an image to a PNG-safe mode without destroying transparency.

        Modes that carry alpha (``RGBA``/``LA``) or palette transparency (``P``
        with a ``transparency`` entry) are converted to ``RGBA`` so the alpha
        channel survives the PNG save; flattening these to ``RGB`` drops alpha and
        visually corrupts the figure. Other non-RGB modes go to ``RGB``.
        """
        mode = img.mode
        if mode in ("RGB", "RGBA"):
            return img
        has_alpha = mode in ("LA", "PA") or (mode == "P" and "transparency" in img.info)
        return img.convert("RGBA" if has_alpha else "RGB")

    @property
    def fingerprint(self) -> str:
        """Run-config fingerprint consulted on resume.

        Passes the marker run parameters that change *what output a given input
        produces* as RESOLVED output-affecting flags via ``extra`` (the contract's
        v0.1.2 mechanism), so a re-run under a different ``--force-ocr`` /
        ``--pages`` reprocesses rather than silently reusing a cached render. The
        page subset is the NORMALISED page-id list (``parse_page_range``), so
        ``--pages 1,3`` and ``--pages 3,1`` and ``--pages 1-3`` fingerprint by
        their actual effect, not by the raw string. Marker takes no free-text
        prompt and has no task selector.
        """
        page_range = self.config.parse_page_range()
        extra = {
            "force_ocr": bool(self.config.force_ocr),
            "pages": page_range,  # normalised list[int] | None
        }
        return run_fingerprint(model=MODEL, backend=BACKEND, extra=extra)

    def _build_doc_metadata(
        self,
        result: OCRResult,
        file_path: Path,
        markdown_path: Path,
        output_root: Path,
    ) -> DocMetadata:
        """Assemble the per-document metadata record from a result."""
        status = result.status
        # Record the error/diagnostic for any non-clean status (failed/partial).
        error = None if status is Status.COMPLETED else result.error
        # safe_checksum so building the FAILED record for an unreadable input does
        # not itself raise (which would re-trip the SYS-02 abort). An empty-string
        # checksum is a valid str that never matches a real sha256:, so the doc is
        # correctly reprocessed (never skipped as "completed") on a later run.
        checksum = (
            sha256_checksum(file_path)
            if status is Status.COMPLETED
            else (safe_checksum(file_path) or "")
        )
        return DocMetadata(
            status=status,
            checksum=checksum,
            model=MODEL,
            backend=BACKEND,
            processing_time=result.processing_time,
            timestamp=utc_timestamp(),
            output_path=str(markdown_path.relative_to(output_root)),
            pages=result.page_count,
            error=error,
            fingerprint=self.fingerprint,
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

        # Idempotency pre-check uses safe_checksum: an unreadable input yields
        # None (never raises), so we fall through to processing, which records a
        # per-file FAILED instead of aborting the run (the SYS-02 contract).
        checksum = safe_checksum(file_path)
        if (
            not reprocess
            and checksum is not None
            and index.is_completed(rel_key, checksum, fingerprint=self.fingerprint)
        ):
            # The .md is verified on-disk by is_completed, so emitting its path is
            # safe for quiet scripting (the file genuinely exists).
            cached_md = markdown_path_for(doc_dir_for(output_root, rel_key), rel_key)
            console.print(f"[yellow]Already processed:[/yellow] {file_path.name}")
            console.print("[dim]Use --reprocess to force reprocessing[/dim]")
            outcome.add(Status.COMPLETED, output_path=str(cached_md))
            return outcome

        console.print(f"[blue]Processing:[/blue] {file_path}")
        console.print(f"[blue]Output:[/blue] {output_root}\n")

        result = self.process_file(file_path)
        meta, markdown_path = self._persist(result, file_path, output_root, rel_key, index)
        outcome.add(
            meta.status,
            detail=None if meta.status is Status.COMPLETED else rel_key,
            # Only a real (non-placeholder) output is emitted to the scripting
            # surface; a FAILED doc's placeholder .md must not be echoed as a
            # written path that a pipeline would ingest as success.
            output_path=str(markdown_path) if meta.status is not Status.FAILED else None,
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
        # Resolve the output root BEFORE discovery so the contract's
        # iter_input_files can prune the resolved output subtree — the engine
        # never re-ingests its own .md/figure outputs on a re-run (and never
        # excludes a legitimate input merely because a path component is named
        # 'ocr', which would process ZERO files under .../toolkits/ocr/...).
        output_root = resolve_output_root(dir_path, output_path)
        output_root.mkdir(parents=True, exist_ok=True)
        index = RootIndex(output_root)

        files = list(iter_input_files(dir_path, output_root, SUPPORTED_EXTENSIONS))
        if not files:
            console.print("[yellow]No supported files found[/yellow]")
            return outcome

        files_to_process: list[tuple[Path, str]] = []
        for f in files:
            rel_key = relative_key(f, dir_path)
            # safe_checksum (not sha256_checksum) so an input that became
            # unreadable between discovery and this pre-filter yields None instead
            # of raising and aborting the whole batch (SYS-02). A None checksum
            # falls through to processing, which records that one file as FAILED.
            checksum = safe_checksum(f)
            if (
                not reprocess
                and checksum is not None
                and index.is_completed(rel_key, checksum, fingerprint=self.fingerprint)
            ):
                if self.config.verbose:
                    console.print(f"[dim]Skipping: {rel_key}[/dim]")
                cached_md = markdown_path_for(doc_dir_for(output_root, rel_key), rel_key)
                outcome.add(Status.COMPLETED, output_path=str(cached_md))
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
            # Display-only stat, guarded: a race-deleted file must not abort the
            # remaining batch here (SYS-02). process_file re-stats inside its own
            # try/except, so an unreadable file is still recorded as FAILED below.
            try:
                file_size = format_file_size(file_path.stat().st_size)
            except OSError:
                file_size = "?"
            console.print(f"[cyan]{rel_key}[/cyan] ({file_size})")

            result = self.process_file(file_path, show_progress=False)
            meta, markdown_path = self._persist(result, file_path, output_root, rel_key, index)
            outcome.add(
                meta.status,
                detail=None if meta.status is Status.COMPLETED else rel_key,
                output_path=str(markdown_path) if meta.status is not Status.FAILED else None,
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
