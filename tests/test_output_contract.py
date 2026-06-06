"""Engine-level conformance tests: marker's REAL output vs the shared contract.

The contract *primitives* (path/key computation, page assembly, metadata writers,
the exit-code policy) are unit-tested inside the ``ocr-output-contract`` package
itself, so they are NOT re-tested here.

What stays here is the engine-side proof: run marker's actual processor (with the
marker *converter* boundary mocked, so no GPU/model is needed) over real
multi-page PDFs and assert the produced output tree conforms to the family-wide
contract via the package's reusable
:func:`ocr_output_contract.conformance.assert_conforms` harness. Cases cover a
multi-page success, a conversion failure (status=failed, nonzero exit), and a
figure whose markdown link actually RESOLVES on disk.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pypdf
import pytest
from ocr_output_contract.conformance import ExpectedDoc, assert_conforms
from PIL import Image

from marker_ocr.config import Config
from marker_ocr.processor import OCRProcessor

PAGE_RULE = "-" * 48  # marker's default page_separator


def _make_pdf(path: Path, n_pages: int = 3) -> Path:
    writer = pypdf.PdfWriter()
    for _ in range(n_pages):
        writer.add_blank_page(width=612, height=792)
    with open(path, "wb") as f:
        writer.write(f)
    return path


def _paginated_markdown(n_pages: int, inline_images: dict[int, str] | None = None) -> str:
    """Build marker-style paginated markdown ({page_id} + rule before each page).

    ``inline_images`` maps a 1-indexed page number to an inline image src exactly
    as marker emits it (``![](<dict-key>)``), so tests can reproduce marker's real
    dangling-inline-link behaviour: the body references the image-dict KEY, which
    must be rewritten in place to the saved figures/ path.
    """
    inline_images = inline_images or {}
    parts = []
    for i in range(n_pages):
        body = f"Body text for page {i + 1}."
        src = inline_images.get(i + 1)
        if src:
            body += f"\n\n![]({src})"
        parts.append(f"\n\n{{{i}}}{PAGE_RULE}\n\n{body}")
    return "".join(parts)


def _rendered(markdown: str, images: dict | None = None) -> MagicMock:
    out = MagicMock()
    out.markdown = markdown
    out.images = images or {}
    return out


@pytest.fixture
def processor():
    """A processor with the model-loading boundary stubbed out."""
    with patch.object(OCRProcessor, "_load_models"):
        proc = OCRProcessor(Config())
    proc._converter = MagicMock()
    return proc


class TestProcessorConformance:
    """Marker's real output tree conforms to the contract (via assert_conforms)."""

    def test_multipage_output_conforms(self, processor, tmp_path):
        pdf = _make_pdf(tmp_path / "sample.pdf", n_pages=3)
        out = tmp_path / "out"
        processor._converter.return_value = _rendered(_paginated_markdown(3))

        outcome = processor.process(pdf, output_path=out)
        assert outcome.exit_code == 0

        assert_conforms(
            out,
            [ExpectedDoc(rel_key="sample.pdf", pages=3, status="completed")],
            require_failures_nonzero_exit=outcome.exit_code != 0,
        )
        body = (out / "sample" / "sample.md").read_text()
        assert "Body text for page 1." in body
        assert "Body text for page 3." in body

    def test_metadata_provenance_fields(self, processor, tmp_path):
        import json

        pdf = _make_pdf(tmp_path / "sample.pdf", n_pages=2)
        out = tmp_path / "out"
        processor._converter.return_value = _rendered(_paginated_markdown(2))

        processor.process(pdf, output_path=out)
        assert_conforms(out, [ExpectedDoc(rel_key="sample.pdf", pages=2, status="completed")])

        doc_meta = json.loads((out / "sample" / "metadata.json").read_text())
        assert doc_meta["backend"] == "marker-pdf"
        assert doc_meta["model"] == "marker"
        assert doc_meta["output_path"] == "sample/sample.md"
        assert doc_meta["key"] == "sample.pdf"

    def test_nested_batch_conforms(self, processor, tmp_path):
        root = tmp_path / "in"
        (root / "a").mkdir(parents=True)
        (root / "b").mkdir(parents=True)
        _make_pdf(root / "a" / "intro.pdf", n_pages=1)
        _make_pdf(root / "b" / "intro.pdf", n_pages=1)
        out = tmp_path / "out"
        processor._converter.return_value = _rendered(_paginated_markdown(1))

        processor.process(root, output_path=out)

        # Both same-basename docs survive in distinct mirrored subtrees, conforming.
        assert_conforms(
            out,
            [
                ExpectedDoc(rel_key="a/intro.pdf", pages=1, status="completed"),
                ExpectedDoc(rel_key="b/intro.pdf", pages=1, status="completed"),
            ],
        )

    def test_total_failure_conforms_with_failed_status(self, processor, tmp_path):
        pdf = _make_pdf(tmp_path / "sample.pdf", n_pages=2)
        out = tmp_path / "out"
        processor._converter.side_effect = RuntimeError("marker conversion blew up")

        outcome = processor.process(pdf, output_path=out)
        # A conversion failure must propagate to a nonzero exit.
        assert outcome.exit_code != 0

        assert_conforms(
            out,
            [ExpectedDoc(rel_key="sample.pdf", status="failed")],
            require_failures_nonzero_exit=True,
        )

    def test_inline_figure_link_rewritten_and_resolves(self, processor, tmp_path):
        """Marker's inline ![](<key>) must be REWRITTEN in place to a resolving link.

        This is the HIGH blocker. Marker inlines the image-dict key into the body;
        the old code left that dangling and appended a duplicate ## Figures link.
        The fix rewrites the inline src in place. The v0.1.1 conformance harness
        now resolves EVERY inline image link, so a dangling link FAILS
        ``assert_conforms`` — this test would have caught the bug.
        """
        pdf = _make_pdf(tmp_path / "sample.pdf", n_pages=2)
        out = tmp_path / "out"

        # Marker hands back a PIL image keyed with its page-encoding name, AND
        # inlines that same key into page 2's body (the real marker behaviour).
        marker_key = "_page_2_Figure_0.jpeg"
        img = Image.new("RGB", (20, 20), color="blue")
        rendered = _rendered(
            _paginated_markdown(2, inline_images={2: marker_key}),
            images={marker_key: img},
        )
        processor._converter.return_value = rendered

        outcome = processor.process(pdf, output_path=out)
        assert outcome.exit_code == 0

        # Conformance (v0.1.1) resolves every inline image link on disk; this
        # would raise if the dangling marker key survived in the body.
        assert_conforms(
            out,
            [ExpectedDoc(rel_key="sample.pdf", pages=2, status="completed", figures=[(1, 2)])],
        )

        doc_dir = out / "sample"
        body = doc_dir / "sample.md"
        text = body.read_text()
        # The original dangling marker key must be GONE from the body...
        assert marker_key not in text
        # ...rewritten in place to the resolving figures/ link...
        assert "./figures/figure_1_page2.png" in text
        # ...and there must be NO appended '## Figures' section (rewrite, not append).
        assert "## Figures" not in text
        # The rewritten inline link must point at a file that actually exists.
        link_target = doc_dir / "figures" / "figure_1_page2.png"
        assert link_target.exists()
        assert (body.parent / Path("./figures/figure_1_page2.png")).resolve() == (
            link_target.resolve()
        )
        # And the bytes are a real PNG.
        assert Image.open(io.BytesIO(link_target.read_bytes())).format == "PNG"

    def test_dangling_inline_link_fails_conformance(self, processor, tmp_path):
        """Sanity check the harness: a body referencing an unsaved image FAILS.

        Directly write a body with a dangling inline image link and assert the
        v0.1.1 conformance harness rejects it, proving the link-resolution guard
        that protects the rewrite above is real (not vacuously satisfied).
        """
        from ocr_output_contract import (
            DocMetadata,
            RootIndex,
            Status,
            doc_dir_for,
            markdown_path_for,
            sha256_checksum,
            utc_timestamp,
            write_doc_metadata,
        )
        from ocr_output_contract.conformance import ConformanceError

        out = tmp_path / "out"
        rel_key = "sample.pdf"
        pdf = _make_pdf(tmp_path / "sample.pdf", n_pages=1)
        doc_dir = doc_dir_for(out, rel_key)
        doc_dir.mkdir(parents=True)
        md = markdown_path_for(doc_dir, rel_key)
        # Body references an image that was never written under that name.
        md.write_text("## Page 1\n\n![](missing_figure.png)\n", encoding="utf-8")
        meta = DocMetadata(
            status=Status.COMPLETED,
            checksum=sha256_checksum(pdf),
            model="marker",
            backend="marker-pdf",
            processing_time=0.1,
            timestamp=utc_timestamp(),
            output_path=str(md.relative_to(out)),
            pages=1,
        )
        write_doc_metadata(doc_dir, rel_key, meta)
        RootIndex(out).record(rel_key, meta)

        with pytest.raises(ConformanceError):
            assert_conforms(out, [ExpectedDoc(rel_key=rel_key, pages=1, status="completed")])

    def test_quiet_emits_written_paths(self, processor, tmp_path):
        """Quiet scripting contract: the written .md path is recoverable from outputs."""
        pdf = _make_pdf(tmp_path / "sample.pdf", n_pages=1)
        out = tmp_path / "out"
        processor._converter.return_value = _rendered(_paginated_markdown(1))

        outcome = processor.process(pdf, output_path=out)
        assert outcome.outputs == [str(out / "sample" / "sample.md")]


class TestPageCountValidation:
    """Page-count validation: a short render is recorded partial, not completed."""

    def test_page_shortfall_is_partial(self, processor, tmp_path):
        # 3-page PDF but marker only recovers 1 page -> partial (not silent
        # "completed" with the wrong count).
        pdf = _make_pdf(tmp_path / "sample.pdf", n_pages=3)
        out = tmp_path / "out"
        processor._converter.return_value = _rendered(_paginated_markdown(1))

        outcome = processor.process(pdf, output_path=out)
        assert outcome.partial == 1
        assert outcome.exit_code != 0
        assert_conforms(
            out,
            [ExpectedDoc(rel_key="sample.pdf", pages=1, status="partial")],
            require_failures_nonzero_exit=outcome.exit_code != 0,
        )

    def test_full_render_is_completed(self, processor, tmp_path):
        pdf = _make_pdf(tmp_path / "sample.pdf", n_pages=3)
        out = tmp_path / "out"
        processor._converter.return_value = _rendered(_paginated_markdown(3))

        outcome = processor.process(pdf, output_path=out)
        assert outcome.completed == 1
        assert outcome.exit_code == 0

    def test_all_empty_pages_is_failure(self, processor, tmp_path):
        # Markers present but no body content -> failure, never a silent success.
        pdf = _make_pdf(tmp_path / "sample.pdf", n_pages=1)
        out = tmp_path / "out"
        empty = f"\n\n{{0}}{PAGE_RULE}\n\n   "
        processor._converter.return_value = _rendered(empty)

        outcome = processor.process(pdf, output_path=out)
        assert outcome.exit_code != 0
        assert_conforms(
            out,
            [ExpectedDoc(rel_key="sample.pdf", status="failed")],
            require_failures_nonzero_exit=True,
        )


class TestFigureNumbering:
    """Figures are numbered in source-page order, even for >9 figure pages."""

    def test_numeric_sort_across_many_pages(self, processor, tmp_path):
        # 12 pages, each with one figure. Lexicographic sort would put
        # _page_10 before _page_2; numeric sort keeps page order.
        n = 12
        pdf = _make_pdf(tmp_path / "sample.pdf", n_pages=n)
        out = tmp_path / "out"
        images = {}
        inline = {}
        for p in range(1, n + 1):
            key = f"_page_{p}_Figure_0.jpeg"
            images[key] = Image.new("RGB", (8, 8), color="red")
            inline[p] = key
        processor._converter.return_value = _rendered(
            _paginated_markdown(n, inline_images=inline), images=images
        )

        outcome = processor.process(pdf, output_path=out)
        assert outcome.exit_code == 0

        figures = out / "sample" / "figures"
        # figure_N must map to page N (numbering follows page order).
        for p in range(1, n + 1):
            assert (figures / f"figure_{p}_page{p}.png").exists()
        # The harness resolves every inline link and figure naming.
        assert_conforms(
            out,
            [
                ExpectedDoc(
                    rel_key="sample.pdf",
                    pages=n,
                    status="completed",
                    figures=[(p, p) for p in range(1, n + 1)],
                )
            ],
        )

    def test_figure_save_failure_is_partial(self, processor, tmp_path):
        # A figure that cannot be saved degrades the doc to partial (not a silent
        # completed with a dropped figure). Use a value _to_pil cannot coerce.
        pdf = _make_pdf(tmp_path / "sample.pdf", n_pages=1)
        out = tmp_path / "out"
        processor._converter.return_value = _rendered(
            _paginated_markdown(1, inline_images={1: "_page_1_Figure_0.jpeg"}),
            images={"_page_1_Figure_0.jpeg": object()},  # un-coercible -> save fails
        )

        outcome = processor.process(pdf, output_path=out)
        assert outcome.partial == 1
        assert outcome.exit_code != 0


class TestRGBATransparency:
    """Transparency-bearing images keep their alpha (RGBA), not flattened to RGB."""

    def test_palette_transparency_preserved(self, processor, tmp_path):
        pdf = _make_pdf(tmp_path / "sample.pdf", n_pages=1)
        out = tmp_path / "out"
        # A palette image with a transparency entry should become RGBA on save.
        p_img = Image.new("P", (10, 10))
        p_img.info["transparency"] = 0
        processor._converter.return_value = _rendered(
            _paginated_markdown(1, inline_images={1: "_page_1_Figure_0.png"}),
            images={"_page_1_Figure_0.png": p_img},
        )

        processor.process(pdf, output_path=out)
        saved = out / "sample" / "figures" / "figure_1_page1.png"
        assert saved.exists()
        assert Image.open(saved).mode == "RGBA"


class TestOutputRootSelfIngestion:
    """A re-run never re-ingests its own ocr/ outputs (default nested root)."""

    def test_rerun_excludes_output_root(self, processor, tmp_path):
        # Default output root is <input>/ocr/, nested in the scanned tree.
        in_dir = tmp_path / "papers"
        in_dir.mkdir()
        _make_pdf(in_dir / "a.pdf", n_pages=1)
        processor._converter.return_value = _rendered(_paginated_markdown(1))

        first = processor.process(in_dir)  # no -o -> default <input>/ocr/
        assert first.completed == 1
        out_root = in_dir / "ocr"
        assert out_root.is_dir()

        # A second run must NOT discover the .md/figure outputs under ocr/ as
        # fresh inputs (the figure/markdown files live there). It should find
        # only a.pdf again (and skip it as already-completed).
        second = processor.process(in_dir)
        # Only the one real input is accounted for; nothing from ocr/ is ingested.
        total = second.completed + second.failed + second.partial
        assert total == 1
        # The root index must contain only the real input key.
        from ocr_output_contract import RootIndex

        idx = RootIndex(out_root)
        assert set(idx.files.keys()) == {"a.pdf"}
