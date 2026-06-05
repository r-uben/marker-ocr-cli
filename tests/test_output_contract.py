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


def _paginated_markdown(n_pages: int) -> str:
    """Build marker-style paginated markdown ({page_id} + rule before each page)."""
    parts = []
    for i in range(n_pages):
        parts.append(f"\n\n{{{i}}}{PAGE_RULE}\n\nBody text for page {i + 1}.")
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

    def test_figure_link_resolves(self, processor, tmp_path):
        """A saved figure must use figure_<N>_page<P>.png AND its md link must resolve."""
        pdf = _make_pdf(tmp_path / "sample.pdf", n_pages=2)
        out = tmp_path / "out"

        # Marker hands back a PIL image keyed with its page-encoding name.
        img = Image.new("RGB", (20, 20), color="blue")
        rendered = _rendered(
            _paginated_markdown(2),
            images={"_page_2_Figure_0.jpeg": img},
        )
        processor._converter.return_value = rendered

        outcome = processor.process(pdf, output_path=out)
        assert outcome.exit_code == 0

        # Contract-conformant figure naming (page parsed from the marker key -> 2).
        assert_conforms(
            out,
            [ExpectedDoc(rel_key="sample.pdf", pages=2, status="completed", figures=[(1, 2)])],
        )

        doc_dir = out / "sample"
        body = doc_dir / "sample.md"
        text = body.read_text()
        # The link in the markdown must point at a file that actually exists.
        assert "figure_1_page2.png" in text
        link_target = doc_dir / "figures" / "figure_1_page2.png"
        assert link_target.exists()
        # Resolve the markdown link relative to the .md file and confirm it lands.
        # Links are of the form ![...](./figures/figure_1_page2.png)
        rel = "./figures/figure_1_page2.png"
        assert (body.parent / Path(rel)).resolve() == link_target.resolve()
        # And the bytes are a real PNG.
        assert Image.open(io.BytesIO(link_target.read_bytes())).format == "PNG"

    def test_quiet_emits_written_paths(self, processor, tmp_path):
        """Quiet scripting contract: the written .md path is recoverable from outputs."""
        pdf = _make_pdf(tmp_path / "sample.pdf", n_pages=1)
        out = tmp_path / "out"
        processor._converter.return_value = _rendered(_paginated_markdown(1))

        outcome = processor.process(pdf, output_path=out)
        assert outcome.outputs == [str(out / "sample" / "sample.md")]
