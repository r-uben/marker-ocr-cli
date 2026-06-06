"""Tests for OCR processor module."""

from pathlib import Path
from unittest.mock import patch

from ocr_output_contract import Status

from marker_ocr.config import Config
from marker_ocr.processor import OCRProcessor, OCRResult, split_marker_pages


class TestOCRResult:
    """Tests for OCRResult dataclass."""

    def test_success_true(self):
        result = OCRResult(
            file_path=Path("test.pdf"),
            pages=[(1, "Extracted content")],
            success=True,
            processing_time=1.0,
        )
        assert result.success is True
        assert result.page_count == 1
        assert result.status is Status.COMPLETED
        assert result.text == "Extracted content"
        assert result.page_numbers == [1]
        assert result.page_texts == ["Extracted content"]

    def test_success_false(self):
        result = OCRResult(
            file_path=Path("test.pdf"),
            pages=[],
            success=False,
            error="Processing failed",
            processing_time=1.0,
        )
        assert result.success is False
        assert result.error == "Processing failed"
        assert result.status is Status.FAILED

    def test_all_empty_pages_is_failure(self):
        # Page markers present but no content -> not a silent "completed".
        result = OCRResult(
            file_path=Path("test.pdf"),
            pages=[(1, "   "), (2, "")],
            success=True,
        )
        assert result.has_content is False
        assert result.status is Status.FAILED

    def test_degraded_is_partial(self):
        # Some content recovered but the render is incomplete -> partial.
        result = OCRResult(
            file_path=Path("test.pdf"),
            pages=[(1, "real content")],
            success=True,
            degraded=True,
        )
        assert result.status is Status.PARTIAL

    def test_default_values(self):
        result = OCRResult(file_path=Path("test.pdf"), pages=[(1, "text")], success=True)
        assert result.page_count == 1
        assert result.error is None
        assert result.processing_time == 0.0
        assert result.images == {}
        assert result.degraded is False


class TestSplitMarkerPages:
    """Tests mapping marker's paginated markdown to per-page blocks."""

    def test_no_markers_single_page(self):
        assert split_marker_pages("just one page of text") == [(1, "just one page of text")]

    def test_empty_input(self):
        assert split_marker_pages("") == []

    def test_paginated_markers_split(self):
        # marker emits "{page_id}" + a rule before each page (page_id 0-indexed),
        # always at the start of its own line. Source page numbers are 1-indexed.
        md = "\n\n{0}" + ("-" * 48) + "\n\nPage one body\n\n{1}" + ("-" * 48) + "\n\nPage two body"
        pages = split_marker_pages(md)
        assert pages == [(1, "Page one body"), (2, "Page two body")]

    def test_unanchored_marker_in_body_is_not_a_split(self):
        # A literal '{5}---' appearing mid-line in body text must NOT be treated
        # as a page boundary (the unanchored regex fragmented pages here).
        md = "Body with a literal {5}--- inline reference then more text"
        assert split_marker_pages(md) == [(1, md)]

    def test_real_full_line_marker_after_inline_brace(self):
        # An inline '{5}---' is ignored but a real full-line marker still splits.
        md = (
            "Intro has {5}--- inline.\n\n"
            "{0}" + ("-" * 48) + "\n\nReal page one\n\n"
            "{1}" + ("-" * 48) + "\n\nReal page two"
        )
        pages = split_marker_pages(md)
        assert [n for n, _ in pages] == [1, 2]
        assert "Real page one" in pages[0][1]
        assert "inline" in pages[0][1]  # preamble folded into the first page
        assert "Real page two" in pages[1][1]


class TestOCRProcessorSaveResults:
    """Tests for saving OCR results through the contract (no model loading)."""

    def _make_processor(self, config: Config) -> OCRProcessor:
        with patch.object(OCRProcessor, "_load_models"):
            return OCRProcessor(config)

    def test_save_results_creates_per_document_folder(self, tmp_path, sample_pdf):
        processor = self._make_processor(Config())
        result = OCRResult(
            file_path=sample_pdf, pages=[(1, "Test content")], success=True, processing_time=1.5
        )
        output_root = tmp_path / "ocr"
        output_root.mkdir()
        output_path = processor.save_results(result, output_root, "sample.pdf")

        assert output_path.parent.name == "sample"
        assert output_path.name == "sample.md"
        assert output_path.exists()

    def test_save_results_emits_page_header_no_frontmatter(self, tmp_path, sample_pdf):
        processor = self._make_processor(Config())
        result = OCRResult(
            file_path=sample_pdf, pages=[(1, "Test content")], success=True, processing_time=1.5
        )
        output_root = tmp_path / "ocr"
        output_root.mkdir()
        content = processor.save_results(result, output_root, "sample.pdf").read_text()

        assert content.startswith("## Page 1")
        assert "Test content" in content
        assert "---\n" not in content.split("Test content")[0]  # no YAML frontmatter

    def test_source_page_numbers_in_headers(self, tmp_path, sample_pdf):
        # Under a --pages subset the body headers use marker's source page ids
        # (here 3 and 5), keeping body and figure page numbers aligned.
        processor = self._make_processor(Config())
        result = OCRResult(
            file_path=sample_pdf,
            pages=[(3, "Third page"), (5, "Fifth page")],
            success=True,
        )
        output_root = tmp_path / "ocr"
        output_root.mkdir()
        content = processor.save_results(result, output_root, "sample.pdf").read_text()
        assert "## Page 3" in content
        assert "## Page 5" in content
