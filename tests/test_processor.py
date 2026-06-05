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
            pages=["Extracted content"],
            success=True,
            processing_time=1.0,
        )
        assert result.success is True
        assert result.page_count == 1
        assert result.status is Status.COMPLETED
        assert result.text == "Extracted content"

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

    def test_default_values(self):
        result = OCRResult(file_path=Path("test.pdf"), pages=["text"], success=True)
        assert result.page_count == 1
        assert result.error is None
        assert result.processing_time == 0.0
        assert result.images == {}


class TestSplitMarkerPages:
    """Tests mapping marker's paginated markdown to per-page blocks."""

    def test_no_markers_single_page(self):
        assert split_marker_pages("just one page of text") == ["just one page of text"]

    def test_empty_input(self):
        assert split_marker_pages("") == []

    def test_paginated_markers_split(self):
        # marker emits "{page_id}" + a rule before each page (page_id 0-indexed).
        md = "\n\n{0}" + ("-" * 48) + "\n\nPage one body\n\n{1}" + ("-" * 48) + "\n\nPage two body"
        pages = split_marker_pages(md)
        assert pages == ["Page one body", "Page two body"]


class TestOCRProcessorSaveResults:
    """Tests for saving OCR results through the contract (no model loading)."""

    def _make_processor(self, config: Config) -> OCRProcessor:
        with patch.object(OCRProcessor, "_load_models"):
            return OCRProcessor(config)

    def test_save_results_creates_per_document_folder(self, tmp_path, sample_pdf):
        processor = self._make_processor(Config())
        result = OCRResult(
            file_path=sample_pdf, pages=["Test content"], success=True, processing_time=1.5
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
            file_path=sample_pdf, pages=["Test content"], success=True, processing_time=1.5
        )
        output_root = tmp_path / "ocr"
        output_root.mkdir()
        content = processor.save_results(result, output_root, "sample.pdf").read_text()

        assert content.startswith("## Page 1")
        assert "Test content" in content
        assert "---\n" not in content.split("Test content")[0]  # no YAML frontmatter
