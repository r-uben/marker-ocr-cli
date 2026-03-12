"""Tests for OCR processor module."""

from pathlib import Path
from unittest.mock import patch

from marker_ocr.config import Config
from marker_ocr.processor import OCRProcessor, OCRResult


class TestOCRResult:
    """Tests for OCRResult dataclass."""

    def test_success_true(self):
        result = OCRResult(
            file_path=Path("test.pdf"),
            text="Extracted content",
            success=True,
            pages=5,
            processing_time=1.0,
        )
        assert result.success is True
        assert result.pages == 5

    def test_success_false(self):
        result = OCRResult(
            file_path=Path("test.pdf"),
            text="",
            success=False,
            error="Processing failed",
            processing_time=1.0,
        )
        assert result.success is False
        assert result.error == "Processing failed"

    def test_default_values(self):
        result = OCRResult(file_path=Path("test.pdf"), text="text", success=True)
        assert result.pages == 0
        assert result.error is None
        assert result.processing_time == 0.0
        assert result.images == {}


class TestOCRProcessorSaveResults:
    """Tests for saving OCR results (no model loading needed)."""

    def _make_processor(self, config: Config) -> OCRProcessor:
        """Create a processor with mocked model loading."""
        with patch.object(OCRProcessor, "_load_models"):
            return OCRProcessor(config)

    def test_save_results_creates_per_document_folder(self, tmp_path, sample_pdf):
        processor = self._make_processor(Config())
        result = OCRResult(
            file_path=sample_pdf, text="Test content", success=True, processing_time=1.5
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        output_path = processor.save_results(result, output_dir)

        assert output_path.parent.name == "sample"
        assert output_path.name == "sample.md"
        assert output_path.exists()

    def test_save_results_clean_markdown(self, tmp_path, sample_pdf):
        processor = self._make_processor(Config())
        result = OCRResult(
            file_path=sample_pdf, text="Test content", success=True, processing_time=1.5
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        output_path = processor.save_results(result, output_dir)
        content = output_path.read_text()

        assert content == "Test content"

    def test_save_results_handles_failure(self, tmp_path, sample_pdf):
        processor = self._make_processor(Config())
        result = OCRResult(
            file_path=sample_pdf,
            text="",
            success=False,
            error="Model error",
            processing_time=1.0,
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        output_path = processor.save_results(result, output_dir)
        content = output_path.read_text()
        assert "OCR Failed" in content
        assert "Model error" in content
