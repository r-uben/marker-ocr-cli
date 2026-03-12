"""Marker OCR CLI - PDF text extraction using Marker's layout-aware pipeline."""

__version__ = "0.2.0"

from marker_ocr.config import Config
from marker_ocr.processor import OCRProcessor

__all__ = ["OCRProcessor", "Config", "__version__"]
