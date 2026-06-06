"""Utility functions for Marker OCR CLI.

Input discovery is owned by the shared ``ocr-output-contract`` package
(:func:`ocr_output_contract.iter_input_files`), which excludes the resolved
output root from the scan. This module keeps only the marker-specific helpers:
file-type checks, size formatting, page counting, and logging setup.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Supported input file extensions (lower-case, leading dot) handed to the
# contract's discovery primitive.
SUPPORTED_EXTENSIONS = {".pdf"}


def setup_logging(level: str = "INFO", verbose: bool = False) -> None:
    """Configure logging for the application."""
    log_level = logging.DEBUG if verbose else getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def is_pdf_file(file_path: Path) -> bool:
    """Check if file is a PDF."""
    return file_path.suffix.lower() == ".pdf"


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def get_pdf_page_count(pdf_path: Path) -> int:
    """Get page count of a PDF."""
    import pypdf

    reader = pypdf.PdfReader(pdf_path)
    return len(reader.pages)
