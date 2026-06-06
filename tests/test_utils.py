"""Tests for utility functions.

Input discovery (recursive walk + output-root exclusion) now lives in the shared
``ocr-output-contract`` package and is unit-tested there; the engine-side proof
that discovery excludes the resolved output root lives in
``test_output_contract.py``. What stays here is the marker-specific surface:
file-type detection and size formatting.
"""

from pathlib import Path

import pytest

from marker_ocr.utils import (
    format_file_size,
    is_pdf_file,
)


class TestFileTypeDetection:
    """Tests for file type detection functions."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("test.pdf", True),
            ("test.PDF", True),
            ("document.pdf", True),
            ("test.jpg", False),
            ("test.txt", False),
            ("test", False),
        ],
    )
    def test_is_pdf_file(self, filename: str, expected: bool):
        assert is_pdf_file(Path(filename)) == expected


class TestFormatFileSize:
    """Tests for file size formatting."""

    @pytest.mark.parametrize(
        "size_bytes,expected",
        [
            (0, "0.0 B"),
            (500, "500.0 B"),
            (1024, "1.0 KB"),
            (1536, "1.5 KB"),
            (1048576, "1.0 MB"),
            (1073741824, "1.0 GB"),
        ],
    )
    def test_format_file_size(self, size_bytes: int, expected: str):
        assert format_file_size(size_bytes) == expected
