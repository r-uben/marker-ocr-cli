"""Tests for utility functions."""

from pathlib import Path

import pytest

from marker_ocr.utils import (
    determine_output_path,
    format_file_size,
    get_supported_files,
    is_pdf_file,
    is_supported_file,
    sanitize_filename,
)


class TestFileTypeDetection:
    """Tests for file type detection functions."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("test.pdf", True),
            ("test.PDF", True),
            ("test.txt", False),
            ("test.docx", False),
            ("test.jpg", False),
            ("test", False),
        ],
    )
    def test_is_supported_file(self, filename: str, expected: bool):
        assert is_supported_file(Path(filename)) == expected

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("test.pdf", True),
            ("test.PDF", True),
            ("document.pdf", True),
            ("test.jpg", False),
            ("test.txt", False),
        ],
    )
    def test_is_pdf_file(self, filename: str, expected: bool):
        assert is_pdf_file(Path(filename)) == expected


class TestSanitizeFilename:
    """Tests for filename sanitization."""

    @pytest.mark.parametrize(
        "input_name,expected",
        [
            ("normal_file", "normal_file"),
            ("file with spaces", "file_with_spaces"),
            ('file<>:"/\\|?*name', "file_name"),
            ("multiple   spaces", "multiple_spaces"),
            ("___leading_trailing___", "leading_trailing"),
            ("", "unnamed"),
        ],
    )
    def test_sanitize_filename(self, input_name: str, expected: str):
        assert sanitize_filename(input_name) == expected

    def test_sanitize_filename_max_length(self):
        long_name = "a" * 300
        result = sanitize_filename(long_name, max_length=200)
        assert len(result) == 200


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


class TestDetermineOutputPath:
    """Tests for output path determination."""

    def test_output_path_for_file(self, tmp_path):
        input_file = tmp_path / "test.pdf"
        input_file.touch()
        result = determine_output_path(input_file)
        assert result == tmp_path / "marker_ocr_output"
        assert result.exists()

    def test_output_path_for_directory(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        result = determine_output_path(input_dir)
        assert result == input_dir / "marker_ocr_output"
        assert result.exists()

    def test_output_path_custom(self, tmp_path):
        input_file = tmp_path / "test.pdf"
        input_file.touch()
        custom_output = tmp_path / "custom_output"
        result = determine_output_path(input_file, output_path=custom_output)
        assert result == custom_output
        assert result.exists()


class TestGetSupportedFiles:
    """Tests for finding supported files."""

    def test_get_supported_files_recursive(self, tmp_path):
        (tmp_path / "root.pdf").touch()
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "nested.pdf").touch()
        (tmp_path / "ignored.txt").touch()
        (tmp_path / "image.png").touch()

        files = get_supported_files(tmp_path, recursive=True)

        assert len(files) == 2
        names = [f.name for f in files]
        assert "root.pdf" in names
        assert "nested.pdf" in names
        assert "ignored.txt" not in names
        assert "image.png" not in names

    def test_get_supported_files_excludes_output_dir(self, tmp_path):
        (tmp_path / "root.pdf").touch()
        output_dir = tmp_path / "marker_ocr_output"
        output_dir.mkdir()
        (output_dir / "output.pdf").touch()

        files = get_supported_files(tmp_path, recursive=True)
        names = [f.name for f in files]
        assert "root.pdf" in names
        assert "output.pdf" not in names

    def test_get_supported_files_sorted(self, tmp_path):
        (tmp_path / "c.pdf").touch()
        (tmp_path / "a.pdf").touch()
        (tmp_path / "b.pdf").touch()

        files = get_supported_files(tmp_path)
        names = [f.name for f in files]
        assert names == ["a.pdf", "b.pdf", "c.pdf"]
