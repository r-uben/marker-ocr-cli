"""Basic import tests."""


def test_import_package():
    import marker_ocr

    assert hasattr(marker_ocr, "__version__")


def test_import_config():
    from marker_ocr import Config

    assert Config is not None


def test_import_cli():
    from marker_ocr.cli import cli

    assert cli is not None


def test_import_utils():
    from marker_ocr.utils import format_file_size, is_pdf_file

    assert format_file_size is not None
    assert is_pdf_file is not None
