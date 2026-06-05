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
    from marker_ocr.utils import get_supported_files, sanitize_filename

    assert get_supported_files is not None
    assert sanitize_filename is not None
