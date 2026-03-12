"""Tests for configuration module."""

import pytest

from marker_ocr.config import Config


class TestConfigDefaults:
    """Tests for default configuration values."""

    def test_default_pages_none(self):
        config = Config()
        assert config.pages is None

    def test_default_force_ocr_false(self):
        config = Config()
        assert config.force_ocr is False

    def test_default_max_file_size(self):
        config = Config()
        assert config.max_file_size_mb == 500.0

    def test_default_output_format(self):
        config = Config()
        assert config.output_format == "markdown"

    def test_default_quiet(self):
        config = Config()
        assert config.quiet is False

    def test_default_verbose(self):
        config = Config()
        assert config.verbose is False


class TestConfigCustom:
    """Tests for custom configuration."""

    def test_custom_pages(self):
        config = Config(pages="0-5")
        assert config.pages == "0-5"

    def test_custom_force_ocr(self):
        config = Config(force_ocr=True)
        assert config.force_ocr is True

    def test_from_env_returns_defaults(self):
        config = Config.from_env()
        assert config.pages is None


class TestConfigFileValidation:
    """Tests for file validation."""

    def test_validate_file_size_passes(self, tmp_path):
        config = Config()
        test_file = tmp_path / "small.txt"
        test_file.write_text("small content")
        config.validate_file_size(test_file)

    def test_validate_file_size_raises_for_large_file(self, tmp_path):
        config = Config(max_file_size_mb=0.0001)
        test_file = tmp_path / "large.txt"
        test_file.write_text("x" * 1000)
        with pytest.raises(ValueError, match="exceeds maximum"):
            config.validate_file_size(test_file)
