"""Tests for CLI module."""

import shutil

import pytest
from click.testing import CliRunner

from marker_ocr import __version__
from marker_ocr.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestCLIBasics:
    """Tests for basic CLI functionality."""

    def test_cli_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Marker OCR" in result.output

    def test_cli_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_cli_shows_options(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "--output-dir" in result.output
        assert "--dry-run" in result.output
        assert "--quiet" in result.output
        assert "--pages" in result.output
        assert "--force-ocr" in result.output
        assert "--reprocess" in result.output


class TestDryRun:
    """Tests for --dry-run mode."""

    def test_dry_run_single_file(self, runner, sample_pdf):
        result = runner.invoke(cli, [str(sample_pdf), "--dry-run"])
        assert result.exit_code == 0
        assert "dry run" in result.output.lower()

    def test_dry_run_directory(self, runner, tmp_path, sample_pdf):
        target = tmp_path / "docs"
        target.mkdir()
        shutil.copy(sample_pdf, target / "test.pdf")
        result = runner.invoke(cli, [str(target), "--dry-run"])
        assert result.exit_code == 0

    def test_dry_run_no_model_loading(self, runner, sample_pdf):
        result = runner.invoke(cli, [str(sample_pdf), "--dry-run"])
        assert result.exit_code == 0


class TestProcessCommand:
    """Tests for file processing."""

    def test_process_missing_file(self, runner):
        result = runner.invoke(cli, ["nonexistent.pdf"])
        assert result.exit_code != 0

    def test_missing_input_path_is_usage_error(self, runner):
        # A bare invocation with no INPUT_PATH (and no --info) must be a usage
        # error (exit 2), not a successful nested --help (exit 0).
        result = runner.invoke(cli, [])
        assert result.exit_code != 0
        assert "INPUT_PATH" in result.output

    def test_quiet_missing_input_path_is_usage_error(self, runner):
        # Even under --quiet a missing INPUT_PATH must fail, not silently exit 0.
        result = runner.invoke(cli, ["--quiet"])
        assert result.exit_code != 0


class TestInfoFlag:
    """Tests for --info flag."""

    def test_info_shows_system_info(self, runner):
        result = runner.invoke(cli, ["--info", "."])
        assert result.exit_code == 0
        assert "System Information" in result.output
        assert "Python" in result.output


class TestQuietMode:
    """Tests for --quiet mode."""

    def test_quiet_dry_run(self, runner, sample_pdf):
        result = runner.invoke(cli, [str(sample_pdf), "--dry-run", "--quiet"])
        assert "Marker OCR" not in result.output
