# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-12

### Added

- `--device` flag -- choose inference device (auto/cpu/cuda/mps)
- Auto-detect Apple Silicon and default to CPU (Surya layout model crashes on MPS)

### Fixed

- Pass `--pages` and `--force-ocr` through to Marker's PdfConverter config
- Page count now correctly read from converter after processing
- Show active device in model loading message

## [0.1.0] - 2026-03-12

### Added

- Initial release
- PDF OCR using Marker's layout-aware pipeline (Surya + Texify)
- Per-document output folders (`output/doc_name/doc_name.md + figures/`)
- Clean markdown output with extracted figures
- `--dry-run` flag -- list files without loading models
- `--quiet` / `-q` flag -- suppress output for scripting
- `--reprocess` flag -- force reprocessing of already-processed files
- `--info` flag -- show device and system information
- `--pages` flag -- process specific page ranges
- `--force-ocr` flag -- force OCR on all pages
- MetadataManager with SHA256 checksums for change detection
- Rich console output with progress indicators
- `.github/workflows/ci.yml` -- CI pipeline (Python 3.11/3.12/3.13)
- `.pre-commit-config.yaml` -- ruff lint + format hooks
