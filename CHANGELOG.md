# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Adopt `ocr-output-contract` v0.1.2 (richer `run_fingerprint(extra=...)`,
  `safe_checksum`); re-locked `uv.lock` to the v0.1.2 pin.

### Fixed

- Figure `page<P>` tag was off-by-one vs the body `## Page N` header in the
  default whole-document case. Marker-pdf shares one raw 0-indexed `page_id`
  between the markdown boundary marker and the image-dict key; both the body and
  the figure tag now route through a single `page_id + 1` helper so a figure's
  page tag matches the body page header for the same source page. Conformance
  fixtures made byte-faithful to marker's shared-`page_id` keys, and a regression
  test asserts figure-page-tag == body-page-number.
- Packaged `marker-ocr` binary exited 0 on missing required input. The `main()`
  no-arg `--help` rewrite is removed; a bare invocation (and any flags without an
  `INPUT_PATH`) now exits nonzero via Click's usage error, while
  `--help`/`--info`/`--version` stay exit 0.
- `--info` falsely reported "marker-pdf not installed" (probed the absent
  `marker.__version__`); now uses `importlib.metadata.version("marker-pdf")`.
- Figure-save failure no longer leaves a dangling inline link or an undiagnosed
  partial: the failed figure's inline reference is stripped from the body and a
  descriptive diagnostic is recorded in `metadata.error` (status stays PARTIAL).
- Idempotency pre-check uses `safe_checksum`, and the batch loop's display
  `stat()` is guarded, so an input that became unreadable between discovery and
  processing is recorded as a per-file failure instead of aborting the run.
- Idempotency fingerprint now passes the resolved output-affecting flags
  (`force_ocr`, normalised `--pages` subset) via `run_fingerprint(extra=...)`.
- Intra-page figure ordering is numeric on the block id (figure 10 after 2),
  not lexicographic.

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
