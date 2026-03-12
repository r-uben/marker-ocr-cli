# Marker OCR CLI

[![CI](https://github.com/r-uben/marker-ocr-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/r-uben/marker-ocr-cli/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/marker-ocr-cli.svg)](https://badge.fury.io/py/marker-ocr-cli)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A command-line tool for OCR processing using [Marker](https://github.com/VikParuchuri/marker)'s layout-aware pipeline. Extract text, equations, tables, and figures from PDFs with high accuracy.

## Installation

Requires Python 3.11+ and a GPU (recommended).

```bash
pip install marker-ocr-cli
```

Or from source:

```bash
git clone https://github.com/r-uben/marker-ocr-cli.git
cd marker-ocr-cli
uv sync
```

## Quick start

```bash
# Process a single file
marker-ocr paper.pdf

# Process a directory
marker-ocr ./papers/ -o ./results/

# Preview what would be processed (no model loading)
marker-ocr ./papers/ --dry-run

# Process specific pages
marker-ocr paper.pdf --pages 0-5

# Force OCR on all pages
marker-ocr paper.pdf --force-ocr
```

## Options

```
Usage: marker-ocr [OPTIONS] INPUT_PATH

Options:
  -o, --output-dir PATH           Output directory (default: <input_dir>/marker_ocr_output/)
  --pages TEXT                    Page range (e.g., '0-5' or '1,3,5')
  --force-ocr                     Force OCR on all pages regardless of embedded text

  --device [auto|cpu|cuda|mps]    Inference device (default: cpu on Apple Silicon)
  --reprocess                     Reprocess already-processed files
  --dry-run                       List files without loading models
  -q, --quiet                     Suppress all output except errors
  -v, --verbose                   Enable verbose/debug output
  --info                          Show system and device info
  --version                       Show version
  --help                          Show this message
```

## Output structure

```
marker_ocr_output/
├── document_name/
│   ├── document_name.md        # OCR markdown (clean text only)
│   └── figures/                # extracted figures
│       ├── figure_1.png
│       └── figure_2.png
├── another_document/
│   └── ...
└── metadata.json               # processing stats, checksums, file list
```

## How it works

Marker uses a pipeline of specialized models rather than a single end-to-end model:

- **Surya** -- layout detection and reading order
- **Surya OCR** -- text recognition
- **Texify** -- equation detection and LaTeX conversion

This approach is faster and more accurate than single-model solutions, especially for academic papers with complex layouts, equations, and tables.

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy marker_ocr/ --ignore-missing-imports
```

## Limitations

- Supported formats: PDF only (Marker processes PDFs natively)
- Models: ~4-5 GB VRAM (auto-downloads on first run)
- GPU recommended for reasonable speed (supports CUDA and MPS)

## License

MIT License - see [LICENSE](LICENSE) for details.
