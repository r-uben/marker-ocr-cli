# Marker OCR CLI

[![CI](https://github.com/r-uben/marker-ocr-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/r-uben/marker-ocr-cli/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/marker-ocr-cli.svg)](https://badge.fury.io/py/marker-ocr-cli)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A command-line tool for OCR processing using [Marker](https://github.com/datalab-to/marker)'s layout-aware pipeline. Extract text, equations, tables, and figures from PDFs with high accuracy.

## Why

Standard PDF text extraction (`pdftotext`, `PyPDF`) breaks on scanned documents, multi-column layouts, equations, and tables. Cloud OCR services (Mathpix, Google Document AI) work but cost money and send your data off-machine. Marker runs locally, handles complex academic paper layouts (two-column, equations in LaTeX, tables, figures), and outputs clean Markdown. This CLI wraps Marker's Python API into a single command with batch processing, progress tracking, and metadata/checksum management.

## Choosing an OCR tool

This is one of five OCR CLI tools with a shared design: clean Markdown output, batch processing, and figure extraction. Pick based on your constraints:

| Tool | Engine | Runs | Cost | Best for |
|------|--------|------|------|----------|
| [deepseek-ocr-cli](https://github.com/r-uben/deepseek-ocr-cli) | DeepSeek vision | Local (Ollama / vLLM) | Free | General-purpose local OCR with multi-backend flexibility |
| [gemini-ocr-cli](https://github.com/r-uben/gemini-ocr-cli) | Google Gemini | Cloud API | Free tier / Pay-per-use | Fast cloud OCR with concurrent processing |
| **marker-ocr-cli** (this repo) | Marker (Surya + Texify) | Local | Free | Academic papers with equations, tables, complex layouts |
| [mistral-ocr-cli](https://github.com/r-uben/mistral-ocr-cli) | Mistral OCR API | Cloud API | ~$1/1k pages | Structured extraction (tables, headers, footers) |
| [nougat-ocr-cli](https://github.com/r-uben/nougat-ocr-cli) | Meta Nougat | Local (GPU) | Free | Academic papers, GPU-accelerated batch processing |

## Hardware requirements

Marker loads 5 neural network models (layout detection, text detection, OCR recognition, table recognition, OCR error correction). This determines the hardware floor:

| Setup | VRAM/RAM | Speed (approx.) | Notes |
|-------|----------|------------------|-------|
| **NVIDIA GPU (CUDA)** | ~4-5 GB VRAM | ~25 pages/sec (A100/H100) | Recommended. Best throughput. |
| **Apple Silicon (MPS)** | ~5 GB unified memory | Moderate | Surya layout model crashes on MPS, so this CLI defaults to CPU on Apple Silicon. Use `--device mps` to force it (at your own risk). |
| **CPU only** | ~4-5 GB RAM | ~5-10 sec/page | Works fine for small jobs. Default on Apple Silicon. |

Models auto-download on first run (~2 GB download, cached in `~/.cache/`).

## Installation

Requires Python 3.11+.

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
  -o, --output-dir PATH           Output directory (default: <input-parent>/ocr/)
  --pages TEXT                    Page range (e.g., '0-5' or '1,3,5')
  --force-ocr                     Force OCR on all pages regardless of embedded text

  --device [auto|cpu|cuda|mps]    Inference device (default: cpu on Apple Silicon)
  --reprocess                     Reprocess already-processed files
  --dry-run                       List files without loading models
  -q, --quiet                     Suppress logs; emit one written .md path per line
  -v, --verbose                   Enable verbose/debug output
  --info                          Show system and device info
  --version                       Show version
  --help                          Show this message
```

## Output structure

Output follows the shared canonical OCR contract (the `ocr-output-contract`
package): the default root is `<input-parent>/ocr/`, the input subtree is mirrored
and keyed on the input-relative path (so same-named PDFs in different folders never
collide), each document's pages live in one `<stem>.md` under `## Page N` headers
(no YAML frontmatter), and figures are normalised to PNG with links that resolve.

```
ocr/
├── document_name/
│   ├── document_name.md        # OCR markdown, pages under '## Page N'
│   ├── metadata.json           # per-document provenance (status, checksum, ...)
│   └── figures/                # extracted figures, normalised to PNG
│       ├── figure_1_page1.png
│       └── figure_2_page3.png
├── another_document/
│   └── ...
└── metadata.json               # root index keyed by input-relative path
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

- PDF only (Marker processes PDFs natively)
- First run downloads ~2 GB of model weights
- CPU-only is usable but slow for large batches

## License

MIT License - see [LICENSE](LICENSE) for details.
