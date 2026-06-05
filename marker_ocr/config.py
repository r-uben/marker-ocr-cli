"""Configuration management for Marker OCR CLI."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration settings for Marker OCR."""

    # Processing
    pages: str | None = None
    force_ocr: bool = False
    device: str | None = None
    max_file_size_mb: float = 500.0

    # Output
    output_format: str = "markdown"
    output_dir: Path | None = None
    verbose: bool = False
    quiet: bool = False

    def apply_device(self) -> None:
        """Set TORCH_DEVICE env var before Marker imports.

        Must be called before any marker imports so that Marker's Settings
        picks up the override. On macOS/MPS, defaults to cpu because Surya's
        layout model crashes on Apple Metal.
        """
        if self.device:
            os.environ["TORCH_DEVICE"] = self.device
        elif not os.environ.get("TORCH_DEVICE"):
            # MPS is broken for Surya — default to cpu on Apple Silicon
            import platform

            if platform.processor() == "arm" or platform.machine() == "arm64":
                os.environ["TORCH_DEVICE"] = "cpu"

    def parse_page_range(self) -> list[int] | None:
        """Parse --pages string into list of ints for Marker's page_range."""
        if not self.pages:
            return None

        pages: list[int] = []
        for part in self.pages.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                pages.extend(range(int(start), int(end) + 1))
            else:
                pages.append(int(part))
        return sorted(set(pages))

    def to_marker_config(self) -> dict:
        """Build config dict for Marker's PdfConverter.

        ``paginate_output`` is forced on so the rendered whole-document markdown
        carries marker's per-page boundary markers (``{page_id}`` + a rule). We
        split on those to recover per-page text and re-emit it under the
        canonical ``## Page N`` headers required by the output contract.
        """
        config: dict = {"paginate_output": True}
        page_range = self.parse_page_range()
        if page_range is not None:
            config["page_range"] = page_range
        if self.force_ocr:
            config["force_ocr"] = True
        return config

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration (no env vars needed for local model)."""
        return cls()

    def validate_file_size(self, file_path: Path) -> None:
        """Validate file size is within limits."""
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(
                f"File size ({file_size_mb:.2f} MB) exceeds maximum "
                f"allowed size ({self.max_file_size_mb} MB)"
            )
