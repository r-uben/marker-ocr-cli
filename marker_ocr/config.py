"""Configuration management for Marker OCR CLI."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration settings for Marker OCR."""

    # Processing
    pages: str | None = None
    force_ocr: bool = False
    max_file_size_mb: float = 500.0

    # Output
    output_format: str = "markdown"
    output_dir: Path | None = None
    verbose: bool = False
    quiet: bool = False

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
