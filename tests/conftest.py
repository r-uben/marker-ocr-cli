"""Pytest configuration and fixtures."""

from pathlib import Path

import pytest


# Mark all tests as unit by default
def pytest_collection_modifyitems(items):
    for item in items:
        if "integration" not in item.keywords:
            item.add_marker(pytest.mark.unit)


# Register custom markers
def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark test as requiring GPU and models")


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a minimal PDF for testing."""
    import pypdf

    pdf_path = tmp_path / "sample.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pdf_path
