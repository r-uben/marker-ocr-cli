"""Tests for metadata tracking module."""

from marker_ocr.metadata import MetadataManager, _file_checksum


class TestFileChecksum:
    def test_checksum_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        assert _file_checksum(f) == _file_checksum(f)

    def test_checksum_changes_with_content(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("version 1")
        cs1 = _file_checksum(f)
        f.write_text("version 2")
        cs2 = _file_checksum(f)
        assert cs1 != cs2

    def test_checksum_format(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("data")
        cs = _file_checksum(f)
        assert cs.startswith("sha256:")
        assert len(cs) == len("sha256:") + 64


class TestMetadataManager:
    def test_init_creates_empty_data(self, tmp_path):
        meta = MetadataManager(tmp_path)
        assert meta.files == {}

    def test_record_and_is_processed(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"fake pdf content")

        meta = MetadataManager(tmp_path)
        meta.record(f, processing_time=1.5, output_path="test/test.md")

        assert meta.is_processed(f)

    def test_is_processed_false_for_unknown(self, tmp_path):
        f = tmp_path / "unknown.pdf"
        f.write_bytes(b"data")

        meta = MetadataManager(tmp_path)
        assert not meta.is_processed(f)

    def test_is_processed_false_after_content_change(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"original content")

        meta = MetadataManager(tmp_path)
        meta.record(f, processing_time=1.0, output_path="test/test.md")
        assert meta.is_processed(f)

        f.write_bytes(b"modified content")
        assert not meta.is_processed(f)

    def test_save_creates_metadata_file(self, tmp_path):
        meta = MetadataManager(tmp_path)
        meta.save()
        assert (tmp_path / "metadata.json").exists()

    def test_save_atomic_write(self, tmp_path):
        meta = MetadataManager(tmp_path)
        f = tmp_path / "test.pdf"
        f.write_bytes(b"data")
        meta.record(f, processing_time=1.0, output_path="test/test.md")

        assert not (tmp_path / "metadata.tmp").exists()

    def test_load_persists_across_instances(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"data")

        meta1 = MetadataManager(tmp_path)
        meta1.record(f, processing_time=1.0, output_path="test/test.md")

        meta2 = MetadataManager(tmp_path)
        assert meta2.is_processed(f)

    def test_load_handles_corrupt_json(self, tmp_path):
        (tmp_path / "metadata.json").write_text("not valid json")
        meta = MetadataManager(tmp_path)
        assert meta.files == {}

    def test_record_metadata_content(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"data")

        meta = MetadataManager(tmp_path)
        meta.record(f, processing_time=2.5, output_path="test/test.md", pages=10)

        entry = meta.files["test.pdf"]
        assert entry["status"] == "completed"
        assert entry["processing_time"] == 2.5
        assert entry["model"] == "marker"
        assert entry["output_path"] == "test/test.md"
        assert entry["pages"] == 10
        assert "checksum" in entry
        assert "timestamp" in entry
