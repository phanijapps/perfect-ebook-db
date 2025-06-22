import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
pytest.importorskip("langchain")
from faiss_ragbuilder import FaissRagBuilder, PageMetadata


def test_markdown_generation(tmp_path):
    meta = PageMetadata(title="Test", author="Author", page_number=1)
    builder = FaissRagBuilder(index_dir=tmp_path / "idx", output_dir=tmp_path / "out", db_path=tmp_path / "db.sqlite")
    md_content = meta.to_markdown("content")
    md_path = builder.output_dir / "page_1.md"
    md_path.write_text(md_content)
    assert md_path.exists()


def test_process_pdf_skipped_if_no_faiss(monkeypatch, tmp_path):
    pytest.importorskip("faiss")

    class DummyPage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    monkeypatch.setattr(
        "faiss_ragbuilder.PdfReader",
        lambda _: type("R", (), {"pages": [DummyPage("hello world")]}),
    )
    builder = FaissRagBuilder(index_dir=tmp_path / "idx", output_dir=tmp_path / "out", db_path=tmp_path / "db.sqlite")
    monkeypatch.setattr(
        builder,
        "clean_ocr",
        lambda text: builder.clean_ocr.__annotations__["return"](
            cleaned_text=text,
            type_of_text="Generic",
            is_clean=True,
        ),
    )
    paths = builder.process_pdf("dummy.pdf", title="T", author="A")
    assert paths
    builder.load_index()
    docs = builder.similarity_search("hello", k=1)
    assert docs
