import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
pytest.importorskip("faiss")

from streamlit_faiss_app import get_builder


class DummyPage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


def test_builder_creates_files(monkeypatch, tmp_path):
    monkeypatch.setattr("streamlit_faiss_app._DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("streamlit_faiss_app._INDEX_DIR", tmp_path / "data" / "faiss_index")
    monkeypatch.setattr("streamlit_faiss_app._PAGES_DIR", tmp_path / "data" / "pages")
    monkeypatch.setattr("streamlit_faiss_app._DB_PATH", tmp_path / "data" / "pages.sqlite")

    builder = get_builder()

    monkeypatch.setattr(
        "book_builder.faiss_ragbuilder.PdfReader",
        lambda _: type("R", (), {"pages": [DummyPage("hello world")]}),
    )
    monkeypatch.setattr(
        builder,
        "clean_ocr",
        lambda text: builder.clean_ocr.__annotations__["return"](
            cleaned_text=text,
            type_of_text="Generic",
            is_clean=True,
        ),
    )

    builder.process_pdf("dummy.pdf", title="T", author="A")

    assert builder.db_path.exists()
    assert builder.index_dir.exists()
    assert any(builder.output_dir.iterdir())
