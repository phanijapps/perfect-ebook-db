"""Simple Streamlit app for PDF indexing and querying with FAISS."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from book_builder.faiss_ragbuilder import FaissRagBuilder


_DATA_DIR = Path("_data")
_INDEX_DIR = _DATA_DIR / "faiss_index"
_PAGES_DIR = _DATA_DIR / "pages"
_DB_PATH = _DATA_DIR / "pages.sqlite"


def get_builder() -> FaissRagBuilder:
    """Return a ``FaissRagBuilder`` configured to use the ``_data`` directory."""
    _DATA_DIR.mkdir(exist_ok=True)
    return FaissRagBuilder(
        index_dir=_INDEX_DIR,
        output_dir=_PAGES_DIR,
        db_path=_DB_PATH,
    )


st.set_page_config(page_title="PDF Indexer", page_icon="📑")
st.title("PDF Indexer with FAISS")

uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")

title = st.text_input("Title")
author = st.text_input("Author")

col1, col2 = st.columns(2)
with col1:
    start_page = st.number_input("From Page", min_value=1, value=1)
with col2:
    end_page = st.number_input(
        "To Page", min_value=0, value=0, help="0 processes to the end"
    )

process_btn = st.button("Process PDF")

builder: FaissRagBuilder | None = None
if process_btn and uploaded_pdf is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.getbuffer())
        tmp_path = tmp.name
    builder = get_builder()
    builder.process_pdf(
        pdf_path=tmp_path,
        title=title or uploaded_pdf.name,
        author=author or "Unknown",
        start_page=start_page,
        end_page=None if end_page == 0 else end_page,
    )
    st.success("PDF processed and indexed")

st.markdown("---")
query = st.text_input("Query")
search_btn = st.button("Search")

if search_btn and query:
    if builder is None:
        builder = get_builder()
    docs = builder.similarity_search(query)
    for doc in docs:
        st.write(f"Page {doc.metadata['page_number']} - {doc.page_content[:200]}")
