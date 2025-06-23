"""Streamlit app for PDF indexing and querying with FAISS."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from book_builder.chroma_ragbuilder import ChromaRagBuilder


_DATA_DIR = Path("../_data")
_INDEX_DIR = _DATA_DIR / "chroma_index"
_PAGES_DIR = _DATA_DIR / "pages"
_DB_PATH = _DATA_DIR / "pages.sqlite"


def get_builder() -> ChromaRagBuilder:
    """Return a ``FaissRagBuilder`` configured to use the ``_data`` directory."""
    _DATA_DIR.mkdir(exist_ok=True)
    return ChromaRagBuilder(
        index_dir=str(_INDEX_DIR),
        output_dir=str(_PAGES_DIR),
    )


st.set_page_config(page_title="PDF Indexer", page_icon="📑")
st.title("PDF Indexer with FAISS")

# Create tabs for upload and query
tab1, tab2 = st.tabs(["Upload & Process", "Query"])

# Initialize session state for builder if it doesn't exist
if "builder" not in st.session_state:
    st.session_state.builder = None

# Tab 1: Upload and Process
with tab1:
    st.header("Upload and Process PDF")
    
    uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")

    title = st.text_input("Title")
    author = st.text_input("Author")
    book_acronym = st.text_input("Book Acronym (used for file naming)", placeholder="e.g., LOTR")

    col1, col2 = st.columns(2)
    with col1:
        start_page = st.number_input("From Page", min_value=1, value=1)
    with col2:
        end_page = st.number_input(
            "To Page", min_value=0, value=0, help="0 processes to the end"
        )
    
    # Option to skip cleanup
    skip_cleanup = st.checkbox("Skip cleanup (faster but may include OCR artifacts)")
    
    process_btn = st.button("Process PDF")

    if process_btn and uploaded_pdf is not None:
        if not book_acronym:
            st.error("Please provide a Book Acronym for file naming.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.getbuffer())
                tmp_path = tmp.name
            
            # Get builder and process PDF
            builder = get_builder()
            
            with st.spinner("Processing PDF... This may take a while."):
                builder.process_pdf(
                    pdf_path=tmp_path,
                    title=title or uploaded_pdf.name,
                    author=author or "Unknown",
                    book_acronym=book_acronym,
                    start_page=start_page,
                    end_page=None if end_page == 0 else end_page,
                    skip_cleanup=skip_cleanup,
                )
            
            # Store builder in session state for use in query tab
            st.session_state.builder = builder
            
            st.success(f"PDF processed and indexed. Generated files.")

# Tab 2: Query
with tab2:
    st.header("Query Indexed Documents")
    
    query = st.text_input("Enter your query")
    k = st.slider("Number of results", min_value=1, max_value=10, value=4)
    search_btn = st.button("Search")

    if search_btn and query:
        # Use builder from session state or create a new one
        builder = st.session_state.builder or get_builder()
        
        docs = builder.similarity_search(query, k=k)
        
        if not docs:
            st.warning("No results found. Please try a different query or process a PDF first.")
        else:
            st.subheader(f"Found {len(docs)} results:")
            
            for i, doc in enumerate(docs):
                with st.expander(f"Result {i+1} - Page {doc.metadata['page_number']} from {doc.metadata['title']}"):
                    st.markdown(f"**Title:** {doc.metadata['title']}")
                    st.markdown(f"**Author:** {doc.metadata['author']}")
                    st.markdown(f"**Page:** {doc.metadata['page_number']}")
                    
                    # Display book acronym if available
                    if 'book_acronym' in doc.metadata:
                        st.markdown(f"**Book ID:** {doc.metadata['book_acronym']}")
                    
                    st.markdown("**Content:**")
                    st.markdown(doc.page_content)
