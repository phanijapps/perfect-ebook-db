#!/usr/bin/env python3
"""
Streamlit app for the Advanced RAG Builder
"""

import streamlit as st
from Ragbuilder_advanced import StreamlitRagBuilder

# Set page configuration
st.set_page_config(
    page_title="Advanced RAG Builder",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the Streamlit RAG Builder
rag_builder_ui = StreamlitRagBuilder(st)

# Create the UI
rag_builder_ui.create_ui()

# Add some additional information at the bottom
st.markdown("---")
st.markdown("""
### About Advanced RAG Builder

This application uses LangChain and Ollama to:
- Clean PDF book pages using a local LLM
- Chunk cleaned text for downstream RAG pipelines
- Store embeddings in a PostgreSQL vector database

#### Features:
- Batch processing of pages
- Configurable cleanup and embedding models
- Detailed progress tracking with rich emoji feedback
- Customizable chunking parameters
""")

# Add a sidebar note
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    **Note:** Make sure you have:
    - PostgreSQL running with pgvector extension
    - Ollama installed with the selected models
    - Proper environment variables set in .env file
    """)
