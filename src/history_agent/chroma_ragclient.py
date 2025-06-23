"""
Chroma-based RAG client
"""

from __future__ import annotations
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv


# -----------------------------------------------------------------------------
# Chroma-based RAG builder
# -----------------------------------------------------------------------------
class ChromaRagClient:
    """RAG client to retrienve from Chroma DB"""

    def __init__(
        self,
        *,
        embedding_model: str = "mxbai-embed-large:latest",
        index_dir: str = "chroma_index",
        output_dir: str = "pages",
    ) -> None:
        load_dotenv()  # type: ignore
        self.embedding_model = embedding_model
        self.index_dir = Path(index_dir)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # initialize Ollama embeddings
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)

        # Initialize or load Chroma vector store
        self.vectordb = Chroma(
            collection_name="History",
            persist_directory=str(self.index_dir),
            embedding_function=self.embeddings,
        )

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Run a semantic search against the persisted Chroma store."""
        return self.vectordb.similarity_search(query, k=k)
