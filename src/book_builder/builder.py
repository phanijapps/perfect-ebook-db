"""Example usage of the metadata functionality.

This module demonstrates how to use the PageMetadata class
to generate markdown with rich metadata without modifying faiss_ragbuilder.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Dict, Any

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_community.vectorstores import FAISS
except Exception:  # pragma: no cover - faiss optional
    FAISS = None  # type: ignore

from .faiss_ragbuilder import FaissRagBuilder
from .metadata import create_metadata, PageMetadata


class RagBuilder:
    """Wrapper around FaissRagBuilder that uses extended PageMetadata."""
    
    def __init__(
        self,
        *,
        cleanup_model: str = "mistral-small:latest",
        embedding_model: str = "nomic-embed-text:latest",
        index_dir: str = "faiss_index",
        output_dir: str = "pages",
        db_path: str = "pages.sqlite",
    ):
        """Initialize with the same parameters as FaissRagBuilder."""
        self.base_builder = FaissRagBuilder(
            cleanup_model=cleanup_model,
            embedding_model=embedding_model,
            index_dir=index_dir,
            output_dir=output_dir,
            db_path=db_path,
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    
    def process_pdf(
        self,
        pdf_path: str,
        *,
        title: str,
        author: str,
        start_page: int = 1,
        end_page: Optional[int] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Process a PDF file with extended metadata.
        
        This method extracts text from a PDF, cleans it using the LLM,
        and stores it as markdown with extended metadata.
        
        Args:
            pdf_path: Path to the PDF file
            title: Document title
            author: Document author
            start_page: First page to process (1-indexed)
            end_page: Last page to process (None for all pages)
            additional_metadata: Additional metadata to include
            
        Returns:
            List[str]: Paths to the generated markdown files
        """
        additional_metadata = additional_metadata or {}
        
        # Extract basic information about the PDF
        reader = PdfReader(pdf_path)
        pages = reader.pages
        end_page = end_page or len(pages)
        
        # Add PDF metadata to additional_metadata
        if reader.metadata:
            for key, value in reader.metadata.items():
                if key not in additional_metadata and value:
                    # Clean up the key name
                    clean_key = key.replace('/', '_').strip()
                    if isinstance(value, str):
                        additional_metadata[clean_key] = value
        
        # Add file information
        file_info = os.stat(pdf_path)
        additional_metadata['file_size'] = f"{file_info.st_size / (1024*1024):.2f} MB"
        additional_metadata['total_pages'] = len(pages)
        
        docs: List[Document] = []
        md_paths: List[str] = []
        
        # Process each page
        for idx in range(start_page - 1, end_page):
            raw_text = pages[idx].extract_text() or ""
            result = self.base_builder.clean_ocr(raw_text)
            
            if not result.is_clean:
                continue
                
            # Create metadata
            page_metadata = create_metadata(
                title=title,
                author=author,
                page_number=idx + 1,
                **additional_metadata
            )
            
            # Generate markdown with metadata
            md_content = page_metadata.to_markdown(result.cleaned_text)
            md_path = self.output_dir / f"page_{idx + 1}.md"
            md_path.write_text(md_content, encoding="utf-8")
            
            # Create chunks for indexing
            for chunk in self.text_splitter.split_text(result.cleaned_text):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "title": title,
                        "author": author,
                        "page_number": idx + 1,
                        "md_path": str(md_path),
                        **additional_metadata
                    }
                )
                docs.append(doc)
            
            md_paths.append(str(md_path))
        
        # Index the documents using the base builder's functionality
        if docs and FAISS:
            self.base_builder._index = FAISS.from_documents(
                docs, self.base_builder.embeddings
            )
            self.base_builder._index.save_local(str(self.base_builder.index_dir))
            
            # Store in database
            cur = self.base_builder._conn.cursor()
            for doc in docs:
                cur.execute(
                    "INSERT OR REPLACE INTO pages (id, page_number, title, author, md_path) VALUES (?, ?, ?, ?, ?)",
                    (
                        doc.metadata.get("id", ""),
                        doc.metadata["page_number"],
                        doc.metadata["title"],
                        doc.metadata["author"],
                        doc.metadata["md_path"],
                    ),
                )
            self.base_builder._conn.commit()
        
        return md_paths
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search using the base builder."""
        return self.base_builder.similarity_search(query, k=k)


# Example usage
if __name__ == "__main__":
    builder = RagBuilder(
        index_dir="_data/faiss_index",
        output_dir="_data/pages",
        db_path="_data/pages.sqlite",
    )
    
    # Process a PDF with metadata
    md_paths = builder.process_pdf(
        pdf_path="example.pdf",
        title="Example Document",
        author="John Doe",
        additional_metadata={
            "category": "Technical",
            "keywords": "example, documentation, test",
            "language": "English",
            "year": "2023"
        }
    )
    
    print(f"Generated {len(md_paths)} markdown files with metadata")
