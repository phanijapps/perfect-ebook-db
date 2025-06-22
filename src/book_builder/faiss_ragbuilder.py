"""Faiss-based RAG builder with markdown output.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):
        return None

from langchain.schema import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
try:
    from langchain_community.vectorstores import FAISS
except Exception:  # pragma: no cover - faiss optional
    FAISS = None  # type: ignore

from pypdf import PdfReader
from pydantic import BaseModel


class DocStruct(BaseModel):
    """Schema for cleaned OCR text."""

    cleaned_text: str
    type_of_text: str = "Generic"
    is_clean: bool = True


SYSTEM_PROMPT = """
You are a helpful assistant that processes OCR-scanned book pages.
Clean the text by:
1. Removing page numbers, headers and footers.
2. Removing table of contents, forewords and other front matter.
3. Removing gibberish or OCR artifacts.
4. Fixing hyphenated line breaks when possible.
5. Keep main content untouched.
Return a JSON object with cleaned_text, type_of_text and is_clean fields.
"""


@dataclass
class PageMetadata:
    title: str
    author: str
    page_number: int

    def to_markdown(self, content: str) -> str:
        md = (
            "## Metadata\n"
            "|Title| Page #  | Author |\n"
            "|--|--|--|\n"
            f"| {self.title} | {self.page_number}  | {self.author}|\n\n"
            "## Content\n" + content
        )
        return md


class FaissRagBuilder:
    """RAG builder that stores cleaned pages as markdown and indexes with FAISS."""

    def __init__(
        self,
        *,
        cleanup_model: str = "mistral-small:latest",
        embedding_model: str = "nomic-embed-text:latest",
        index_dir: str = "faiss_index",
        output_dir: str = "pages",
        db_path: str = "pages.sqlite",
    ) -> None:
        load_dotenv()
        self.cleanup_model = cleanup_model
        self.embedding_model = embedding_model
        self.index_dir = Path(index_dir)
        self.output_dir = Path(output_dir)
        self.db_path = Path(db_path)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)
        self.llm = ChatOllama(model=self.cleanup_model, temperature=0).with_structured_output(
            schema=DocStruct, method="json_mode"
        )

        self._conn = sqlite3.connect(self.db_path)
        self._init_db()
        self._index: Optional[FAISS] = None

    def _init_db(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS pages (
            id TEXT PRIMARY KEY,
            page_number INTEGER,
            title TEXT,
            author TEXT,
            md_path TEXT
        )"""
        )
        self._conn.commit()

    # ---------------- processing helpers -----------------
    def clean_ocr(self, raw_text: str) -> DocStruct:
        msgs = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=raw_text)]
        return self.llm.invoke(msgs)

    def _store_page(self, meta: PageMetadata, cleaned_text: str) -> tuple[Document, str]:
        md_content = meta.to_markdown(cleaned_text)
        md_path = self.output_dir / f"page_{meta.page_number}.md"
        md_path.write_text(md_content, encoding="utf-8")
        return Document(page_content=cleaned_text, metadata={
            "title": meta.title,
            "author": meta.author,
            "page_number": meta.page_number,
            "md_path": str(md_path),
        }), str(md_path)

    def process_pdf(
        self,
        pdf_path: str,
        *,
        title: str,
        author: str,
        start_page: int = 1,
        end_page: Optional[int] = None,
    ) -> List[str]:
        reader = PdfReader(pdf_path)
        pages = reader.pages
        end_page = end_page or len(pages)
        docs: List[Document] = []
        md_paths: List[str] = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        for idx in range(start_page - 1, end_page):
            raw_text = pages[idx].extract_text() or ""
            result = self.clean_ocr(raw_text)
            if not result.is_clean:
                continue
            meta = PageMetadata(title=title, author=author, page_number=idx + 1)
            doc, md_path = self._store_page(meta, result.cleaned_text)
            for chunk in splitter.split_text(result.cleaned_text):
                docs.append(Document(page_content=chunk, metadata=doc.metadata))
            md_paths.append(md_path)
        if not FAISS:
            raise ImportError("faiss library is required for indexing")
        if docs:
            self._index = FAISS.from_documents(docs, self.embeddings)
            self._index.save_local(str(self.index_dir))
            # store ids
            cur = self._conn.cursor()
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
            self._conn.commit()
        return md_paths

    def load_index(self) -> None:
        if not FAISS:
            raise ImportError("faiss library is required for indexing")
        if self._index is None:
            self._index = FAISS.load_local(str(self.index_dir), self.embeddings)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if not FAISS:
            raise ImportError("faiss library is required for search")
        if self._index is None:
            self.load_index()
        return self._index.similarity_search(query, k=k)
