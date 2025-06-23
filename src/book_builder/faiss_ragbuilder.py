"""Faiss-based RAG builder with markdown output.
"""

from __future__ import annotations
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from pydantic import Field, SecretStr
import faiss, pickle

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):
        return None

from langchain.schema import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.utils.utils import secret_from_env
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
try:
    from langchain_community.vectorstores import FAISS as LCFAISS
except Exception:  # pragma: no cover - faiss optional
    LCFAISS = None  # type: ignore

from pypdf import PdfReader
from pydantic import BaseModel


class DocStruct(BaseModel):
    """Schema for cleaned OCR text."""

    cleaned_text: str
    type_of_text: str = "Generic"
    is_clean: bool = True

faiss.omp_set_num_threads(8)

SYSTEM_PROMPT = """
You are a helpful assistant that processes scanned book pages.
Clean the text by:
If the text doesnt make sense for a chapter book return is_clean as false.
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
    book_acronym: str

    def to_markdown(self, content: str) -> str:
        md = (
            "## Metadata\n"
            "|Title| Page #  | Author | Book ID |\n"
            "|--|--|--|--|\n"
            f"| {self.title} | {self.page_number}  | {self.author}| {self.book_acronym} |\n\n"
            "## Content\n" + content
        )
        return md

class ChatOpenRouter(ChatOpenAI):
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("OPENROUTER_API_KEY", default=None),
    )
    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 **kwargs):
        openai_api_key = (
            openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        )
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=openai_api_key,
            **kwargs
        )


class FaissRagBuilder:
    """RAG builder that stores cleaned pages as markdown and indexes with LCFAISS."""

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
        
        # Use ChatOpenRouter with Gemini as requested
        self.llm = ChatOpenRouter(
            model="google/gemini-flash-1.5-8b",
        ).with_structured_output(
           schema=DocStruct, method="json_mode"
        )

        self._conn = sqlite3.connect(self.db_path)
        self._init_db()
        self._index: Optional[LCFAISS] = None

    def _init_db(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS pages (
            id TEXT PRIMARY KEY,
            page_number INTEGER,
            title TEXT,
            author TEXT,
            md_path TEXT,
            book_acronym TEXT
        )"""
        )
        self._conn.commit()

    # ---------------- processing helpers -----------------
    def clean_ocr(self, raw_text: str) -> DocStruct:
        """Clean OCR text using LLM with handling for backslash characters."""
        import json
        import re
        
        msgs = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=raw_text)]
        
        try:
            # Get raw response without structured output
            raw_response = self.llm.without_structured_output().invoke(msgs).content
            
            # Extract JSON part if it's in markdown format
            json_match = re.search(r'```json\s*(.*?)\s*```', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If not in markdown code block, try to find JSON object
                json_match = re.search(r'(\{.*\})', raw_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = raw_response
            
            # Remove problematic single backslashes before parsing
            # Replace single backslashes that aren't part of valid escape sequences
            fixed_json = re.sub(r'\\([^"\\/bfnrtu])', r'\1', json_str)
            
            # Parse the fixed JSON
            parsed_json = json.loads(fixed_json)
            
            # Create and return DocStruct
            return DocStruct(
                cleaned_text=parsed_json.get("cleaned_text", raw_text),
                type_of_text=parsed_json.get("type_of_text", "Generic"),
                is_clean=parsed_json.get("is_clean", True)
            )
        except Exception as e:
            # If any error occurs, return the raw text as is
            print(f"Error processing text with LLM: {str(e)}")
            return DocStruct(
                cleaned_text=raw_text,
                type_of_text="Raw OCR",
                is_clean=True
            )

    def _store_page(self, meta: PageMetadata, cleaned_text: str) -> tuple[Document, str]:
        md_content = meta.to_markdown(cleaned_text)
        md_path = self.output_dir / f"{meta.book_acronym}_{meta.page_number}.md"
        md_path.write_text(md_content, encoding="utf-8")
        return Document(page_content=cleaned_text, metadata={
            "title": meta.title,
            "author": meta.author,
            "page_number": meta.page_number,
            "book_acronym": meta.book_acronym,
            "md_path": str(md_path),
        }), str(md_path)

    def process_pdf(
        self,
        pdf_path: str,
        *,
        title: str,
        author: str,
        book_acronym: str,
        start_page: int = 1,
        end_page: Optional[int] = None,
        skip_cleanup: bool = False,
    ) -> List[str]:
        reader = PdfReader(pdf_path)
        pages = reader.pages
        end_page = end_page or len(pages)
        docs: List[Document] = []
        md_paths: List[str] = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        
        # Create index after every 20 documents
        batch_size = 20
        batch_count = 0
        
        for idx in range(start_page - 1, end_page):
            raw_text = pages[idx].extract_text() or ""
            
            # Skip cleanup if requested
            if skip_cleanup:
                cleaned_text = raw_text
                is_clean = True
            else:
                result = self.clean_ocr(raw_text)
                cleaned_text = result.cleaned_text
                is_clean = result.is_clean
                
                # Only check is_clean when we're actually doing cleanup
                if not is_clean:
                    continue
                
            meta = PageMetadata(title=title, author=author, page_number=idx + 1, book_acronym=book_acronym)
            doc, md_path = self._store_page(meta, cleaned_text)
            
            for chunk in splitter.split_text(cleaned_text):
                docs.append(Document(page_content=chunk, metadata=doc.metadata))
            
            md_paths.append(md_path)
            batch_count += 1
            
            # Process batch if we've reached batch_size
            if batch_count >= batch_size:
                self._process_batch(docs)
                batch_count = 0
        
        # Process any remaining documents
        if docs and batch_count > 0:
            self._process_batch(docs)
            
        return md_paths
        
    def _process_batch(self, docs: List[Document]) -> None:
        """Process a batch of documents by creating embeddings and storing in the database."""
        if not LCFAISS:
            raise ImportError("faiss library is required for indexing")
            
        if not docs:
            return
            
        # Create or update the index
        if self._index is None:
            self._index = LCFAISS.from_documents(docs, self.embeddings)
        else:
            self._index.add_documents(docs)
            
        # Save the index
        self._index.save_local(str(self.index_dir))
        
        # Store in database
        cur = self._conn.cursor()
        for doc in docs:
            cur.execute(
                "INSERT OR REPLACE INTO pages (id, page_number, title, author, md_path, book_acronym) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    doc.metadata.get("id", ""),
                    doc.metadata["page_number"],
                    doc.metadata["title"],
                    doc.metadata["author"],
                    doc.metadata["md_path"],
                    doc.metadata["book_acronym"],
                ),
            )
        self._conn.commit()

    def load_index(self) -> None:
        if not LCFAISS:
            raise ImportError("faiss library is required for indexing")

        if self._index is not None:
            return
        if self._index is None:
            print("Loading index")
            self._index = LCFAISS.load_local(str(self.index_dir), self.embeddings,allow_dangerous_deserialization=True)
            print("Index Loaded.")
        


    def load_index_new(self) -> None:
        if self._index is not None:
            return

        # 1) mmap the binary

        idx = faiss.read_index(
            str(self.index_dir / "index.faiss"),
            faiss.IO_FLAG_MMAP | faiss.IO_FLAG_ONDISK_SAME_DIR
        )
        print("Faiss loaded")

        # 2) load only the docstore pickle
        with open(self.index_dir / "index.pkl", "rb") as f:
            docs, id_map = pickle.load(f)
        
        print ("pickle Loaded")

        # 3) re-wrap in LangChain
        self._index = LCFAISS(
            embedding_function=self.embeddings.embed_query,
            index=idx,
            docstore=docs,
            index_to_docstore_id=id_map
        )
        print("✅ Memory-mapped index ready in <100 ms")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if not LCFAISS:
            raise ImportError("faiss library is required for search")
        if self._index is None:
            self.load_index_new()
        print(f"Searching... {query}")
        return self._index.similarity_search(query, k=k)
