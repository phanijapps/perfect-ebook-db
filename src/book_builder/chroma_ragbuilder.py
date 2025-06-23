"""
Chroma-based RAG builder with markdown output.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from pydantic import Field, SecretStr, BaseModel
from langchain.schema import HumanMessage, SystemMessage, Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.utils.utils import secret_from_env
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from pypdf import PdfReader
from dotenv import load_dotenv
import re

# -----------------------------------------------------------------------------
# Data model for OCR-cleaned text
# -----------------------------------------------------------------------------
class DocStruct(BaseModel):
    """Schema for cleaned OCR text."""
    cleaned_text: str
    type_of_text: str = "Generic"
    is_clean: bool = True

# -----------------------------------------------------------------------------
# System prompt for cleaning OCR with an LLM
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a helpful assistant that processes scanned book pages. It is important to keep the main text untouched.
Clean the text by:
1. Removing page numbers, headers and footers.
2. Removing table of contents, forewords and other front matter.
3. Removing gibberish, escape sequences, non utf-8 chars or OCR artifacts.
4. Fixing hyphenated line breaks when possible.
5. Remove escape sequence characters.
5. Keep main content untouched.
If the text doesn't make sense for a chapter book return is_clean as false.
Return a JSON object with cleaned_text, type_of_text and is_clean fields.
"""

# -----------------------------------------------------------------------------
# Simple metadata container
# -----------------------------------------------------------------------------
@dataclass
class PageMetadata:
    title: str
    author: str
    page_number: int
    book_acronym: str

    def to_markdown(self, content: str) -> str:
        return (
            f"{content}"
        )

# -----------------------------------------------------------------------------
# Router for OpenRouter / Gemini
# -----------------------------------------------------------------------------
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
        openai_api_key = openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=openai_api_key,
            **kwargs
        )

# -----------------------------------------------------------------------------
# Chroma-based RAG builder
# -----------------------------------------------------------------------------
class ChromaRagBuilder:
    """RAG builder that stores cleaned pages as markdown and indexes with Chroma."""

    def __init__(
        self,
        *,
        cleanup_model: str = "mistral-small:latest",
        embedding_model: str = "mxbai-embed-large:latest",
        index_dir: str = "chroma_index",
        output_dir: str = "pages",
    ) -> None:
        load_dotenv()  # type: ignore
        self.cleanup_model = cleanup_model
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

        # LLM for OCR cleanup
        #self.llm = ChatOpenRouter(
        #    model="google/gemini-flash-1.5-8b",
        #).with_structured_output(schema=DocStruct, method="json_mode")
        self.llm = ChatOllama(
            model="llama3.2:3b"
        ).with_structured_output(schema=DocStruct, method="json_mode")

    
    def clean_to_utf8(self, text: str) -> str:
        """
        Removes all Unicode characters that can't be encoded in UTF-8,
        keeping only valid UTF-8 encodable characters.
        """
        if not isinstance(text, str):
            raise TypeError("Expected input to be of type 'str'.")

        # Remove characters outside valid UTF-8 range (keep ASCII and valid UTF-8)
        text = ''.join(char for char in text if ord(char) < 0x110000)

        # Remove surrogate range (U+D800–U+DFFF)
        text = re.sub(r'[\uD800-\uDFFF]', '', text)

        # Final encoding/decoding to ensure UTF-8 compatibility
        text = text.encode('utf-8', errors='ignore').decode('utf-8')

        return text


    def clean_ocr(self, raw_text: str) -> DocStruct:
        """Clean OCR text using LLM with backslash handling."""
        print("Cleaning ocr")
        clean_raw_text = self.clean_to_utf8(raw_text)

        msgs = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=clean_raw_text)]
        try:
            cleaned_doc = self.llm.invoke(msgs)
        except:
            cleaned_doc = DocStruct(cleaned_text=clean_raw_text, is_clean=True, type_of_text="Generic")
        print(f"Is Clean {cleaned_doc.is_clean}")
        return cleaned_doc
      

    def _store_page(self, meta: PageMetadata, cleaned_text: str) -> Document:

        print(f"Page Number {meta.page_number}")
     
        doc = Document(page_content=cleaned_text, metadata={
            "id": f"{meta.book_acronym}_{meta.page_number}",
            "title": meta.title,
            "author": meta.author,
            "page_number": meta.page_number,
            "book_acronym": meta.book_acronym
            
        })
        return doc

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
    ) -> None:
        reader = PdfReader(pdf_path)
        pages = reader.pages
        end_page = end_page or len(pages)
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        
        batch: List[Document] = []

        for idx in range(start_page - 1, end_page):
            raw = pages[idx].extract_text() or ""
            if skip_cleanup:
                cleaned, is_clean = self.clean_to_utf8(raw), True
            else:
                struct = self.clean_ocr(raw)
                cleaned, is_clean = struct.cleaned_text, struct.is_clean
                if not is_clean:
                    continue

            meta = PageMetadata(title=title, author=author, page_number=idx+1, book_acronym=book_acronym)
            doc = self._store_page(meta, cleaned)
            
            for chunk in splitter.split_text(cleaned):
                doc_chunk = Document(page_content=chunk, metadata=doc.metadata)
                batch.append(doc_chunk)

            # index in batches of 20 pages
            if len(batch) >= 20:
                self._process_batch(batch)
                batch = []

        # leftover
        if batch:
            self._process_batch(batch)

        return

    def _process_batch(self, docs: List[Document]) -> None:
        """Embed and store a batch of Document chunks into Chroma."""
        if not docs:
            return
        # ingest and index
        self.vectordb.add_documents(docs)
  

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Run a semantic search against the persisted Chroma store."""
        return self.vectordb.similarity_search(query, k=k)
