import os
import re
from langchain.schema import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_core.documents import Document
from dotenv import load_dotenv
from pypdf import PdfReader
from pydantic import BaseModel


class DocStruct(BaseModel):
    cleaned_text: str
    type_of_text: str
    is_clean: bool


# Load .env values
load_dotenv()

DB_CONN_STRING = f"postgresql+psycopg://{os.getenv('POSTGRES_USERNAME')}:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/vec_db"
EMBEDDING_MODEL="nomic-embed-text:latest"
LLM_MODEL="mistral-small:latest"

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Ensure documents table exists
def init_pgvector_table(collection_name: str = "documents", vector_size: int = 768):
    """
        Initialize PGVector Store
    """

    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=DB_CONN_STRING,
        use_jsonb=True,
    )


# System prompt for OCR cleanup
SYSTEM_PROMPT = """
You are a helpful assistant that processes OCR-scanned book pages. 
Clean the text by:
1. Removing page numbers, headers, footers, and repeated titles.
2. Removing table of contents, forewords, prefaces, and similar front matter.
3. Removing gibberish, OCR artifacts, or broken words (e.g., Th1s, f1gur3, Page | 103).
4. Fixing split words across lines if possible (e.g., infor-\nmation → information).
5. Leave the main content untouched — do not summarize or rewrite.
Do NOT change sentence structure or fix grammar unless it's OCR-related.

Return your response in the following JSON format:
{
  "cleaned_text": "The cleaned OCR text with all fixes applied",
  "type_of_text": "The type of text (e.g., academic, narrative, technical, instructional, etc.)",
  "is_clean": true or false (whether the text is clean and contains meaningful content, set to false if it's mostly gibberish, front matter, or not worth processing)
}
"""



# LLM setup
llm = ChatOllama(model=LLM_MODEL, temperature=0).with_structured_output(schema=DocStruct,method="json_mode")

import json

def clean_ocr(raw_text: str) -> DocStruct:
    """
    Clean OCR text and return structured JSON.
    
    Args:
        raw_text: The raw OCR text to clean
        
    Returns:
        DocStruct: A dictionary with the following structure:
        {
            "cleaned_text": str,
            "type_of_text": str,
            "is_clean": bool
        }
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=raw_text)]
    response = llm.invoke(messages)
    print(response)
    return response    

# Keep the original function for backward compatibility
def clean_ocr_text(raw_text: str) -> str:
    result = clean_ocr(raw_text)
    return result["cleaned_text"]

def chunk_text(cleaned_text: str, chunk_size: int = 2000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(cleaned_text)

def store_chunks(chunks, page_number: int = 0, store: PGVector = None):
    """
        Store Chunks in Postgres Vector DB
    """
    docs = [
        Document(
            page_content=chunk,
            metadata={"source": "input.pdf", "page_number": page_number}
        ) for chunk in chunks
    ]

    store.add_documents(documents=docs)


def should_skip_page(raw_text: str) -> tuple:
    """
    Determine if a page should be skipped based on both heuristics and LLM analysis.
    
    Args:
        raw_text: The raw OCR text to analyze
        
    Returns:
        tuple: (should_skip, ocr_result)
            - should_skip: Boolean indicating if the page should be skipped
            - ocr_result: The result from clean_ocr if the page is not skipped based on heuristics,
                         None otherwise
    """
    # First check using heuristics
    if not raw_text.strip():
        return True, None
        
    # If it passes heuristics, use LLM to clean and analyze
    ocr_result = clean_ocr(raw_text)
    
    # If LLM determines the content is not clean/useful, skip it
    if not ocr_result.is_clean:
        return True, ocr_result
        
    return False, ocr_result


def process_pdf(pdf_path: str, vec_store):
    reader = PdfReader(pdf_path)
    for i, page in enumerate(reader.pages):
        page_number = i + 1
        if page_number == 1000:
            break
        
        raw_text = page.extract_text() or ""
        
        print(f"\n📄 Analyzing page {page_number}")
        should_skip, ocr_result = should_skip_page(raw_text)
        
        if should_skip:
            if not raw_text.strip():
                print(f"📛 Page {page_number} is empty — skipped.")
            elif ocr_result is None:
                print(f"⚠️ Page {page_number} seems like gibberish based on heuristics — skipped.")
            else:
                print(f"⚠️ Page {page_number} was determined not worth processing by LLM — skipped.")
            continue
            
        # Process the page since it's worth keeping
        cleaned = ocr_result.cleaned_text
        print(f"✅ Cleaned text (page {page_number}):")
        print(cleaned[:300] + "..." if len(cleaned) > 300 else cleaned)
        print(f"📊 Text type: {ocr_result.type_of_text}")

        chunks = chunk_text(cleaned)
        print(f"📚 {len(chunks)} chunks created.")
        store_chunks(chunks, page_number=page_number, store=vec_store)
        print(f"💾 Stored chunks from page {page_number}.")

if __name__ == "__main__":
    vec_store = init_pgvector_table(collection_name="indian_history")
    process_pdf("/home/videogamer/Downloads/11vols_indian_history.pdf",vec_store=vec_store)
