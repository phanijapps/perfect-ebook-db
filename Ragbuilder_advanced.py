#!/usr/bin/env python3
"""
Ragbuilder_advanced.py - Advanced RAG Builder with OOP approach

Features:
1. Batch processing - control number of pages to read at once
2. Choice of cleanup model and embedding models
3. Start and end pages and PDF file as command line arguments
4. Message streaming with rich emoji feedback
5. Streamlit-friendly design for easy integration
"""

import os
import argparse
import concurrent.futures
import time
from typing import List, Tuple, Optional, Dict, Any, Callable, Iterator, Union

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
    """Structure for cleaned document data"""
    cleaned_text: str
    type_of_text: str = "Generic"
    is_clean: bool = True


# Message streaming callback type
MessageCallback = Callable[[str], None]

class RagBuilder:
    """Advanced RAG Builder with configurable models and batch processing"""
    
    # Emojis for different message types
    EMOJIS = {
        "start": "🚀",
        "processing": "⚙️",
        "analyzing": "🔍",
        "cleaning": "🧹",
        "chunking": "✂️",
        "storing": "💾",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "skip": "⏭️",
        "empty": "📛",
        "gibberish": "🤪",
        "summary": "📊",
        "pdf": "📚",
        "page": "📄",
        "chunk": "📦",
        "model": "🤖",
        "database": "🗄️",
        "complete": "🏁",
        "time": "⏱️"
    }
    
    # System prompt for OCR cleanup
    DEFAULT_SYSTEM_PROMPT = """
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
      "type_of_text": "The type of text (e.g., academic, narrative, technical, instructional, etc.)", set this to generic by default
      "is_clean": true or false (whether the text is clean and contains meaningful content, set to false if it's mostly gibberish, front matter, or not worth processing) and true when you are unsure..
    }
    """

    def __init__(
        self,
        cleanup_model: str = "mistral-small:latest",
        embedding_model: str = "nomic-embed-text:latest",
        collection_name: str = "documents",
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        vector_size: int = 768,
        system_prompt: Optional[str] = None,
        db_conn_string: Optional[str] = None,
        batch_size: int = 1,
        message_callback: Optional[MessageCallback] = None,
        verbose: bool = True
    ):
        """
        Initialize the RAG Builder with configurable parameters
        
        Args:
            cleanup_model: Model name for text cleanup
            embedding_model: Model name for embeddings
            collection_name: Name of the vector collection in the database
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            vector_size: Size of the embedding vectors
            system_prompt: Custom system prompt for cleanup
            db_conn_string: Database connection string (if None, loads from .env)
            batch_size: Number of pages to process in parallel
            message_callback: Optional callback function for streaming messages
            verbose: Whether to print messages to console
        """
        # Load environment variables
        load_dotenv()
        
        # Set instance variables
        self.cleanup_model = cleanup_model
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_size = vector_size
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.batch_size = batch_size
        self.message_callback = message_callback
        self.verbose = verbose
        
        # Set up database connection
        if db_conn_string is None:
            self.db_conn_string = f"postgresql+psycopg://{os.getenv('POSTGRES_USERNAME')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'vec_db')}"
        else:
            self.db_conn_string = db_conn_string
        
        # Stream initialization message
        self.stream_message("start", f"Initializing RAG Builder")
        
        # Initialize models
        self.stream_message("model", f"Loading cleanup model: {self.cleanup_model}")
        self.stream_message("model", f"Loading embedding model: {self.embedding_model}")
        
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)
        self.llm = ChatOllama(model=self.cleanup_model, temperature=0).with_structured_output(schema=DocStruct, method="json_mode")
        
        # Initialize vector store
        self.stream_message("database", f"Connecting to vector database")
        self.vec_store = self._init_pgvector_table()
        
        # Configuration summary
        self.stream_message("success", f"RAG Builder initialized successfully")
        self.stream_message("summary", f"Configuration:")
        self.stream_message("model", f"Cleanup model: {self.cleanup_model}")
        self.stream_message("model", f"Embedding model: {self.embedding_model}")
        self.stream_message("database", f"Collection: {self.collection_name}")
        self.stream_message("processing", f"Batch size: {self.batch_size} pages")
        self.stream_message("chunk", f"Chunk size: {self.chunk_size}, overlap: {self.chunk_overlap}")
        
    def stream_message(self, msg_type: str, message: str) -> None:
        """
        Stream a message with an emoji prefix
        
        Args:
            msg_type: Type of message (used to select emoji)
            message: The message text
        """
        emoji = self.EMOJIS.get(msg_type, "ℹ️")
        formatted_msg = f"{emoji} {message}"
        
        # Send to callback if provided (for Streamlit integration)
        if self.message_callback:
            self.message_callback(formatted_msg)
        
        # Print to console if verbose
        if self.verbose:
            print(formatted_msg)
    
    def stream_progress(self, current: int, total: int, prefix: str = "Progress") -> None:
        """
        Stream a progress bar message
        
        Args:
            current: Current progress value
            total: Total value for 100% progress
            prefix: Prefix text for the progress bar
        """
        percent = int(100 * current / total)
        bar_length = 20
        filled_length = int(bar_length * current / total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        message = f"{prefix}: [{bar}] {percent}% ({current}/{total})"
        self.stream_message("processing", message)
    
    def _init_pgvector_table(self) -> PGVector:
        """Initialize PGVector Store"""
        return PGVector(
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            connection=self.db_conn_string,
            use_jsonb=True,
        )
    
    def clean_ocr(self, raw_text: str) -> DocStruct:
        """
        Clean OCR text and return structured result
        
        Args:
            raw_text: The raw OCR text to clean
            
        Returns:
            DocStruct: A structured result with cleaned text and metadata
        """
        self.stream_message("cleaning", "Cleaning text with LLM")
        start_time = time.time()
        
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=raw_text)]
        response = self.llm.invoke(messages)
        
        elapsed = time.time() - start_time
        self.stream_message("time", f"Cleaning completed in {elapsed:.2f} seconds")
        
        return response
    
    def should_skip_page(self, raw_text: str) -> Tuple[bool, Optional[DocStruct]]:
        """
        Determine if a page should be skipped based on heuristics and LLM analysis
        
        Args:
            raw_text: The raw OCR text to analyze
            
        Returns:
            tuple: (should_skip, ocr_result)
        """
        self.stream_message("analyzing", "Analyzing page content")
        
        # First check using heuristics
        if not raw_text.strip():
            self.stream_message("empty", "Page appears to be empty")
            return True, None
        
        # If it passes heuristics, use LLM to clean and analyze
        ocr_result = self.clean_ocr(raw_text)
        
        # If LLM determines the content is not clean/useful, skip it
        if not ocr_result.is_clean:
            self.stream_message("skip", f"Page content not worth processing: {ocr_result.type_of_text}")
            return True, ocr_result
        
        self.stream_message("success", f"Page content is valuable: {ocr_result.type_of_text}")
        return False, ocr_result
    
    def chunk_text(self, cleaned_text: str) -> List[str]:
        """
        Split text into chunks for embedding
        
        Args:
            cleaned_text: The cleaned text to chunk
            
        Returns:
            List[str]: List of text chunks
        """
        self.stream_message("chunking", f"Splitting text into chunks (size: {self.chunk_size}, overlap: {self.chunk_overlap})")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_text(cleaned_text)
        
        self.stream_message("chunk", f"Created {len(chunks)} chunks")
        return chunks
    
    def store_chunks(self, chunks: List[str], page_number: int = 0, source: str = "input.pdf"):
        """
        Store chunks in the vector database
        
        Args:
            chunks: List of text chunks
            page_number: Page number for metadata
            source: Source document name for metadata
        """
        self.stream_message("storing", f"Storing {len(chunks)} chunks in vector database")
        start_time = time.time()
        
        docs = [
            Document(
                page_content=chunk,
                metadata={"source": source, "page_number": page_number}
            ) for chunk in chunks
        ]
        
        self.vec_store.add_documents(documents=docs)
        
        elapsed = time.time() - start_time
        self.stream_message("success", f"Chunks stored successfully in {elapsed:.2f} seconds")
    
    def process_page(self, page_data: Tuple[int, str, str]) -> Dict[str, Any]:
        """
        Process a single page
        
        Args:
            page_data: Tuple of (page_number, raw_text, source)
            
        Returns:
            Dict: Results of processing the page
        """
        page_number, raw_text, source = page_data
        
        self.stream_message("page", f"Processing page {page_number}")
        start_time = time.time()
        
        should_skip, ocr_result = self.should_skip_page(raw_text)
        
        result = {
            "page_number": page_number,
            "processed": False,
            "skipped": should_skip,
            "reason": None,
            "chunks_created": 0,
            "text_type": None,
            "processing_time": 0
        }
        
        if should_skip:
            if not raw_text.strip():
                self.stream_message("empty", f"Page {page_number} is empty — skipped")
                result["reason"] = "empty"
            elif ocr_result is None:
                self.stream_message("gibberish", f"Page {page_number} seems like gibberish based on heuristics — skipped")
                result["reason"] = "gibberish_heuristic"
            else:
                self.stream_message("warning", f"Page {page_number} was determined not worth processing by LLM — skipped")
                result["reason"] = "not_worth_processing_llm"
                
            elapsed = time.time() - start_time
            result["processing_time"] = elapsed
            self.stream_message("time", f"Page {page_number} processing completed in {elapsed:.2f} seconds")
            return result
            
        # Process the page since it's worth keeping
        cleaned = ocr_result.cleaned_text
        self.stream_message("success", f"Cleaned text (page {page_number}):")
        preview = cleaned[:300] + "..." if len(cleaned) > 300 else cleaned
        self.stream_message("success", preview)
        self.stream_message("success", f"Text type: {ocr_result.type_of_text}")
        
        chunks = self.chunk_text(cleaned)
        self.store_chunks(chunks, page_number=page_number, source=source)
        
        elapsed = time.time() - start_time
        self.stream_message("success", f"Page {page_number} processed successfully in {elapsed:.2f} seconds")
        
        result["processed"] = True
        result["chunks_created"] = len(chunks)
        result["text_type"] = ocr_result.type_of_text
        result["processing_time"] = elapsed
        
        return result
    
    def process_pdf(self, pdf_path: str, start_page: int = 1, end_page: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a PDF file with batch processing
        
        Args:
            pdf_path: Path to the PDF file
            start_page: First page to process (1-indexed)
            end_page: Last page to process (1-indexed, None for all pages)
            
        Returns:
            Dict: Summary of processing results
        """
        overall_start_time = time.time()
        self.stream_message("pdf", f"Loading PDF: {pdf_path}")
        
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        # Adjust end_page if not specified
        if end_page is None or end_page > total_pages:
            end_page = total_pages
            
        # Validate page range
        if start_page < 1:
            start_page = 1
        if start_page > total_pages:
            error_msg = f"Start page {start_page} exceeds total pages {total_pages}"
            self.stream_message("error", error_msg)
            raise ValueError(error_msg)
            
        self.stream_message("pdf", f"Processing PDF: {pdf_path}")
        self.stream_message("page", f"Pages: {start_page} to {end_page} (of {total_pages})")
        self.stream_message("processing", f"Batch size: {self.batch_size} pages")
        
        # Prepare pages for processing
        self.stream_message("processing", "Extracting text from pages")
        pages_to_process = []
        for i in range(start_page - 1, end_page):
            page_number = i + 1
            raw_text = reader.pages[i].extract_text() or ""
            pages_to_process.append((page_number, raw_text, os.path.basename(pdf_path)))
            
            # Stream progress updates
            if (i - (start_page - 1)) % max(1, (end_page - start_page + 1) // 10) == 0:
                self.stream_progress(i - (start_page - 1) + 1, end_page - start_page + 1, "Text extraction")
        
        self.stream_message("processing", f"Beginning processing of {len(pages_to_process)} pages")
        
        # Process pages in batches
        results = []
        completed_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            future_to_page = {executor.submit(self.process_page, page_data): page_data for page_data in pages_to_process}
            for future in concurrent.futures.as_completed(future_to_page):
                page_data = future_to_page[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update completed count and show progress
                    completed_count += 1
                    if completed_count % max(1, len(pages_to_process) // 10) == 0:
                        self.stream_progress(completed_count, len(pages_to_process), "Page processing")
                        
                except Exception as exc:
                    error_msg = f"Error processing page {page_data[0]}: {exc}"
                    self.stream_message("error", error_msg)
                    results.append({
                        "page_number": page_data[0],
                        "processed": False,
                        "skipped": True,
                        "reason": f"error: {str(exc)}",
                        "chunks_created": 0,
                        "text_type": None,
                        "processing_time": 0
                    })
                    
        # Compile summary
        processed_pages = sum(1 for r in results if r["processed"])
        skipped_pages = sum(1 for r in results if r["skipped"])
        total_chunks = sum(r["chunks_created"] for r in results)
        total_time = time.time() - overall_start_time
        
        # Calculate average processing time for successful pages
        processing_times = [r["processing_time"] for r in results if r["processed"]]
        avg_time_per_page = sum(processing_times) / len(processing_times) if processing_times else 0
        
        summary = {
            "pdf_path": pdf_path,
            "total_pages_in_range": len(pages_to_process),
            "processed_pages": processed_pages,
            "skipped_pages": skipped_pages,
            "total_chunks_created": total_chunks,
            "total_processing_time": total_time,
            "average_time_per_page": avg_time_per_page,
            "page_details": results
        }
        
        self.stream_message("summary", f"Processing Summary:")
        self.stream_message("pdf", f"PDF: {os.path.basename(pdf_path)}")
        self.stream_message("success", f"Pages processed: {processed_pages}/{len(pages_to_process)}")
        self.stream_message("warning", f"Pages skipped: {skipped_pages}")
        self.stream_message("chunk", f"Total chunks created: {total_chunks}")
        self.stream_message("time", f"Total processing time: {total_time:.2f} seconds")
        self.stream_message("time", f"Average time per page: {avg_time_per_page:.2f} seconds")
        self.stream_message("complete", f"PDF processing complete!")
        
        return summary


class StreamlitRagBuilder:
    """Streamlit integration for the RagBuilder class"""
    
    def __init__(self, st=None):
        """
        Initialize the Streamlit RAG Builder
        
        Args:
            st: Streamlit module (pass st from your Streamlit app)
        """
        self.st = st
        self.messages = []
        self.rag_builder = None
    
    def message_callback(self, message: str) -> None:
        """
        Callback function for streaming messages to Streamlit
        
        Args:
            message: The message to display
        """
        if self.st:
            self.messages.append(message)
            # Update the Streamlit UI with the new message
            with self.st.container():
                for msg in self.messages:
                    self.st.text(msg)
    
    def create_rag_builder(self, **kwargs) -> RagBuilder:
        """
        Create a RagBuilder instance with Streamlit integration
        
        Args:
            **kwargs: Arguments to pass to RagBuilder constructor
            
        Returns:
            RagBuilder: Configured RagBuilder instance
        """
        # Clear previous messages
        self.messages = []
        
        # Create RagBuilder with message callback
        self.rag_builder = RagBuilder(
            message_callback=self.message_callback,
            verbose=False,  # Don't print to console, only to Streamlit
            **kwargs
        )
        
        return self.rag_builder
    
    def create_ui(self):
        """Create a Streamlit UI for the RAG Builder"""
        if not self.st:
            raise ValueError("Streamlit module not provided. Pass 'st' to the constructor.")
        
        self.st.title("📚 Advanced RAG Builder")
        self.st.write("Process PDF documents for RAG applications with advanced options")
        
        with self.st.sidebar:
            self.st.header("Configuration")
            
            pdf_file = self.st.file_uploader("Upload PDF", type=["pdf"])
            
            col1, col2 = self.st.columns(2)
            with col1:
                start_page = self.st.number_input("Start Page", min_value=1, value=1)
            with col2:
                end_page = self.st.number_input("End Page", min_value=0, value=0, 
                                               help="0 means process until the end of the document")
            
            batch_size = self.st.slider("Batch Size (pages)", min_value=1, max_value=10, value=2)
            
            self.st.subheader("Models")
            cleanup_model = self.st.selectbox(
                "Cleanup Model",
                ["mistral-small:latest", "llama3:latest", "llama3-groq:latest", "mixtral:latest"]
            )
            
            embedding_model = self.st.selectbox(
                "Embedding Model",
                ["nomic-embed-text:latest", "mxbai-embed-large:latest"]
            )
            
            self.st.subheader("Chunking")
            chunk_size = self.st.slider("Chunk Size", min_value=500, max_value=4000, value=2000, step=100)
            chunk_overlap = self.st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
            
            collection_name = self.st.text_input("Collection Name", value="documents")
            
            process_button = self.st.button("Process PDF")
        
        # Main area for displaying messages
        message_area = self.st.container()
        
        # Process PDF when button is clicked
        if process_button and pdf_file:
            # Save uploaded file to a temporary location
            temp_pdf_path = f"/tmp/{pdf_file.name}"
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            
            # Create RAG Builder
            rag_builder = self.create_rag_builder(
                cleanup_model=cleanup_model,
                embedding_model=embedding_model,
                collection_name=collection_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                batch_size=batch_size
            )
            
            # Process PDF
            with message_area:
                try:
                    end_page_val = None if end_page == 0 else end_page
                    rag_builder.process_pdf(
                        pdf_path=temp_pdf_path,
                        start_page=start_page,
                        end_page=end_page_val
                    )
                except Exception as e:
                    self.st.error(f"Error processing PDF: {str(e)}")


def main():
    """Main function to parse arguments and run the RAG builder"""
    parser = argparse.ArgumentParser(description="Advanced RAG Builder for PDF processing")
    
    # Required arguments
    parser.add_argument("pdf_file", help="Path to the PDF file to process")
    
    # Optional arguments
    parser.add_argument("--start-page", type=int, default=1, help="First page to process (1-indexed)")
    parser.add_argument("--end-page", type=int, help="Last page to process (1-indexed, default: all pages)")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of pages to process in parallel")
    parser.add_argument("--cleanup-model", default="mistral-small:latest", help="Model to use for text cleanup")
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest", help="Model to use for embeddings")
    parser.add_argument("--collection", default="documents", help="Name of the vector collection")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Size of text chunks")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks")
    
    args = parser.parse_args()
    
    # Initialize and run the RAG builder
    rag_builder = RagBuilder(
        cleanup_model=args.cleanup_model,
        embedding_model=args.embedding_model,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size
    )
    
    # Process the PDF
    rag_builder.process_pdf(
        pdf_path=args.pdf_file,
        start_page=args.start_page,
        end_page=args.end_page
    )


if __name__ == "__main__":
    main()
