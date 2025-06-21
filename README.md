# 📚 Advanced RAG Builder

This project uses LangChain and Ollama to:
- Clean PDF Books pages using a local LLM (e.g., mistral or llama3)
- Chunk cleaned text for downstream RAG pipelines
- Store embeddings in a PostgreSQL vector database
- Provide rich feedback with emoji-enhanced messages

## ✨ Features

- **Object-Oriented Design**: Clean, modular code structure
- **Batch Processing**: Control the number of pages to process in parallel
- **Model Selection**: Choose different cleanup and embedding models
- **Page Range Control**: Specify start and end pages for processing
- **Rich Feedback**: Detailed progress tracking with emoji-enhanced messages
- **Streamlit Integration**: User-friendly web interface

## 🚀 Getting Started

### Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file with your PostgreSQL database credentials:

```
POSTGRES_DB=vec_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USERNAME=your_username
POSTGRES_PASSWORD=your_password
```

## 🖥️ Command Line Usage

Process a PDF file from the command line:

```bash
python Ragbuilder_advanced.py /path/to/your/document.pdf --start-page 1 --end-page 50 --batch-size 4
```

### Command Line Options

- `pdf_file`: Path to the PDF file to process (required)
- `--start-page`: First page to process (default: 1)
- `--end-page`: Last page to process (default: process all pages)
- `--batch-size`: Number of pages to process in parallel (default: 1)
- `--cleanup-model`: Model to use for text cleanup (default: "mistral-small:latest")
- `--embedding-model`: Model to use for embeddings (default: "nomic-embed-text:latest")
- `--collection`: Name of the vector collection (default: "documents")
- `--chunk-size`: Size of text chunks (default: 2000)
- `--chunk-overlap`: Overlap between chunks (default: 200)

## 🌐 Streamlit App

Run the Streamlit web interface:

```bash
streamlit run streamlit_app.py
```

The Streamlit app provides a user-friendly interface with:
- PDF file upload
- Page range selection
- Model selection
- Chunking parameter configuration
- Real-time processing feedback

## 🧩 Integration

### Using in Your Own Projects

```python
from Ragbuilder_advanced import RagBuilder

# Initialize the RAG builder
rag_builder = RagBuilder(
    cleanup_model="mistral-small:latest",
    embedding_model="nomic-embed-text:latest",
    collection_name="my_documents",
    batch_size=4
)

# Process a PDF
results = rag_builder.process_pdf(
    pdf_path="/path/to/document.pdf",
    start_page=10,
    end_page=50
)

# Access the results
print(f"Processed {results['processed_pages']} pages")
print(f"Created {results['total_chunks_created']} chunks")
```

### Streamlit Integration

```python
import streamlit as st
from Ragbuilder_advanced import StreamlitRagBuilder

# Initialize with Streamlit
rag_builder_ui = StreamlitRagBuilder(st)

# Create the UI
rag_builder_ui.create_ui()
```
