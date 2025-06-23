# Metadata for FAISS RAG Builder

This module extends the `PageMetadata` class from `faiss_ragbuilder.py` to provide improved markdown generation without modifying the original file.

## Features

- Extends the original `PageMetadata` class without modifying it
- Adds support for additional metadata fields
- Improves markdown formatting with better section detection
- Provides a wrapper for the `FaissRagBuilder` class that uses the extended metadata

## Usage

### Basic Usage

```python
from book_builder.metadata import create_metadata

# Create metadata with additional fields
metadata = create_metadata(
    title="Example Document",
    author="John Doe",
    page_number=42,
    category="Technical",
    language="English",
    year="2023"
)

# Generate markdown with metadata
markdown = metadata.to_markdown("Your content here")
```

### Using the RagBuilder

```python
from book_builder.rag_example import RagBuilder

# Create a RAG builder
builder = RagBuilder(
    index_dir="_data/faiss_index",
    output_dir="_data/pages",
    db_path="_data/pages.sqlite"
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
```

### Command Line Usage

You can also use the standalone script to process a single page from a PDF:

```bash
python src/metadata_example.py example.pdf 1 --title "Example Document" --author "John Doe" --category "Technical" --language "English"
```

## How It Works

The `PageMetadata` class extends the original `PageMetadata` class and overrides the `to_markdown` method to provide enhanced functionality:

1. It includes all the basic metadata from the original class
2. It adds support for additional metadata fields
3. It improves the formatting of the content by detecting and formatting headings
4. It provides a more structured output with clear sections

This approach allows you to use the enhanced functionality without modifying the original `faiss_ragbuilder.py` file.
