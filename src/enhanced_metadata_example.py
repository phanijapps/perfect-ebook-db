#!/usr/bin/env python3
"""Example script demonstrating the use of enhanced metadata.

This script shows how to use the EnhancedPageMetadata class to generate
markdown with rich metadata without modifying the original faiss_ragbuilder.py.
"""

import argparse
import os
from pathlib import Path

from pypdf import PdfReader

from book_builder.enhanced_metadata import create_enhanced_metadata


def process_page(
    pdf_path: str,
    page_number: int,
    output_dir: str,
    title: str = None,
    author: str = None,
    **additional_metadata
):
    """Process a single page from a PDF and generate enhanced markdown.
    
    Args:
        pdf_path: Path to the PDF file
        page_number: Page number to process (1-indexed)
        output_dir: Directory to save the markdown file
        title: Document title (defaults to PDF filename if None)
        author: Document author (defaults to "Unknown" if None)
        **additional_metadata: Additional metadata to include
    
    Returns:
        str: Path to the generated markdown file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set defaults for title and author
    if title is None:
        title = Path(pdf_path).stem
    
    if author is None:
        author = "Unknown"
    
    # Read the PDF
    reader = PdfReader(pdf_path)
    
    # Validate page number
    if page_number < 1 or page_number > len(reader.pages):
        raise ValueError(f"Page number {page_number} is out of range (1-{len(reader.pages)})")
    
    # Extract text from the page
    page_idx = page_number - 1
    raw_text = reader.pages[page_idx].extract_text() or ""
    
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
    additional_metadata['total_pages'] = len(reader.pages)
    
    # Create enhanced metadata
    metadata = create_enhanced_metadata(
        title=title,
        author=author,
        page_number=page_number,
        **additional_metadata
    )
    
    # Generate markdown with enhanced metadata
    md_content = metadata.to_markdown(raw_text)
    md_path = output_path / f"page_{page_number}.md"
    md_path.write_text(md_content, encoding="utf-8")
    
    return str(md_path)


def main():
    """Main function to parse arguments and process a PDF page."""
    parser = argparse.ArgumentParser(description="Generate enhanced markdown from a PDF page")
    
    # Required arguments
    parser.add_argument("pdf_file", help="Path to the PDF file")
    parser.add_argument("page_number", type=int, help="Page number to process (1-indexed)")
    
    # Optional arguments
    parser.add_argument("--output-dir", default="_output", help="Directory to save the markdown file")
    parser.add_argument("--title", help="Document title (defaults to PDF filename)")
    parser.add_argument("--author", help="Document author (defaults to 'Unknown')")
    parser.add_argument("--category", help="Document category")
    parser.add_argument("--language", help="Document language")
    parser.add_argument("--year", help="Publication year")
    parser.add_argument("--keywords", help="Keywords (comma-separated)")
    
    args = parser.parse_args()
    
    # Process additional metadata
    additional_metadata = {}
    if args.category:
        additional_metadata["category"] = args.category
    if args.language:
        additional_metadata["language"] = args.language
    if args.year:
        additional_metadata["year"] = args.year
    if args.keywords:
        additional_metadata["keywords"] = args.keywords
    
    # Process the page
    md_path = process_page(
        pdf_path=args.pdf_file,
        page_number=args.page_number,
        output_dir=args.output_dir,
        title=args.title,
        author=args.author,
        **additional_metadata
    )
    
    print(f"Generated markdown file: {md_path}")


if __name__ == "__main__":
    main()
