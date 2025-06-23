"""Tests for the enhanced metadata functionality."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from book_builder.enhanced_metadata import EnhancedPageMetadata, create_enhanced_metadata


def test_enhanced_metadata_creation():
    """Test creating enhanced metadata."""
    meta = create_enhanced_metadata(
        title="Test Document",
        author="Test Author",
        page_number=42,
        category="Technical",
        language="English",
        year="2023"
    )
    
    assert meta.title == "Test Document"
    assert meta.author == "Test Author"
    assert meta.page_number == 42
    assert meta.additional_metadata["category"] == "Technical"
    assert meta.additional_metadata["language"] == "English"
    assert meta.additional_metadata["year"] == "2023"


def test_enhanced_to_markdown():
    """Test the enhanced to_markdown method."""
    meta = EnhancedPageMetadata(
        title="Test Document",
        author="Test Author",
        page_number=42,
        additional_metadata={
            "category": "Technical",
            "language": "English"
        }
    )
    
    content = "CHAPTER 1\nThis is a test content.\nWith multiple lines.\nANOTHER SECTION\nMore content here."
    md = meta.to_markdown(content)
    
    # Check that the markdown contains the basic metadata
    assert "# Page Information" in md
    assert "## Basic Metadata" in md
    assert "Test Document" in md
    assert "Test Author" in md
    assert "42" in md
    
    # Check that the additional metadata is included
    assert "## Additional Metadata" in md
    assert "category" in md
    assert "Technical" in md
    assert "language" in md
    assert "English" in md
    
    # Check that the content is properly formatted
    assert "## Content" in md
    assert "### CHAPTER 1" in md
    assert "This is a test content." in md
    assert "### ANOTHER SECTION" in md
    assert "More content here." in md


def test_enhanced_to_markdown_no_additional_metadata():
    """Test the enhanced to_markdown method without additional metadata."""
    meta = EnhancedPageMetadata(
        title="Test Document",
        author="Test Author",
        page_number=42
    )
    
    content = "This is a test content.\nWith multiple lines."
    md = meta.to_markdown(content)
    
    # Check that the markdown contains the basic metadata
    assert "# Page Information" in md
    assert "## Basic Metadata" in md
    
    # Check that there's no additional metadata section
    assert "## Additional Metadata" not in md
    
    # Check that the content is included
    assert "## Content" in md
    assert "This is a test content." in md
    assert "With multiple lines." in md


def test_create_enhanced_metadata_factory():
    """Test the factory function for creating enhanced metadata."""
    # Test with no additional metadata
    meta1 = create_enhanced_metadata(
        title="Test Document",
        author="Test Author",
        page_number=42
    )
    assert isinstance(meta1, EnhancedPageMetadata)
    assert meta1.additional_metadata == {}
    
    # Test with additional metadata
    meta2 = create_enhanced_metadata(
        title="Test Document",
        author="Test Author",
        page_number=42,
        category="Technical",
        language="English"
    )
    assert isinstance(meta2, EnhancedPageMetadata)
    assert meta2.additional_metadata["category"] == "Technical"
    assert meta2.additional_metadata["language"] == "English"
