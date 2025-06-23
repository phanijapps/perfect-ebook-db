"""Metadata handling for the FAISS RAG builder.

This module extends the PageMetadata class from faiss_ragbuilder.py to provide
improved markdown generation without modifying the original file.
"""

from __future__ import annotations

from typing import Dict, Any, Optional

from .faiss_ragbuilder import PageMetadata as BasePageMetadata


class PageMetadata(BasePageMetadata):
    """Extended version of PageMetadata with additional metadata capabilities."""

    def __init__(
        self,
        title: str,
        author: str,
        page_number: int,
        additional_metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize with base metadata and optional additional metadata.
        
        Args:
            title: The title of the document
            author: The author of the document
            page_number: The page number
            additional_metadata: Optional dictionary of additional metadata
        """
        super().__init__(title=title, author=author, page_number=page_number)
        self.additional_metadata = additional_metadata or {}

    def to_markdown(self, content: str) -> str:
        """Generate markdown with metadata and content.
        
        This overrides the base to_markdown method to include additional metadata
        and format the content in a more structured way.
        
        Args:
            content: The page content
            
        Returns:
            str: Formatted markdown with metadata and content
        """
        # Start with basic metadata
        md_parts = [
            "# Page Information\n",
            "## Basic Metadata\n",
            "|Title| Page #  | Author |\n",
            "|--|--|--|\n",
            f"| {self.title} | {self.page_number}  | {self.author}|\n\n",
        ]
        
        # Add additional metadata if present
        if self.additional_metadata:
            md_parts.append("## Additional Metadata\n")
            md_parts.append("|Property|Value|\n")
            md_parts.append("|--|--|\n")
            
            for key, value in self.additional_metadata.items():
                md_parts.append(f"| {key} | {value} |\n")
            md_parts.append("\n")
        
        # Add content with better formatting
        md_parts.append("## Content\n")
        
        # Process content to identify potential sections
        lines = content.split("\n")
        formatted_content = []
        
        for line in lines:
            # Check if line might be a heading and format accordingly
            stripped = line.strip()
            if stripped and all(c.isupper() for c in stripped if c.isalpha()):
                # Likely a heading in all caps
                formatted_content.append(f"### {stripped}\n")
            elif stripped and stripped.startswith("Chapter") or stripped.startswith("CHAPTER"):
                # Chapter heading
                formatted_content.append(f"### {stripped}\n")
            else:
                formatted_content.append(line + "\n")
        
        md_parts.append("".join(formatted_content))
        
        return "".join(md_parts)


def create_metadata(
    title: str,
    author: str,
    page_number: int,
    **additional_metadata
) -> PageMetadata:
    """Factory function to create a PageMetadata instance.
    
    Args:
        title: The title of the document
        author: The author of the document
        page_number: The page number
        **additional_metadata: Additional metadata as keyword arguments
        
    Returns:
        PageMetadata: An instance with the provided metadata
    """
    return PageMetadata(
        title=title,
        author=author,
        page_number=page_number,
        additional_metadata=additional_metadata
    )
