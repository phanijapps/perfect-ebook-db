"""Prompt templates used by the history agent."""

from langchain.schema import Document


def format_documents(documents: list[Document]) -> str:
    """Format a list of documents into a single string."""
    return "\n\n".join(
        f"Source: {getattr(doc, 'metadata', {}).get('source', 'unknown')}\nContent: {doc.page_content}"
        for doc in documents
    )


REFLECTION_PROMPT = """
As a history research specialist, critique this draft about Indian history:
- Identify factual inaccuracies
- Note missing key events
- Suggest deeper analysis areas
- Flag unsupported claims

Draft: {draft}
Context: {context}

Provide detailed critique:
"""


def create_research_prompt(question: str, documents: list[Document]) -> str:
    """Build the research prompt using provided documents."""
    return f"""
    As an Indian history specialist, write a detailed report on:
    '{question}'

    Use these verified sources:
    {format_documents(documents)}

    Include:
    - Chronological events
    - Key figures
    - Cultural impact
    - Primary sources
    """

