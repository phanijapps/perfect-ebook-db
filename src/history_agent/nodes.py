"""Graph nodes used to assemble the history research agent."""

from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv

from .chroma_ragclient import ChromaRagClient
from .openrouter_client import ChatOpenRouter
from .prompts import REFLECTION_PROMPT, create_research_prompt
from .state import ResearchState

_DATA_DIR = Path("../_data")
_INDEX_DIR = _DATA_DIR / "chroma_index"

load_dotenv()


def rag_tool_func(query: str, k: int = 4) -> str:
    """Retrieve relevant context from Chroma."""
    rag_client = ChromaRagClient(index_dir=str(_INDEX_DIR))
    docs = rag_client.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)


def planning_node(state: ResearchState) -> dict:
    """Break down the question into research steps."""
    plan_prompt = (
        f"Given the research question: '{state['question']}', "
        "break it down into a numbered list of research steps or sub-questions."
    )
    llm = ChatOpenRouter(model="google/gemini-2.0-flash-001")

    plan = llm.invoke(plan_prompt).content
    steps = [line for line in plan.split('\n') if line.strip()]
    return {"plan": steps}


def execute_plan_node(state: ResearchState) -> dict:
    """Gather context for each research step."""
    context_chunks = []
    for step in state["plan"]:
        context = rag_tool_func(step)
        context_chunks.append({"step": step, "context": context})
    return {"contexts": context_chunks}


def research_node(state: ResearchState) -> dict:
    """Compile a research report from gathered context."""
    llm = ChatOpenRouter(model="google/gemini-2.0-flash-001")

    combined_context = "\n\n".join(chunk["context"] for chunk in state["contexts"])
    research_prompt = (
        f"Based on the following research context, write a detailed report on the question: {state['question']}\n\n"
        f"<context>{combined_context}</context>\n\n"
        f"ONLY use the provided context and dont use any other info."
    )
    report = llm.invoke(research_prompt).content
    return {"draft": report}


def generate_draft(state: ResearchState) -> dict:
    """Generate the initial research draft."""
    llm = ChatOpenRouter(model="google/gemini-2.0-flash-001")
    prompt = create_research_prompt(state["question"], state["contexts"])
    draft = llm.invoke(prompt).content
    return {"draft": draft}


def generate_critique(state: ResearchState) -> dict:
    """Produce a self-critique of the draft."""
    llm = ChatOpenRouter(model="google/gemini-2.0-flash-001")
    prompt = REFLECTION_PROMPT.format(
        draft=state["draft"],
        context="\n\n".join(chunk["context"] for chunk in state["contexts"]),
    )
    critique = llm.invoke(prompt).content
    return {"critique": critique}


def revise_draft(state: ResearchState) -> dict:
    """Revise the draft using the critique."""
    revision_prompt = f"""
    Revise this historical research draft using the critique:

    Original Draft:
    {state['draft']}

    Critique:
    {state['critique']}

    Revised Draft:
    """
    llm = ChatOpenRouter(model="google/gemini-2.0-flash-001")
    revised = llm.invoke(revision_prompt).content
    return {"draft": revised, "iteration": state.get("iteration", 0) + 1}


def should_continue(state: ResearchState) -> str:
    """Decide whether additional revisions are needed."""
    if state.get("iteration", 0) >= 2:
        return "end"
    return "continue"


def compile_report(state: ResearchState) -> dict:
    """Format the final report."""
    return {"final_report": state["draft"]}

