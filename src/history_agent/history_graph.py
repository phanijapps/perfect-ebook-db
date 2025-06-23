from typing import TypedDict, Annotated, List, Union, Optional, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOpenAI
from langchain_core.utils.utils import secret_from_env
from pathlib import Path
from dotenv import load_dotenv
import os
import sys

# Add the parent directory to sys.path to allow absolute imports
sys.path.append(str(Path(__file__).parent.parent))
from history_agent.chroma_ragclient import ChromaRagClient
from history_agent.state import ResearchState

load_dotenv()

_DATA_DIR = Path("../_data")
_INDEX_DIR = _DATA_DIR / "chroma_index"
_PAGES_DIR = _DATA_DIR / "pages"
_DB_PATH = _DATA_DIR / "pages.sqlite"


class ChatOpenRouter(ChatOpenAI):
    
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

# Reflection prompt template
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

def format_documents(documents):
    return "\n\n".join(
        f"Source: {getattr(doc, 'metadata', {}).get('source', 'unknown')}\nContent: {doc.page_content}"
        for doc in documents
    )


def create_research_prompt(question, documents):
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


def rag_tool_func(query: str, k: int = 4):
    rag_client = ChromaRagClient(index_dir=str(_INDEX_DIR))
    docs = rag_client.similarity_search(query, k=k)
    # Format as needed for the planner/executor
    return "\n\n".join(d.page_content for d in docs)




def planning_node(state):
    """
    Plan the research..
    """
    plan_prompt = (
        f"Given the research question: '{state['question']}', "
        "break it down into a numbered list of research steps or sub-questions."
    )
    llm = ChatOpenRouter(model="google/gemini-2.0-flash-001") # Or ChatOllama()
    
    plan = llm.invoke(plan_prompt).content
    steps = [line for line in plan.split('\n') if line.strip()]
    print(steps)
    return {"plan": steps}



def execute_plan_node(state):
    """
    Execute each research
    """
    context_chunks = []
    for step in state["plan"]:
        context = rag_tool_func(step)
        context_chunks.append({"step": step, "context": context})
    return {"contexts": context_chunks}


def research_node(state):
    # Combine all context chunks into a single context string
    llm = ChatOpenRouter(model="google/gemini-2.0-flash-001") # Or ChatOllama()

    combined_context = "\n\n".join(chunk["context"] for chunk in state["contexts"])
    research_prompt = (
        f"Based on the following research context, write a detailed report on the question: {state['question']}\n\n"
        f"<context>{combined_context}</context>\n\n"
        f"ONLY use the provided context and dont use any other info."
    )
    report = llm.invoke(research_prompt).content
    return {"draft": report}



def generate_draft(state: ResearchState):
    """Generate initial research draft"""
    llm = ChatOpenRouter(model="google/gemini-2.0-flash-001") # Or ChatOllama()
    prompt = create_research_prompt(state["question"], state["contexts"])
    draft = llm.invoke(prompt).content
    return {"draft": draft}

def generate_critique(state: ResearchState):
    """Generate self-critique using reflection"""
    llm = ChatOpenRouter(model="google/gemini-2.0-flash-001")  # More conservative model
    prompt = REFLECTION_PROMPT.format(
        draft=state["draft"],
        context="\n\n".join(chunk["context"] for chunk in state["contexts"])
    )
    critique = llm.invoke(prompt).content
    return {"critique": critique}

def revise_draft(state: ResearchState):
    """Incorporate critique into revised draft"""
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

def should_continue(state: ResearchState):
    """Decide whether to continue refining"""
    if state.get("iteration", 0) >= 2:  # Max 2 refinements
        return "end"
    return "continue"

def compile_report(state: ResearchState):
    """Format final output"""
    return {"final_report": state["draft"]}



# Initialize graph
graph = StateGraph(ResearchState)

# Define nodes
graph.add_node("plan_node", planning_node)
graph.add_node("execute_plan", execute_plan_node)
graph.add_node("research_plan", research_node)
graph.add_node("critique_draft", generate_critique)
graph.add_node("revise", revise_draft)
graph.add_node("finalize", compile_report)

# Define graph structure
graph.set_entry_point("plan_node")
graph.add_edge("plan_node","execute_plan")
graph.add_edge("execute_plan","research_plan")
graph.add_edge("research_plan", "critique_draft")
graph.add_conditional_edges(
    "critique_draft",
    should_continue,
    {
        "continue": "revise",
        "end": "finalize"
    }
)
graph.add_edge("revise", "critique_draft")  # Loop back for refinement
graph.add_edge("finalize", END)

# Compile the agent
research_agent = graph.compile()
