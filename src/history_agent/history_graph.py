"""History research agent assembled using LangGraph."""

from langgraph.graph import StateGraph, END

from .nodes import (
    planning_node,
    execute_plan_node,
    research_node,
    generate_critique,
    revise_draft,
    compile_report,
    should_continue,
)
from .state import ResearchState

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
graph.add_edge("plan_node", "execute_plan")
graph.add_edge("execute_plan", "research_plan")
graph.add_edge("research_plan", "critique_draft")
graph.add_conditional_edges(
    "critique_draft",
    should_continue,
    {"continue": "revise", "end": "finalize"},
)
graph.add_edge("revise", "critique_draft")
graph.add_edge("finalize", END)

# Compile the agent
research_agent = graph.compile()
