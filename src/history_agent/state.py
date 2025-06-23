from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage
from langchain.schema import Document
import operator

class ResearchState(TypedDict):
    question: str
    plan: list
    contexts: list
    documents: List[Document]
    draft: str
    critique: str
    final_report: str
    messages: Annotated[List[BaseMessage], operator.add]
    iteration: int = 0
