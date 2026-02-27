import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class MainAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    active_skill: str | None
    skill_parameters: dict
    loaded_files: dict          # {path -> schema}


class DataAnalysisState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    file_path: str
    schema: dict                # passed in from main agent, not re-analyzed
    task: str


# Keep backward-compatible alias
AgentState = MainAgentState
