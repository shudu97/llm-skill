import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


class MainAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    session_summary: str
    active_skill: str | None
    skill_parameters: dict
    loaded_files: dict          # {path -> schema}


class DataAnalysisState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    file_path: str
    schema: dict                # passed in from main agent, not re-analyzed
    task: str
    last_script_path: str | None


# Keep backward-compatible alias
AgentState = MainAgentState
