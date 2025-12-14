import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
