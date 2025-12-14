"""
Simple ReAct Agent using LangGraph and Ollama
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.skills import SkillManager
from src.tool import execute_script, view_skill


# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


# Create the agent
class ReActAgent:
    def __init__(self, model_name: str = "llama3.2"):
        """Initialize the ReAct agent.

        Args:
            model_name: Name of the Ollama model to use (default: llama3.2)
        """
        # Define tools
        self.tools = [execute_script, view_skill]

        # Initialize skill manager and load summaries
        self.skill_manager = SkillManager()
        self.skill_summaries = self.skill_manager.get_skill_summaries()

        # Initialize the LLM with tools
        self.llm = ChatOllama(model=model_name, temperature=0)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Create the graph
        self.graph = self._create_graph()

    def _should_continue(self, state: AgentState) -> str:
        """Determine if the agent should continue or end."""
        last_message = state["messages"][-1]

        # If there are no tool calls, we're done
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return "end"
        return "continue"

    def _call_model(self, state: AgentState) -> dict:
        """Call the LLM with the current state."""
        messages = state["messages"]
        response = self.llm_with_tools.invoke(messages)
        print(response)
        return {"messages": [response]}

    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow for the ReAct agent."""
        # Create the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self.tools))

        # Set entry point
        workflow.set_entry_point("agent")

        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )

        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")

        # Compile the graph
        return workflow.compile()

    def run(self, user_input: str) -> str:
        """Run the agent with a user input.

        Args:
            user_input: The user's question or request

        Returns:
            The agent's final response
        """
        # Create initial state with system message containing skill summaries
        messages = []

        # Add system message with skill summaries if available
        if self.skill_summaries and self.skill_summaries != "No skills available.":
            system_content = f"""You are a helpful assistant with access to specialized skills.

{self.skill_summaries}

When a user asks for something that matches a skill, use the view_skill tool to load the detailed instructions for that skill, then follow those instructions to complete the task."""
            messages.append(SystemMessage(content=system_content))

        # Add user message
        messages.append(HumanMessage(content=user_input))

        initial_state = {"messages": messages}

        # Run the graph
        result = self.graph.invoke(initial_state)

        # Return the last message content
        return result["messages"][-1].content

    def stream(self, user_input: str):
        """Stream the agent's execution step by step.

        Args:
            user_input: The user's question or request

        Yields:
            Each step of the agent's execution
        """
        # Create initial state with system message containing skill summaries
        messages = []

        # Add system message with skill summaries if available
        if self.skill_summaries and self.skill_summaries != "No skills available.":
            system_content = f"""You are a helpful assistant with access to specialized skills.

{self.skill_summaries}

When a user asks for something that matches a skill, use the view_skill tool to load the detailed instructions for that skill, then follow those instructions to complete the task."""
            messages.append(SystemMessage(content=system_content))

        # Add user message
        messages.append(HumanMessage(content=user_input))

        initial_state = {"messages": messages}

        for step in self.graph.stream(initial_state):
            yield step
