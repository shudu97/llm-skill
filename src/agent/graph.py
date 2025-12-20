"""
Simple ReAct Agent using LangGraph and Ollama
"""

from langchain.agents import create_agent
from langchain.agents.middleware.file_search import FilesystemFileSearchMiddleware
from langchain.agents.middleware.shell_tool import (
    HostExecutionPolicy,
    ShellToolMiddleware,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.agent.prompts import get_system_prompt
from src.agent.state import AgentState
from src.skills import SkillManager
from src.tool import view_skill


# Create the agent
class ReActAgent:
    def __init__(self, model_name: str = "llama3.2"):
        """Initialize the ReAct agent.

        Args:
            model_name: Name of the Ollama model to use (default: llama3.2)
        """
        # Define tools
        self.tools = [view_skill]

        # Initialize skill manager and load summaries
        self.skill_manager = SkillManager()
        self.skill_summaries = self.skill_manager.get_skill_summaries()

        # Initialize the LLM with tools
        self.llm = ChatOllama(model=model_name, temperature=0)

        # Create the agent
        self.agent = self._create_agent()

    def _create_agent(self):
        # Always generate system prompt (even if no skills available)
        system_content = get_system_prompt(self.skill_summaries)

        agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=SystemMessage(content=system_content),
            state_schema=AgentState,
            middleware=[
                FilesystemFileSearchMiddleware(root_path="upload"),
                ShellToolMiddleware(
                    workspace_root=".",
                    startup_commands="source .venv/bin/activate",
                    execution_policy=HostExecutionPolicy(),
                ),
            ],
        )

        return agent

    def run(self, user_input: str) -> str:
        """Run the agent with a user input.

        Args:
            user_input: The user's question or request

        Returns:
            The agent's final response
        """
        initial_state = {"messages": [HumanMessage(content=user_input)]}

        # Run the graph
        result = self.agent.invoke(initial_state)

        # Return the last message content
        return result["messages"][-1].content
