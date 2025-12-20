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

from src.agent.middleware.skill import SkillMiddleware
from src.agent.prompts import get_system_prompt
from src.agent.state import AgentState


# Create the agent
class ReActAgent:
    def __init__(self, model_name: str = "llama3.2"):
        """Initialize the ReAct agent.

        Args:
            model_name: Name of the Ollama model to use (default: llama3.2)
        """
        # Initialize the LLM
        self.llm = ChatOllama(model=model_name, temperature=0)

        # Initialize skill middleware
        self.skill_middleware = SkillMiddleware(skills_dir="src/skills")

        # Create the agent
        self.agent = self._create_agent()

    def _create_agent(self):
        # Generate system prompt with skill summaries from middleware
        system_content = get_system_prompt(self.skill_middleware.skill_summaries)

        agent = create_agent(
            model=self.llm,
            system_prompt=SystemMessage(content=system_content),
            state_schema=AgentState,
            middleware=[
                self.skill_middleware,
                FilesystemFileSearchMiddleware(root_path="upload"),
                ShellToolMiddleware(
                    workspace_root=".",
                    startup_commands=[
                        "export PATH=/opt/homebrew/bin:$PATH",
                        "source .venv/bin/activate",
                    ],
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
