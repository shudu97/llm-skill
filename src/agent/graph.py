"""
Simple ReAct Agent using LangGraph and Ollama
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.file_search import FilesystemFileSearchMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.agent.middleware.bash import execute_script
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
        self.file_search_middleware = FilesystemFileSearchMiddleware(root_path="upload")
        self.hilp_middleware = HumanInTheLoopMiddleware(
            interrupt_on={"view_skill": True}
        )

        # Create the agent
        self.agent = self._create_agent()

    def _create_agent(self):
        # Generate system prompt with skill summaries from middleware
        system_content = get_system_prompt(self.skill_middleware.skill_summaries)

        agent = create_agent(
            model=self.llm,
            tools=[execute_script],
            system_prompt=SystemMessage(content=system_content),
            state_schema=AgentState,
            middleware=[
                self.hilp_middleware,
                self.skill_middleware,
                self.file_search_middleware,
            ],
            checkpointer=InMemorySaver(),
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

        config = {
            "configurable": {
                "thread_id": "my_thread_id",
            }
        }

        # Run the graph
        result = self.agent.invoke(initial_state, config=config)

        if result.get("__interrupt__"):
            view_decision = input("View Decision: ")
            # shell_decision = input("Shell Decision: ")

            if view_decision == "approve":
                result = self.agent.invoke(
                    Command(resume={"decisions": [{"type": "approve"}]}),
                    config=config,
                )

        # Return the last message content
        return result["messages"][-1].content
