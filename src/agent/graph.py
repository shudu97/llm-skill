"""
Simple ReAct Agent using LangGraph and Ollama
"""

import os
import sqlite3

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.file_search import FilesystemFileSearchMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from src.agent.callbacks import AgentCallback
from src.agent.middleware.bash import BashMiddleware
from src.agent.middleware.skill import SkillMiddleware
from src.agent.prompts import get_system_prompt
from src.agent.state import MainAgentState
from src.agent.subagent import DataAnalysisSubagent


# Create the agent
class ReActAgent:
    def __init__(
        self,
        callback: AgentCallback,
        user_id: str,
        thread_id: str,
        model_name: str = "llama3.2",
        db_path: str | None = None,
    ):
        """Initialize the ReAct agent.

        Args:
            callback: Callback interface for user interactions
            user_id: Identity of the user running the agent
            thread_id: Conversation thread identifier
            model_name: Name of the Ollama model to use (default: llama3.2)
            db_path: Path to the SQLite database file (defaults to AGENT_DB_PATH env var or agent.db)
        """
        self.callback = callback
        self.user_id = user_id
        self.thread_id = thread_id

        # Initialize the LLM
        self.llm = ChatOllama(model=model_name, temperature=0)

        # Initialize middleware
        self.skill_middleware = SkillMiddleware(skills_dir=os.getenv("SKILL_DIR"))
        self.file_search_middleware = FilesystemFileSearchMiddleware(root_path="")
        self.bash_middleware = BashMiddleware()
        self.hilp_middleware = HumanInTheLoopMiddleware(interrupt_on={"bash": True})

        # Resolve db path
        resolved_db_path = db_path or os.getenv("AGENT_DB_PATH", "data/agent.db")

        # Initialize subagents
        self.data_analysis_subagent = DataAnalysisSubagent(
            bash_middleware=self.bash_middleware,
            model_name=model_name,
            db_path=resolved_db_path,
        )

        # Create the agent
        self.agent = self._create_agent(resolved_db_path)

    def _create_agent(self, db_path: str):
        # Generate system prompt with skill summaries from middleware
        system_content = get_system_prompt(self.skill_middleware.skill_summaries)

        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)

        agent = create_agent(
            model=self.llm,
            system_prompt=SystemMessage(content=system_content),
            state_schema=MainAgentState,
            tools=[
                self.data_analysis_subagent.get_tool(),
            ],
            middleware=[
                self.hilp_middleware,
                self.skill_middleware,
                self.file_search_middleware,
                self.bash_middleware,
            ],
            checkpointer=checkpointer,
        )

        return agent

    def _process_messages_for_tool_calls(self, messages: list) -> None:
        """Process messages to extract and display tool calls and results.

        Args:
            messages: List of messages from the agent
        """
        for msg in messages:
            # Check for AI messages with tool calls
            if (
                isinstance(msg, AIMessage)
                and hasattr(msg, "tool_calls")
                and msg.tool_calls
            ):
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get("name", "unknown")
                    tool_input = tool_call.get("args", {})
                    self.callback.on_tool_call(tool_name, tool_input)

            # Check for tool messages with results
            elif isinstance(msg, ToolMessage):
                tool_name = msg.name if hasattr(msg, "name") else "unknown"
                result = msg.content
                self.callback.on_tool_result(tool_name, result)

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
                "thread_id": self.thread_id,
            }
        }

        # Run the graph with streaming to capture tool calls
        final_result = None
        for event in self.agent.stream(
            initial_state, config=config, stream_mode="values"
        ):
            messages = event.get("messages", [])
            # Process only new messages (skip initial user message)
            if len(messages) > 1:
                self._process_messages_for_tool_calls([messages[-1]])
            final_result = event

        # Extract the bash command from the interrupt data
        interrupt_data = final_result.get("__interrupt__", [])
        if interrupt_data and len(interrupt_data) > 0:
            # __interrupt__ is a list, get the first interrupt item
            first_interrupt = interrupt_data[0]
            command = "Unknown Command"

            # Extract command from the Interrupt object structure
            try:
                action_requests = first_interrupt.value.get("action_requests", [])
                if action_requests:
                    command = action_requests[0]["args"]["command"]
            except (KeyError, IndexError, AttributeError):
                pass  # Keep default "Unknown Command"

            # Request approval through callback (interface-agnostic)
            approved = self.callback.request_approval(command)

            if approved:
                # Stream the resumed execution
                for event in self.agent.stream(
                    Command(resume={"decisions": [{"type": "approve"}]}),
                    config=config,
                    stream_mode="values",
                ):
                    messages = event.get("messages", [])
                    if messages:
                        self._process_messages_for_tool_calls([messages[-1]])
                    final_result = event
            else:
                # Stream the rejection
                for event in self.agent.stream(
                    Command(resume={"decisions": [{"type": "reject"}]}),
                    config=config,
                    stream_mode="values",
                ):
                    final_result = event

        # Return the last message content
        return final_result["messages"][-1].content
