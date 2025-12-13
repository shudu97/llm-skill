"""
Simple ReAct Agent using LangGraph and Ollama
"""
from typing import TypedDict, Annotated
import operator
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# Define simple tools for the agent
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use this for any math calculations.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2" or "10 * 5")
    """
    try:
        result = eval(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool
def get_word_length(word: str) -> str:
    """Get the length of a word.

    Args:
        word: The word to count characters in
    """
    return f"The word '{word}' has {len(word)} characters."


@tool
def reverse_string(text: str) -> str:
    """Reverse a string.

    Args:
        text: The text to reverse
    """
    return f"Reversed: {text[::-1]}"


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
        self.tools = [calculator, get_word_length, reverse_string]

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
        # Create initial state
        initial_state = {"messages": [HumanMessage(content=user_input)]}

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
        initial_state = {"messages": [HumanMessage(content=user_input)]}

        for step in self.graph.stream(initial_state):
            yield step


def main():
    """Example usage of the ReAct agent."""
    print("Initializing ReAct Agent with Ollama...\n")

    # Create the agent
    agent = ReActAgent(model_name="llama3.2")

    # Example queries
    queries = [
        "What is 25 multiplied by 4?",
        "How many characters are in the word 'LangGraph'?",
        "Reverse the string 'Hello World' and then tell me how many characters it has.",
    ]

    for query in queries:
        print(f"Query: {query}")
        print("-" * 50)

        # Stream the execution to see the agent's reasoning
        for step in agent.stream(query):
            print(step)
            print()

        print("=" * 50)
        print()


if __name__ == "__main__":
    main()
