"""
CLI runner for the ReAct agent.
Handles the command-line interface and user interaction loop.
"""

from prompt_toolkit import prompt

from src.agent.graph import ReActAgent
from src.cli.callbacks import CLICallback
from src.utils.logger import logger


def run_cli(model_name: str = "qwen3:8b") -> None:
    """Run the CLI interface for the ReAct agent.

    Args:
        model_name: Name of the Ollama model to use (default: qwen3:8b)
    """
    # Initialize CLI callback
    callback = CLICallback()

    # Initialize the agent with CLI callback
    agent = ReActAgent(callback=callback, model_name=model_name)

    logger.info("Agent initialized. Type 'exit' or 'quit' to end the conversation.\n")

    # Continuous conversation loop
    while True:
        # Get query from user input with placeholder
        query = prompt("\n>>> ", placeholder="Send a message").strip()

        # Check for exit commands
        if query.lower() in ["exit", "quit", "q"]:
            logger.info("Ending conversation. Goodbye!")
            break

        # Skip empty queries
        if not query:
            continue

        # Get the response
        try:
            response = agent.run(query)
            logger.info(f"Response: {response}")
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            logger.info("Please try again or type 'exit' to quit.")
