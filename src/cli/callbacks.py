"""
CLI implementation of agent callbacks.
Uses questionary for interactive prompts and terminal output.
"""

import questionary

from src.agent.callbacks import AgentCallback
from src.utils.logger import logger


class CLICallback(AgentCallback):
    """CLI implementation of agent callbacks."""

    def request_approval(self, command: str) -> bool:
        """Request user approval for executing a command via terminal UI.

        Args:
            command: The command that needs approval

        Returns:
            True if approved, False if rejected
        """
        # Display the command with formatting
        print(f"\n\033[1mBash Command:\033[0m {command}\n")

        # Use questionary for a nicer selection interface
        decision = questionary.select(
            "Do you want to execute this command?",
            choices=[
                questionary.Choice("Yes", value="approve"),
                questionary.Choice("No", value="reject"),
            ],
            style=questionary.Style(
                [
                    ("selected", "fg:#673ab7 bold"),
                    ("pointer", "fg:#673ab7 bold"),
                    ("question", "bold"),
                ]
            ),
        ).ask()

        return decision == "approve"

    def on_progress(self, message: str) -> None:
        """Display progress message to terminal.

        Args:
            message: Progress message to display
        """
        logger.info(message)

    def on_error(self, error: str) -> None:
        """Display error message to terminal.

        Args:
            error: Error message to display
        """
        logger.error(error)
