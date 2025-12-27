"""
CLI implementation of agent callbacks.
Uses questionary for interactive prompts and terminal output.
"""

import json

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

    def on_tool_call(self, tool_name: str, tool_input: dict) -> None:
        """Display tool call to terminal.

        Args:
            tool_name: Name of the tool being called
            tool_input: Input arguments for the tool
        """
        # Skip displaying bash tool calls (already shown in approval prompt)
        if tool_name == "bash":
            return

        # Format tool input in a readable way
        if tool_input:
            # Extract the most important parameter for display
            if len(tool_input) == 1:
                # Single parameter - show it inline
                key, value = next(iter(tool_input.items()))
                print(f"\n\033[36m▶\033[0m \033[1m{tool_name}\033[0m: {value}")
            else:
                # Multiple parameters - show key ones
                params = ", ".join(f"{k}={repr(v)[:50]}" for k, v in tool_input.items())
                print(f"\n\033[36m▶\033[0m \033[1m{tool_name}\033[0m({params})")
        else:
            print(f"\n\033[36m▶\033[0m \033[1m{tool_name}\033[0m")

    def on_tool_result(self, tool_name: str, result: str) -> None:
        """Display tool result to terminal.

        Args:
            tool_name: Name of the tool that was called
            result: Result returned by the tool
        """
        # Truncate long results for display
        max_length = 200
        if len(result) > max_length:
            display_result = result[:max_length] + "..."
        else:
            display_result = result

        print(f"\n\033[32m✓ Tool Result:\033[0m \033[90m{display_result}\033[0m\n")
