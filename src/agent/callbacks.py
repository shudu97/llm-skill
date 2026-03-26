"""
Abstract callback interface for agent interactions.
This allows different interfaces (CLI, API, etc.) to implement their own UI logic.
"""

from abc import ABC, abstractmethod


class AgentCallback(ABC):
    """Abstract base class for agent callbacks."""

    @abstractmethod
    def request_approval(self, command: str) -> tuple[bool, str | None]:
        """Request user approval for executing a command.

        Args:
            command: The command that needs approval

        Returns:
            (approved, feedback) — approved is True if permitted, False if rejected.
            feedback is an optional message the user provides when rejecting.
        """
        pass

    @abstractmethod
    def handle_ask_user_question(self, input_data: dict) -> dict:
        """Present Claude's clarifying questions to the user and collect answers.

        Args:
            input_data: The AskUserQuestion tool input containing a 'questions' list.
                Each question has 'question', 'header', 'options', and 'multiSelect'.

        Returns:
            Dict mapping each question string to the user's selected answer label(s).
        """
        pass

    def on_progress(self, message: str) -> None:
        """Optional callback for progress updates.

        Args:
            message: Progress message to display
        """
        pass

    def on_error(self, error: str) -> None:
        """Optional callback for error notifications.

        Args:
            error: Error message to display
        """
        pass

    def on_tool_call(self, tool_name: str, tool_input: dict) -> None:
        """Optional callback when a tool is about to be called.

        Args:
            tool_name: Name of the tool being called
            tool_input: Input arguments for the tool
        """
        pass

    def on_tool_result(self, tool_name: str, result: str) -> None:
        """Optional callback when a tool call completes.

        Args:
            tool_name: Name of the tool that was called
            result: Result returned by the tool
        """
        pass
