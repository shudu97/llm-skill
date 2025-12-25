"""
Abstract callback interface for agent interactions.
This allows different interfaces (CLI, API, etc.) to implement their own UI logic.
"""

from abc import ABC, abstractmethod


class AgentCallback(ABC):
    """Abstract base class for agent callbacks."""

    @abstractmethod
    def request_approval(self, command: str) -> bool:
        """Request user approval for executing a command.

        Args:
            command: The command that needs approval

        Returns:
            True if approved, False if rejected
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
