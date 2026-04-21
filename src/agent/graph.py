"""
ReAct Agent backed by the Claude Agent SDK.
"""

import asyncio
import os
import random

from claude_agent_sdk import (
    ClaudeAgentOptions,
    HookMatcher,
    ResultMessage,
    SystemMessage,
    TaskProgressMessage,
    query,
)
from rich.console import Console

from src.agent.callbacks import AgentCallback
from src.agent.constants import SPINNER_WORDS
from src.agent.plugins import register_plugins

_console = Console()


class ReActAgent:
    def __init__(
        self,
        callback: AgentCallback,
        user_id: str,
        session_id: str | None,
    ):
        """Initialize the ReAct agent.

        Args:
            callback: Callback interface for user interactions
            user_id: Identity of the user running the agent
            session_id: Existing session ID to resume, or None for a new conversation
        """
        self.callback = callback
        self.user_id = user_id
        self.session_id = session_id

    async def run(self, user_input: str) -> tuple[str, str | None]:
        """Run the agent with a user input.

        Args:
            user_input: The user's question or request

        Returns:
            (response_text, session_id) — session_id can be used to resume later
        """
        callback = self.callback

        new_session_id: str | None = None
        final_result: str = ""

        plugins, add_dirs = register_plugins(os.getenv("PLUGIN_DIR"))

        initial_word = random.choice(SPINNER_WORDS)
        with _console.status(f"[#00BFFF]{initial_word}...[/#00BFFF]", spinner="dots", spinner_style="#00BFFF") as status:
            async def tool_display_hook(input_data, tool_use_id, context):
                tool_name = input_data.get("tool_name", "")
                tool_input = input_data.get("tool_input", {})
                if tool_name.lower() != "bash":
                    status.stop()
                    callback.on_tool_call(tool_name, tool_input)
                    status.start()
                return {}

            async def bash_approval_hook(input_data, tool_use_id, context):
                command = input_data.get("tool_input", {}).get("command", "")
                status.stop()
                approved, feedback = await asyncio.to_thread(callback.request_approval, command)
                status.start()
                snippet = command[:60] + "..." if len(command) > 60 else command
                status.update(f"Bash({snippet})")
                if not approved:
                    return {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": feedback or "User rejected",
                        }
                    }
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow",
                    }
                }

            options = ClaudeAgentOptions(
                allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Agent"],
                setting_sources=["project"],
                resume=self.session_id,
                plugins=plugins,
                add_dirs=add_dirs,
                hooks={
                    "PreToolUse": [
                        HookMatcher(matcher=None, hooks=[tool_display_hook]),
                        HookMatcher(matcher="Bash", hooks=[bash_approval_hook]),
                    ]
                },
            )

            async for message in query(prompt=user_input, options=options):
                if isinstance(message, TaskProgressMessage):
                    status.update(message.description)
                elif isinstance(message, SystemMessage):
                    if hasattr(message, "data") and message.data:
                        new_session_id = message.data.get("session_id")
                elif isinstance(message, ResultMessage):
                    final_result = message.result or ""
                    if new_session_id is None:
                        new_session_id = message.session_id

        # Persist session_id for subsequent calls
        self.session_id = new_session_id
        return final_result, new_session_id
