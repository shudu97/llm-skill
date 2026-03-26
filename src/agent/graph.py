"""
ReAct Agent backed by the Claude Agent SDK.
"""

import asyncio

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    HookMatcher,
    ResultMessage,
    SystemMessage,
    ToolUseBlock,
    query,
)

from src.agent.callbacks import AgentCallback


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

        async def bash_approval_hook(input_data, tool_use_id, context):
            command = input_data.get("tool_input", {}).get("command", "")
            approved, feedback = await asyncio.to_thread(callback.request_approval, command)
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
            hooks={
                "PreToolUse": [
                    HookMatcher(matcher="Bash", hooks=[bash_approval_hook])
                ]
            },
        )

        new_session_id: str | None = None
        final_result: str = ""

        async for message in query(prompt=user_input, options=options):
            if isinstance(message, SystemMessage):
                if hasattr(message, "data") and message.data:
                    new_session_id = message.data.get("session_id")
            elif isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        callback.on_tool_call(block.name, block.input)
            elif isinstance(message, ResultMessage):
                final_result = message.result or ""
                if new_session_id is None:
                    new_session_id = message.session_id

        # Persist session_id for subsequent calls
        self.session_id = new_session_id
        return final_result, new_session_id
