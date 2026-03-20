"""
LiteLLM pre-call hook to remap tool call IDs to unique UUIDs.

Command-A (and similar enterprise models) generate sequential tool call IDs
(e.g. call_1, call_2) that restart from call_1 on every API request. When a
session is resumed, the conversation history already contains these IDs, so the
model's new call_1 collides with the historical call_1, causing a 400 error:
  "tool call 'id' is not unique"

Fix: before forwarding each request to the enterprise gateway, remap every
tool_call ID in the history to a unique UUID. Both the assistant message
tool_calls[].id and the corresponding tool message tool_call_id are updated
together so the pairing stays consistent.
"""

import uuid

from litellm.integrations.custom_logger import CustomLogger


class ToolIdRemapHook(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        messages = data.get("messages")
        if not messages:
            return data

        # id_map is overwritten per assistant message so that each tool_use
        # occurrence gets a fresh UUID. The subsequent tool_result blocks (which
        # always immediately follow their parent assistant message) pick up the
        # most-recent mapping, keeping pairs consistent.
        id_map: dict[str, str] = {}

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if not isinstance(content, list):
                continue

            # Anthropic format: assistant messages contain tool_use blocks
            if role == "assistant":
                for block in content:
                    if block.get("type") == "tool_use":
                        old_id = block.get("id")
                        if old_id:
                            new_id = f"call_{uuid.uuid4().hex[:24]}"
                            id_map[old_id] = new_id  # always overwrite
                            block["id"] = new_id

            # Anthropic format: tool results are inside user messages
            elif role == "user":
                for block in content:
                    if block.get("type") == "tool_result":
                        old_id = block.get("tool_use_id")
                        if old_id and old_id in id_map:
                            block["tool_use_id"] = id_map[old_id]

        return data


# Module-level instance so LiteLLM's get_instance_fn retrieves an instance,
# not the class itself (which would be treated as a callable function).
tool_id_remap_hook = ToolIdRemapHook()
