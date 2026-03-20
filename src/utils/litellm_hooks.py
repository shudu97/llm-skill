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


class ToolIdRemapHook:
    def __init__(self, *args, **kwargs):
        pass

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        messages = data.get("messages")
        if not messages:
            return data

        id_map: dict[str, str] = {}

        for msg in messages:
            role = msg.get("role")

            # Remap IDs in assistant messages that have tool_calls
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    old_id = tc.get("id")
                    if old_id and old_id not in id_map:
                        id_map[old_id] = f"call_{uuid.uuid4().hex[:24]}"
                    if old_id:
                        tc["id"] = id_map[old_id]

            # Remap matching tool_call_id in tool result messages
            if role == "tool":
                old_id = msg.get("tool_call_id")
                if old_id and old_id in id_map:
                    msg["tool_call_id"] = id_map[old_id]

        return data
