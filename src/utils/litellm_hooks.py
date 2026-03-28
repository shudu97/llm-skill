"""
LiteLLM pre-call hook to remap tool call IDs and normalize tool result content.

Command-A (and similar enterprise models) generate sequential tool call IDs
(e.g. call_1, call_2) that restart from call_1 on every API request. When a
session is resumed, the conversation history already contains these IDs, so the
model's new call_1 collides with the historical call_1, causing a 400 error:
  "tool call 'id' is not unique"

Fix 1 (ID remapping): before forwarding each request to the enterprise gateway,
remap every tool_call ID in the history to a unique UUID. Both the assistant
message tool_calls[].id and the corresponding tool message tool_call_id are
updated together so the pairing stays consistent.

Fix 2 (tool_result content normalization): LiteLLM's Anthropic→OpenAI
conversion has two bugs in translate_anthropic_messages_to_openai:
  - tool_result with content=null is silently dropped (key exists but value is
    None, so the "content not in block" guard is False and no branch matches)
  - tool_result with content=[item1, item2, ...] creates one OpenAI tool
    message PER item; gateways then convert these back into multiple
    tool_result blocks, exceeding the number of tool_use blocks

We fix both by normalising tool_result content to a plain string before
LiteLLM runs the conversion.
"""

import logging
import uuid

from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


class ToolIdRemapHook(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        messages = data.get("messages")
        logger.warning(
            "ToolIdRemapHook called: call_type=%s model=%s num_messages=%s",
            call_type,
            data.get("model"),
            len(messages) if messages else 0,
        )
        if messages:
            for i, msg in enumerate(messages):
                role = msg.get("role")
                content = msg.get("content")
                if isinstance(content, list):
                    for j, block in enumerate(content):
                        btype = block.get("type")
                        if btype in ("tool_use", "tool_result"):
                            logger.warning(
                                "  msg[%d] role=%s block[%d] type=%s id=%s content_type=%s",
                                i, role, j, btype,
                                block.get("id") or block.get("tool_use_id"),
                                type(block.get("content")).__name__,
                            )
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
                        # Normalise content to a plain string so LiteLLM's
                        # Anthropic→OpenAI conversion always emits exactly one
                        # tool message per tool_result block.
                        block["content"] = self._normalise_tool_result_content(
                            block.get("content")
                        )

        # Remove orphaned tool_result blocks (those whose tool_use_id has no
        # matching tool_use in the immediately preceding assistant message).
        self._trim_excess_tool_results(messages)

        return data

    def _normalise_tool_result_content(self, content) -> str:
        """Collapse tool_result content to a plain string.

        LiteLLM's translate_anthropic_messages_to_openai has two bugs:
        - content=None: the key exists so the "not in block" guard is False,
          and isinstance(None, str/list) are both False → tool message dropped.
        - content=[item, item, ...]: each item becomes a separate OpenAI tool
          message, causing downstream gateways to produce extra tool_result
          blocks.
        Normalising to a string fixes both before the conversion runs.
        """
        if not content:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "\n".join(parts)
        return ""

    def _trim_excess_tool_results(self, messages: list) -> None:
        """Remove tool_result blocks that have no matching tool_use in the
        immediately preceding assistant message."""
        for i, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            tool_result_blocks = [b for b in content if b.get("type") == "tool_result"]
            if not tool_result_blocks:
                continue

            # Find the immediately preceding assistant message
            preceding_tool_use_ids: set[str] = set()
            for j in range(i - 1, -1, -1):
                prev = messages[j]
                if prev.get("role") == "assistant":
                    prev_content = prev.get("content")
                    if isinstance(prev_content, list):
                        preceding_tool_use_ids = {
                            b["id"]
                            for b in prev_content
                            if b.get("type") == "tool_use" and b.get("id")
                        }
                    break

            if not preceding_tool_use_ids:
                continue

            # Keep only tool_result blocks whose tool_use_id is in the
            # preceding assistant message; drop orphaned ones.
            valid_results = [
                b for b in tool_result_blocks
                if b.get("tool_use_id") in preceding_tool_use_ids
            ]
            if len(valid_results) < len(tool_result_blocks):
                other_blocks = [b for b in content if b.get("type") != "tool_result"]
                msg["content"] = other_blocks + valid_results


# Module-level instance so LiteLLM's get_instance_fn retrieves an instance,
# not the class itself (which would be treated as a callable function).
tool_id_remap_hook = ToolIdRemapHook()
