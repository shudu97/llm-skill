#!/usr/bin/env python3
"""
Stop hook for the data-visualization subagent.
Validates the ECharts option JSON in <echarts-option> tags before the agent stops.
Exits non-zero with an error message to trigger a retry if validation fails.

Can also be imported as a module to use validate_echarts_hook() as a Claude Agent
SDK SubagentStop callback — pass it via ClaudeAgentOptions hooks.
"""
import json
import os
import re
import sys
from datetime import datetime

_LOG = os.path.join(os.environ.get("CLAUDE_PROJECT_DIR", "."), "data", "validate_echarts.log")


def log(msg: str) -> None:
    with open(_LOG, "a") as f:
        f.write(f"{datetime.now().isoformat()} {msg}\n")


def validate(option_str: str) -> tuple[bool, str | None]:
    try:
        option = json.loads(option_str)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"

    if not isinstance(option, dict):
        return False, "ECharts option must be a JSON object"

    series = option.get("series", [])
    if isinstance(series, list):
        cartesian_types = {"bar", "line", "scatter"}
        has_cartesian = any(
            isinstance(s, dict) and s.get("type") in cartesian_types
            for s in series
        )
        if has_cartesian and not option.get("xAxis") and not option.get("yAxis"):
            return False, "Cartesian chart (bar/line/scatter) is missing xAxis or yAxis"

    return True, None


def main():
    payload = json.load(sys.stdin)

    if payload.get("stop_hook_active"):
        sys.exit(0)

    transcript = payload.get("transcript", [])

    text = ""
    for msg in reversed(transcript):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, list):
                text = " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            else:
                text = str(content)
            break

    match = re.search(r"<echarts-option>\s*(.*?)\s*</echarts-option>", text, re.DOTALL)
    if not match:
        sys.exit(0)

    valid, error = validate(match.group(1))
    if not valid:
        log(f"FAIL {error}")
        print(f"ECharts validation failed: {error}. Fix the option and output it again in <echarts-option> tags.")
        sys.exit(1)

    log("PASS")
    sys.exit(0)


# ── Claude Agent SDK hook (imported as a Python callback) ─────────────────────

def _extract_text(messages: list) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, list):
                return " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            return str(content)
    return ""


async def validate_echarts_hook(hook_input, tool_use_id, context):
    """Claude Agent SDK SubagentStop callback.

    Use with ClaudeAgentOptions:
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
        from .claude.hooks.validate_echarts import validate_echarts_hook

        options = ClaudeAgentOptions(
            hooks={
                "SubagentStop": [
                    HookMatcher(
                        matcher="data-visualization",
                        hooks=[validate_echarts_hook],
                    )
                ]
            }
        )
    """
    if hook_input.get("stop_hook_active"):
        return {}

    with open(hook_input["agent_transcript_path"]) as f:
        messages = [json.loads(line) for line in f]

    text = _extract_text(messages)

    match = re.search(r"<echarts-option>\s*(.*?)\s*</echarts-option>", text, re.DOTALL)
    if not match:
        return {}

    valid, error = validate(match.group(1))
    if not valid:
        log(f"FAIL (SDK) {error}")
        return {
            "continue_": False,
            "reason": f"ECharts validation failed: {error}. Fix the option and output it again in <echarts-option> tags.",
        }

    log("PASS (SDK)")
    return {}


if __name__ == "__main__":
    main()
