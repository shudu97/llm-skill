#!/usr/bin/env python3
"""
Convert an intermediate blocks file (with raw MCP responses) into the
compile.py-compatible blocks format.

Intermediate format — what Claude writes after calling mcp-echarts:
  [
    { "type": "text",  "content": "..." },
    { "type": "chart", "title": "optional", "mcp_output": "<JSON string from MCP>" },
    ...
  ]

Output format — what compile.py expects:
  [
    { "type": "text",  "content": "..." },
    { "type": "chart", "title": "optional", "options": { ...parsed ECharts option... } },
    ...
  ]

The only transformation is: for each chart block, parse "mcp_output" (string)
into "options" (object) and remove the "mcp_output" key.
"""

import argparse
import json
import sys
from pathlib import Path


def assemble(blocks: list[dict]) -> list[dict]:
    out = []
    for i, block in enumerate(blocks):
        btype = block.get("type")

        if btype == "text":
            if "content" not in block:
                sys.exit(f"Error: text block at index {i} is missing 'content'.")
            out.append({"type": "text", "content": block["content"]})

        elif btype == "chart":
            if "mcp_output" not in block and "options" not in block:
                sys.exit(
                    f"Error: chart block at index {i} is missing both "
                    f"'mcp_output' and 'options'."
                )

            if "mcp_output" in block:
                try:
                    options = json.loads(block["mcp_output"])
                except json.JSONDecodeError as e:
                    sys.exit(
                        f"Error: chart block at index {i} — "
                        f"could not parse 'mcp_output' as JSON: {e}"
                    )
            else:
                options = block["options"]

            chart_block: dict = {"type": "chart", "options": options}
            if block.get("title"):
                chart_block["title"] = block["title"]
            out.append(chart_block)

        else:
            sys.exit(f"Error: block at index {i} has unknown type '{btype}'.")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse raw MCP echarts responses in an intermediate blocks file\n"
            "and output a compile.py-compatible blocks JSON array.\n\n"
            "Intermediate chart block format:\n"
            '  {"type":"chart","title":"optional","mcp_output":"<JSON string from MCP>"}\n\n'
            "Output is written to --output or stdout for piping into compile.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", help="Path to intermediate blocks JSON file")
    parser.add_argument(
        "--output",
        help="Write assembled blocks to this file (default: stdout)",
    )
    args = parser.parse_args()

    if args.input:
        raw = Path(args.input).read_text(encoding="utf-8")
    else:
        raw = sys.stdin.read().strip()
        if not raw:
            parser.error("Provide --input or pipe the intermediate blocks JSON via stdin.")

    try:
        blocks = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.exit(f"Error: could not parse input as JSON: {e}")

    if not isinstance(blocks, list):
        sys.exit("Error: input must be a JSON array.")

    assembled = assemble(blocks)
    result = json.dumps(assembled, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(result + "\n", encoding="utf-8")
        print(f"Blocks written to: {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
