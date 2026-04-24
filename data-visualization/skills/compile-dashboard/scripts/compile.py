#!/usr/bin/env python3
"""
Generate a self-contained HTML report from an ordered sequence of text and chart blocks.

Block schema (JSON array):
  [
    { "type": "text",  "content": "..." },
    { "type": "chart", "options": { ...ECharts option... }, "title": "optional" },
    ...
  ]

Blocks are rendered in the order they appear — mix and interleave text and charts freely.
"""

import argparse
import html
import json
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_THEME = "tech-blue"
_THEME_CDN = "https://cdn.jsdelivr.net/npm/echarts@5/theme/{theme}.js"


# ── HTML fragments ─────────────────────────────────────────────────────────────

_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>REPORT_TITLE</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
THEME_SCRIPT_TAG  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #e8f0f7;
      color: #222;
      padding: 36px 24px;
      min-height: 100vh;
    }
    .report {
      max-width: 1100px;
      margin: 0 auto;
      background: #fff;
      border-radius: 14px;
      box-shadow: 0 4px 24px rgba(0,0,0,.09);
      overflow: hidden;
    }
    .report-header {
      background: linear-gradient(135deg, #003a5d 0%, #007bb6 100%);
      color: #fff;
      padding: 32px 40px;
    }
    .report-header h1 { font-size: 1.55rem; font-weight: 700; letter-spacing: -.01em; }
    .report-header .meta { font-size: 0.82rem; opacity: .6; margin-top: 8px; }
    .block { padding: 28px 40px; }
    .block + .block { border-top: 1px solid #dde8f0; }
    .block-chart-title { font-size: 1rem; font-weight: 600; color: #007bb6; margin-bottom: 14px; }
    .chart-container { width: 100%; height: 420px; }
    .block-text { font-size: 0.95rem; line-height: 1.8; color: #444; white-space: pre-wrap; }
  </style>
</head>
<body>
  <div class="report">
"""

_HEADER_BLOCK = """    <div class="report-header">
      <h1>HEADER_TITLE</h1>
      <div class="meta">Generated TIMESTAMP</div>
    </div>
"""

_FOOTER = """  </div>
  <script>
    (function () {
      var charts = CHART_INSTANCES;
      charts.forEach(function (item) {
        var el = document.getElementById(item.id);
        if (!el) return;
        var c = echarts.init(el, THEME_NAME_JS);
        c.setOption(item.option);
        window.addEventListener('resize', function () { c.resize(); });
      });
    })();
  </script>
</body>
</html>
"""


# ── Renderer ───────────────────────────────────────────────────────────────────

def _render(title: str, blocks: list[dict], theme: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Theme script tag — empty for "default", CDN script for everything else
    if theme and theme != "default":
        theme_script_tag = f'  <script src="{_THEME_CDN.format(theme=theme)}"></script>\n'
        theme_name_js = f"'{theme}'"
    else:
        theme_script_tag = ""
        theme_name_js = "null"

    parts: list[str] = []
    chart_instances: list[dict] = []
    chart_idx = 0

    parts.append(
        _HEAD
        .replace("REPORT_TITLE", html.escape(title))
        .replace("THEME_SCRIPT_TAG", theme_script_tag)
    )
    parts.append(
        _HEADER_BLOCK
        .replace("HEADER_TITLE", html.escape(title))
        .replace("TIMESTAMP", timestamp)
    )

    for block in blocks:
        btype = block.get("type")

        if btype == "chart":
            chart_id = f"echarts-chart-{chart_idx}"
            chart_idx += 1
            chart_title = block.get("title", "")
            title_html = (
                f'      <div class="block-chart-title">{html.escape(chart_title)}</div>\n'
                if chart_title else ""
            )
            parts.append(
                f'    <div class="block">\n'
                f'{title_html}'
                f'      <div id="{chart_id}" class="chart-container"></div>\n'
                f'    </div>\n'
            )
            chart_instances.append({"id": chart_id, "option": block["options"]})

        elif btype == "text":
            content = html.escape(block.get("content", ""))
            parts.append(
                f'    <div class="block">\n'
                f'      <div class="block-text">{content}</div>\n'
                f'    </div>\n'
            )

    parts.append(
        _FOOTER
        .replace("CHART_INSTANCES", json.dumps(chart_instances, ensure_ascii=False))
        .replace("THEME_NAME_JS", theme_name_js)
    )

    return "".join(parts)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an HTML report from an ordered list of text and chart blocks.\n\n"
            "Block format (JSON array):\n"
            '  [{"type":"text","content":"..."},\n'
            '   {"type":"chart","title":"optional","options":{...ECharts option...}}]'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--title", default="Analysis Report", help="Report title shown in the header")
    parser.add_argument(
        "--theme",
        default=DEFAULT_THEME,
        help=f"ECharts theme name (default: {DEFAULT_THEME}). Use 'default' for no theme.",
    )
    parser.add_argument("--blocks", help="Blocks as a JSON array string")
    parser.add_argument("--blocks-file", help="Path to a JSON file containing the blocks array")
    parser.add_argument("--output", required=True, help="Destination path for the HTML file")
    args = parser.parse_args()

    if args.blocks_file:
        blocks = json.loads(Path(args.blocks_file).read_text(encoding="utf-8"))
    elif args.blocks:
        blocks = json.loads(args.blocks)
    else:
        data = sys.stdin.read().strip()
        if not data:
            parser.error("Provide --blocks, --blocks-file, or pipe the JSON array via stdin.")
        blocks = json.loads(data)

    if not isinstance(blocks, list):
        sys.exit("Error: blocks must be a JSON array.")

    html_content = _render(args.title, blocks, args.theme)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_content, encoding="utf-8")
    print(f"Report saved: {args.output}")


if __name__ == "__main__":
    main()
