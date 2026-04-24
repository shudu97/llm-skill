---
name: compile-dashboard
description: >
  Generates a self-contained HTML dashboard report from an ordered sequence of text and chart blocks.
  Use whenever the user wants data findings visualized as ECharts charts combined with written analysis,
  in any combination and order.
---

# Generate HTML Dashboard Report

This skill is part of the **data-visualization** plugin. You have three scripts in `${CLAUDE_SKILL_DIR}/scripts/`:
- **assemble.py** — parses raw MCP text responses into a compile-ready blocks file
- **compile.py** — renders the blocks file into a standalone HTML report

---

## Workflow overview

```
mcp-echarts tools (outputType: "option")
        │
        │  returns pretty-printed JSON string per tool call
        ▼
assemble.py  ← intermediate blocks file (mcp_output fields)
        │
        │  parses each mcp_output string → options object
        ▼
compile.py   ← assembled blocks file (options fields)
        │
        ▼
  HTML report
```

---

## Step 1 — call mcp-echarts tools

| `outputType` | Returns |
|---|---|
| `"png"` *(default)* | Base64 PNG — **not usable here** |
| `"svg"` | SVG string — **not usable here** |
| `"option"` | Pretty-printed ECharts JSON string — **always use this** |

**Always pass `outputType: "option"`.**

| Tool | Best for |
|---|---|
| `mcp__echarts__generate_bar_chart` | Categorical comparisons, rankings |
| `mcp__echarts__generate_line_chart` | Trends over time |
| `mcp__echarts__generate_area_chart` | Cumulative or stacked trends |
| `mcp__echarts__generate_pie_chart` | Part-to-whole proportions |
| `mcp__echarts__generate_scatter_chart` | Correlation between two variables |
| `mcp__echarts__generate_heatmap_chart` | Two-dimensional density / matrix data |
| `mcp__echarts__generate_echarts` | Any chart via raw `echartsOption` JSON |

Call as many tools as needed. Each returns a JSON string — keep them.

---

## Step 2 — write the intermediate blocks file

Write a JSON array where chart blocks use `mcp_output` (the raw string from the tool result):

```json
[
  { "type": "text", "content": "Overall revenue declined 12% in Q1..." },
  { "type": "chart", "title": "Revenue by Segment", "mcp_output": "<JSON string from MCP tool>" },
  { "type": "text", "content": "Enterprise was the primary driver..." },
  { "type": "chart", "title": "Month-over-Month Change", "mcp_output": "<JSON string from MCP tool>" },
  { "type": "text", "content": "Recommended actions: ..." }
]
```

| Block type | Required fields | Optional fields |
|---|---|---|
| `"text"` | `content` (string) | — |
| `"chart"` | `mcp_output` (JSON string from MCP) | `title` (string) |

---

## Step 3 — assemble and compile

Run `assemble.py` to parse the MCP strings into option objects, then pipe into `compile.py`:

```bash
python ${CLAUDE_SKILL_DIR}/scripts/assemble.py \
  --input /tmp/raw_blocks.json \
  --output /tmp/blocks.json

python ${CLAUDE_SKILL_DIR}/scripts/compile.py \
  --title "Revenue Analysis — Q1 2025" \
  --blocks-file /tmp/blocks.json \
  --output data/reports/q1_revenue.html
```

Or pipe directly:

```bash
python ${CLAUDE_SKILL_DIR}/scripts/assemble.py --input /tmp/raw_blocks.json \
  | python ${CLAUDE_SKILL_DIR}/scripts/compile.py \
      --title "Revenue Analysis — Q1 2025" \
      --output data/reports/q1_revenue.html
```

The default chart theme is `tech-blue`. Override with `--theme <name>` (e.g. `macarons`, `vintage`, `dark`). Use `--theme default` for no theme.

The script prints the saved path on success. Report that path to the user.
