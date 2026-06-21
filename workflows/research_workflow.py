#!/usr/bin/env python3
"""Research workflow — clarify, research, report.

Usage:
    python workflows/research_workflow.py

Requires ANTHROPIC_API_KEY to be set in the environment.
Outputs: output/research_reports/<slug>_<date>.md  and  .pdf
"""

import datetime
import os
import re
import sys

import anthropic
from dotenv import load_dotenv
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

# ── palette (matches generate_report.py) ─────────────────────────────────────
C_PURPLE = colors.HexColor("#534AB7")
C_GRAY   = colors.HexColor("#5F5E5A")
C_BLACK  = colors.HexColor("#2C2C2A")
C_WHITE  = colors.white

W, H   = A4
MARGIN = 18 * mm
MODEL  = "claude-sonnet-4-6"

SECTION_KEYS = [
    "Executive Summary",
    "Key Findings",
    "Detailed Analysis",
    "Sources & Citations",
    "Limitations & Caveats",
]

SYSTEM_PROMPT = """\
You are an expert research analyst. Thoroughly research the given topic and produce \
a structured, factual report.

Use the web_search tool to retrieve current, authoritative sources. Cite sources \
inline as [Source: title or URL].

Structure your response with exactly these headings (use ## markdown):

## Executive Summary
## Key Findings
## Detailed Analysis
## Sources & Citations
## Limitations & Caveats

Use bullet points for lists. Deliver findings only — no meta-commentary about the research process.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — clarifying questions
# ─────────────────────────────────────────────────────────────────────────────
_QUESTIONS = [
    ("audience",   "1. Who is the target audience?",
     False, "e.g. expert researchers / executives / general public / students"),
    ("scope",      "2. How deep should the scope be?",
     False, "e.g. broad overview / focused subtopic / comprehensive deep-dive"),
    ("specifics",  "3. What specific questions or angles should be covered?",
     False, "Free text — be as specific as you like"),
    ("exclusions", "4. Any aspects to exclude?",
     True,  "Optional — press Enter to skip"),
    ("time_scope", "5. What time scope should be covered?",
     False, "e.g. all time / last 5 years / last 1 year"),
]


def _prompt(question: str, hint: str, optional: bool) -> str:
    suffix = " (press Enter to skip)" if optional else ""
    print(f"\n  [{hint}]")
    while True:
        answer = input(f"  {question}{suffix}\n  > ").strip()
        if answer or optional:
            return answer
        print("  Please provide an answer.")


def clarify(topic: str) -> dict:
    print(f"\n{'─' * 58}")
    print(f"  Topic: {topic}")
    print(f"{'─' * 58}")
    print("  A few quick questions before we start:\n")
    brief = {"topic": topic}
    for key, question, optional, hint in _QUESTIONS:
        brief[key] = _prompt(question, hint, optional)
    return brief


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — research via Claude API
# ─────────────────────────────────────────────────────────────────────────────
def _build_prompt(brief: dict) -> str:
    lines = [
        f"Research topic: {brief['topic']}",
        f"Target audience: {brief['audience']}",
        f"Scope: {brief['scope']}",
        f"Specific questions / angles: {brief['specifics']}",
        f"Time scope: {brief['time_scope']}",
    ]
    if brief.get("exclusions"):
        lines.append(f"Exclude: {brief['exclusions']}")
    lines.append(
        "\nResearch this thoroughly and return a structured report using the specified headings."
    )
    return "\n".join(lines)


def _with_web_search(client: anthropic.Anthropic, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    tools = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 15}]

    for _ in range(25):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )
        print(".", end="", flush=True)

        if resp.stop_reason == "end_turn":
            return "\n".join(b.text for b in resp.content if hasattr(b, "text"))

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            return "\n".join(b.text for b in resp.content if hasattr(b, "text"))

        messages.append({"role": "assistant", "content": resp.content})
        messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": b.id, "content": ""}
                for b in tool_uses
            ],
        })

    return "\n".join(b.text for b in resp.content if hasattr(b, "text"))


def _without_web_search(client: anthropic.Anthropic, user_prompt: str) -> str:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text


def research(brief: dict) -> str:
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit(
            "\nERROR: ANTHROPIC_API_KEY is not set.\n"
            "  Windows:  set ANTHROPIC_API_KEY=sk-ant-...\n"
            "  Mac/Linux: export ANTHROPIC_API_KEY=sk-ant-...\n"
        )

    client = anthropic.Anthropic(
        api_key=api_key,
        default_headers={"anthropic-beta": "web-search-2025-03-05"},
    )
    user_prompt = _build_prompt(brief)

    print("\n  Researching", end="", flush=True)
    try:
        text = _with_web_search(client, user_prompt)
    except Exception as exc:
        print(f"\n  Web search unavailable ({exc.__class__.__name__}: {exc}).")
        print("  Falling back to Claude knowledge only", end="", flush=True)
        client_plain = anthropic.Anthropic(api_key=api_key)
        text = _without_web_search(client_plain, user_prompt)
    print(" done.")
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — parse sections
# ─────────────────────────────────────────────────────────────────────────────
def parse_sections(raw: str) -> dict:
    pattern = re.compile(
        r"##\s+(" + "|".join(re.escape(k) for k in SECTION_KEYS) + r")\s*\n",
        re.IGNORECASE,
    )
    parts = pattern.split(raw)
    sections: dict = {}
    i = 1
    while i < len(parts) - 1:
        sections[parts[i].strip()] = parts[i + 1].strip()
        i += 2
    if not sections:
        sections["Detailed Analysis"] = raw.strip()
    return sections


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4a — Markdown output
# ─────────────────────────────────────────────────────────────────────────────
def save_markdown(brief: dict, sections: dict, path: str) -> None:
    today = datetime.date.today().isoformat()
    lines = [
        f"# Research Report: {brief['topic']}",
        f"_Generated: {today}_",
        "",
        "---",
        "",
        "## Research Brief",
        f"- **Topic:** {brief['topic']}",
        f"- **Audience:** {brief['audience']}",
        f"- **Scope:** {brief['scope']}",
        f"- **Specific questions:** {brief['specifics']}",
        f"- **Time scope:** {brief['time_scope']}",
    ]
    if brief.get("exclusions"):
        lines.append(f"- **Exclusions:** {brief['exclusions']}")
    lines += ["", "---", ""]

    for key in SECTION_KEYS:
        body = sections.get(key, "")
        if body:
            lines += [f"## {key}", "", body, ""]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4b — PDF output
# ─────────────────────────────────────────────────────────────────────────────
def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "RTitle", parent=base["Title"],
            fontSize=20, textColor=C_WHITE, spaceAfter=4,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "RSub", parent=base["Normal"],
            fontSize=10, textColor=C_WHITE, spaceAfter=0,
            fontName="Helvetica", alignment=TA_CENTER,
        ),
        "h2": ParagraphStyle(
            "RH2", parent=base["Heading2"],
            fontSize=12, textColor=C_PURPLE,
            fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "RBody", parent=base["Normal"],
            fontSize=9.5, textColor=C_BLACK, leading=14,
            fontName="Helvetica", spaceAfter=5,
        ),
    }


def _md_to_paras(text: str, style) -> list:
    paras = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("- ", "* ")):
            line = "• " + line[2:]
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        line = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", line)
        # escape bare ampersands not already escaped
        line = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;)", "&amp;", line)
        paras.append(Paragraph(line, style))
    return paras


def save_pdf(brief: dict, sections: dict, path: str) -> None:
    today = datetime.date.today().isoformat()
    st = _styles()
    story = []

    # title banner
    banner = Table(
        [
            [Paragraph("Research Report", st["title"])],
            [Paragraph(brief["topic"].title(), st["subtitle"])],
            [Paragraph(f"Generated {today}  ·  {MODEL}", st["subtitle"])],
        ],
        colWidths=[W - 2 * MARGIN],
    )
    banner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), C_PURPLE),
        ("ROWPADDING",    (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, 0),  18),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 18),
    ]))
    story += [banner, Spacer(1, 8 * mm)]

    # brief
    story.append(Paragraph("Research Brief", st["h2"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_PURPLE))
    story.append(Spacer(1, 3 * mm))
    brief_rows = [
        ("Topic", brief["topic"]),
        ("Audience", brief["audience"]),
        ("Scope", brief["scope"]),
        ("Specific questions", brief["specifics"]),
        ("Time scope", brief["time_scope"]),
    ]
    if brief.get("exclusions"):
        brief_rows.append(("Exclusions", brief["exclusions"]))
    for label, value in brief_rows:
        story.append(Paragraph(f"<b>{label}:</b>  {value}", st["body"]))
    story.append(Spacer(1, 4 * mm))

    # report sections
    for key in SECTION_KEYS:
        body = sections.get(key, "")
        if not body:
            continue
        story.append(Paragraph(key, st["h2"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=C_PURPLE))
        story.append(Spacer(1, 2 * mm))
        story.extend(_md_to_paras(body, st["body"]))
        story.append(Spacer(1, 4 * mm))

    SimpleDocTemplate(
        path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
    ).build(story)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def _slug(text: str, max_len: int = 40) -> str:
    text = re.sub(r"[^a-z0-9\s]", "", text.lower().strip())
    return re.sub(r"\s+", "_", text)[:max_len].rstrip("_")


def run() -> None:
    print("\n" + "=" * 58)
    print("  Research Workflow")
    print("=" * 58)
    topic = input("\nEnter research topic: ").strip()
    if not topic:
        sys.exit("No topic provided.")

    brief    = clarify(topic)
    raw_text = research(brief)
    sections = parse_sections(raw_text)

    today    = datetime.date.today().isoformat()
    out_dir  = os.path.join("output", "research_reports")
    os.makedirs(out_dir, exist_ok=True)
    base     = os.path.join(out_dir, f"{_slug(topic)}_{today}")

    print("\n  Saving outputs...")
    save_markdown(brief, sections, base + ".md")
    print(f"  Markdown -> {base}.md")
    save_pdf(brief, sections, base + ".pdf")
    print(f"  PDF      -> {base}.pdf")
    print("\n  Done.")


if __name__ == "__main__":
    run()
