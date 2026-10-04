"""Build the write-up as one self-contained HTML file you can print to PDF.

    python -m src.make_report            # writes docs/report.html
Then open it in a browser and use  Print -> Save as PDF.

It takes docs/writeup.md and fills three kinds of placeholders:
  {{FIGURE:path|caption}}  embeds an image (your models/*.png) - a visible warning box if the file is missing
  {{PROMPTS}}              the real system prompts, generated from src/agents/prompts.py
  {{SCREENSHOTS}}          every image in docs/screenshots/ in file-name order (name them 01_..., 02_...)
"""
import base64
import html
import mimetypes
import re
from pathlib import Path

import markdown

from src import config

DOCS = config.ROOT_DIR / "docs"
CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1b1b1b; max-width: 900px; margin: 0 auto; padding: 0 12px; }
h1 { font-size: 20pt; border-bottom: 2px solid #2a78d6; padding-bottom: 6px; }
h2 { font-size: 14pt; margin-top: 22px; color: #184f95; border-bottom: 1px solid #d0d7e2; padding-bottom: 3px; page-break-after: avoid; }
h3 { font-size: 11.5pt; page-break-after: avoid; }
table { border-collapse: collapse; margin: 8px 0; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #c9d1dc; padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: #eef3fa; }
pre, code { font-family: Consolas, monospace; font-size: 8.8pt; }
pre { background: #f5f7fa; border: 1px solid #dde3ec; padding: 8px 10px; white-space: pre-wrap; page-break-inside: avoid; }
blockquote { border-left: 4px solid #eb6834; margin: 10px 0; padding: 4px 12px; background: #fff7f2; }
figure { margin: 12px 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; border: 1px solid #dde3ec; }
figcaption { font-size: 9pt; color: #555; }
.missing { border: 2px dashed #d03b3b; color: #d03b3b; padding: 10px; margin: 10px 0; font-weight: 600; }
"""


def _figure(path, caption, base=None):
    """<figure> with the image embedded as base64, or a visible warning box if the file does not exist."""
    file = (base or config.ROOT_DIR) / path
    if not file.exists():
        return f'<div class="missing">Missing image: {html.escape(str(path))} ({html.escape(caption)})</div>', [str(path)]
    mime = mimetypes.guess_type(file.name)[0] or "image/png"
    data = base64.b64encode(file.read_bytes()).decode()
    return (f'<figure><img src="data:{mime};base64,{data}" alt="{html.escape(caption)}">'
            f'<figcaption>{html.escape(caption)}</figcaption></figure>'), []


def prompts_markdown():
    from src.agents.prompts import build_system_prompt
    from src.agents.tools import TOOLS_BY_AGENT
    from src.ui_helpers import AGENT_NAMES
    parts = []
    for intent in config.INTENTS:
        prompt = build_system_prompt(intent, [t.name for t in TOOLS_BY_AGENT[intent]])
        parts.append(f"### {AGENT_NAMES[intent]}\n\n```\n{prompt}\n```")
    return "\n\n".join(parts)


def screenshots_html(folder):
    images = sorted(p for p in folder.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}) if folder.exists() else []
    if not images:
        return '<div class="missing">No screenshots yet: put PNG files in docs/screenshots/ (01_billing.png, 02_handoff.png, ...) and rebuild.</div>', 0
    out = []
    for p in images:
        caption = re.sub(r"^\d+[_-]*", "", p.stem).replace("_", " ").replace("-", " ").strip().capitalize()
        out.append(_figure(p.name, caption, base=folder)[0])
    return "\n".join(out), len(images)


def build_html(source=None, screenshots_dir=None):
    """Returns (html_text, list_of_missing_figures)."""
    text = (source or DOCS / "writeup.md").read_text(encoding="utf-8")
    missing, figures = [], {}

    def stash(html_fragment):
        key = f"@@HTML{len(figures)}@@"
        figures[key] = html_fragment
        return key

    def figure_sub(m):
        fragment, miss = _figure(m.group(1), m.group(2))
        missing.extend(miss)
        return stash(fragment)

    text = re.sub(r"\{\{FIGURE:([^|}]+)\|([^}]*)\}\}", figure_sub, text)
    shots_html, n_shots = screenshots_html(screenshots_dir or DOCS / "screenshots")
    if n_shots == 0:
        missing.append("docs/screenshots/*")
    text = text.replace("{{SCREENSHOTS}}", stash(shots_html))
    text = text.replace("{{PROMPTS}}", prompts_markdown())
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    for key, fragment in figures.items():
        body = body.replace(f"<p>{key}</p>", fragment).replace(key, fragment)
    page = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>FlowDesk multi-agent support chatbot - write-up</title><style>{CSS}</style></head><body>{body}</body></html>'
    return page, missing


def main():
    page, missing = build_html()
    out = DOCS / "report.html"
    out.write_text(page, encoding="utf-8")
    print(f"Wrote {out} ({len(page) // 1024} KB). Open it in a browser, then Print -> Save as PDF.")
    if missing:
        print("\nStill missing (shown as red boxes in the report):")
        for m in missing:
            print("  -", m)
    else:
        print("All figures and screenshots found.")


if __name__ == "__main__":
    main()
