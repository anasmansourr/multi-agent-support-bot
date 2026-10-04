import base64
import re

from src import config
from src.make_report import build_html, prompts_markdown, screenshots_html

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")


def test_all_23_required_sections_exist_in_order():
    text = (config.ROOT_DIR / "docs" / "writeup.md").read_text(encoding="utf-8")
    numbers = [int(n) for n in re.findall(r"^## (\d+)\.", text, flags=re.M)]
    assert numbers == list(range(1, 24))


def test_prompts_in_the_report_are_the_real_ones():
    md = prompts_markdown()
    for name in ["Billing Agent", "Technical Agent", "Feature Request Agent", "General Agent"]:
        assert f"### {name}" in md
    assert "check_invoice" in md and "documentation does not provide enough information" in md


def test_report_builds_with_placeholders_replaced(tmp_path):
    page, missing = build_html(screenshots_dir=tmp_path)
    assert "{{" not in page and "@@HTML" not in page
    assert "<h2>23. Reflection" in page and "Never mention tool names" in page
    assert "docs/screenshots/*" in missing                      # empty folder is reported, not crashed on


def test_images_are_embedded_and_missing_ones_flagged(tmp_path, monkeypatch):
    (tmp_path / "01_billing_tool_call.png").write_bytes(PNG)
    html, n = screenshots_html(tmp_path)
    assert n == 1 and "data:image/png;base64," in html and "Billing tool call" in html
    page, missing = build_html(screenshots_dir=tmp_path)
    assert "docs/screenshots/*" not in missing
    assert all(m.startswith("models/") for m in missing)         # only the figures that live on the author's laptop


def test_summary_numbers_are_present():
    page, _ = build_html()
    for fact in ["89.9%", "78.3%", "0.898", "336,148", "26/26", "176 automated tests"]:
        assert fact in page, fact
