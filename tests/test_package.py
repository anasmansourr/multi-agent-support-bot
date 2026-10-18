import zipfile

import pytest

from src.package import build_zip, collect_files, scan_for_secrets, verify_zip

FAKE_KEY = "gsk_" + "A1b2C3d4E5" * 3          # built at runtime so this test file itself never contains a key


def make_project(root):
    def w(rel, text="x", binary=False):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text if binary else text.encode())
    for rel in ["README.md", "requirements.txt", ".env.example", "app.py", "docs/writeup.md", "src/config.py",
                "src/__init__.py", "tests/test_x.py", "models/training_curves.png",
                "src/routing/router.py", "src/classifier/model.py", "data/knowledge_base/refund_policy.md"]:
        w(rel)
    w(".env.example", "GROQ_API_KEY=your_key_here\n")
    w("models/classifier.pt", b"\x00\x01binary", binary=True)
    for i in range(10):
        w(f"data/knowledge_base/doc{i}.md", "# doc")
    # things that must never be packaged
    w(".env", "GROQ_API_KEY" + "=" + FAKE_KEY + "\n")
    w("venv/lib/x.py")
    w("src/__pycache__/m.pyc")
    w("models/wd0.pt")
    w("data/vectorstore/index.faiss")
    w("data/raw/banking77_labels.txt")
    w("data/feature_requests.jsonl")
    w(".git/config")
    w("old.zip")
    return root


def names(files):
    return {str(f).replace("\\", "/") for f in files}


def test_collect_files_keeps_the_right_things(tmp_path):
    got = names(collect_files(make_project(tmp_path)))
    assert {"README.md", "app.py", ".env.example", "models/classifier.pt", "models/training_curves.png",
            "data/knowledge_base/doc0.md", "tests/test_x.py"} <= got
    for banned in [".env", "venv/lib/x.py", "src/__pycache__/m.pyc", "models/wd0.pt", "data/vectorstore/index.faiss",
                   "data/raw/banking77_labels.txt", "data/feature_requests.jsonl", ".git/config", "old.zip"]:
        assert banned not in got, banned


def test_clean_project_builds_and_verifies(tmp_path):
    root = make_project(tmp_path / "proj")
    out = tmp_path / "out" / "s.zip"
    build_zip(root, out)
    problems, warnings = verify_zip(out)
    assert problems == []
    assert any("PDF" in w for w in warnings) and any("screenshots" in w for w in warnings)
    with zipfile.ZipFile(out) as z:
        assert all(n.startswith("multi_agent_support_bot/") for n in z.namelist())
        assert not any(n.endswith("/.env") for n in z.namelist())


def test_a_key_in_any_included_file_stops_the_build(tmp_path):
    root = make_project(tmp_path / "proj")
    (root / "docs" / "notes.md").write_text(f"my key is {FAKE_KEY}")
    findings = scan_for_secrets(root, collect_files(root))
    assert findings and findings[0][0] == "docs/notes.md"
    with pytest.raises(RuntimeError, match="Refusing to build"):
        build_zip(root, tmp_path / "o.zip")
    assert not (tmp_path / "o.zip").exists()


def test_real_env_value_in_a_text_file_is_caught_but_the_placeholder_is_not(tmp_path):
    root = make_project(tmp_path / "proj")
    (root / "docs" / "a.md").write_text("GROQ_API_KEY=your_key_here")
    assert scan_for_secrets(root, collect_files(root)) == []
    (root / "docs" / "b.md").write_text("GROQ_API_KEY" + "=" + "abcdefghijklmnop")
    assert [f for f, _ in scan_for_secrets(root, collect_files(root))] == ["docs/b.md"]


def test_verify_catches_a_hand_made_bad_zip(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("multi_agent_support_bot/README.md", "x")
        z.writestr("multi_agent_support_bot/.env", "GROQ_API_KEY" + "=" + "whatever123456")
        z.writestr("multi_agent_support_bot/venv/a.py", "x")
    problems, _ = verify_zip(bad)
    text = " | ".join(problems)
    assert "FORBIDDEN secret file" in text and "missing required file: models/classifier.pt" in text
    assert "should not be in ZIP" in text and "possible secret" in text and "knowledge-base" in text


def test_screenshots_and_pdf_remove_the_warnings(tmp_path):
    root = make_project(tmp_path / "proj")
    (root / "docs" / "screenshots").mkdir()
    (root / "docs" / "screenshots" / "01_a.png").write_bytes(b"png")
    (root / "docs" / "report.pdf").write_bytes(b"%PDF")
    out = tmp_path / "s.zip"
    build_zip(root, out)
    assert verify_zip(out) == ([], [])
