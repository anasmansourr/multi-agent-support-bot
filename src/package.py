"""Build the submission ZIP safely.

    python -m src.package            # scan for secrets, build submission/multi_agent_support_bot.zip, verify it
    python -m src.package --smoke    # also unpack the ZIP in a temp folder and run the tests + classifier from there

What goes in: source, tests, docs, the knowledge base, the processed data, and the trained model + its result files.
What never goes in: .env, venv, caches, .git, experiment checkpoints (models/wd*.pt), the rebuildable vector index,
downloaded raw data, runtime files (feature_requests.jsonl) and old ZIPs.
The build REFUSES to continue if any file looks like it contains an API key.
"""
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from src import config

TOP = "multi_agent_support_bot"
EXCLUDED_DIRS = {"venv", ".venv", "env", "__pycache__", ".git", ".pytest_cache", ".mypy_cache", "submission",
                 "node_modules", "vectorstore", "raw"}
EXCLUDED_FILES = {"feature_requests.jsonl", ".DS_Store"}
SECRET_PATTERNS = {
    "Groq-style key": re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{30,}"),
    "OpenAI-style key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "GROQ_API_KEY with a real value": re.compile(r"GROQ_API_KEY\s*=\s*(?!your_key_here)(?!\s)[^\s#\"']{8,}"),
}
REQUIRED = ["README.md", "requirements.txt", ".env.example", "app.py", "models/classifier.pt", "docs/writeup.md",
            "src/config.py", "src/routing/router.py", "src/classifier/model.py", "data/knowledge_base/refund_policy.md"]


def is_excluded(rel):
    parts = rel.parts
    name = rel.name
    if any(p in EXCLUDED_DIRS for p in parts[:-1]):
        return not (parts[-1] == ".gitkeep")            # keep nothing from excluded dirs, even .gitkeep
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return True
    if name in EXCLUDED_FILES or rel.suffix in {".pyc", ".zip", ".log"}:
        return True
    if rel.match("models/wd*.pt"):
        return True
    return False


def collect_files(root):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for f in filenames:
            rel = (Path(dirpath) / f).relative_to(root)
            if not is_excluded(rel):
                files.append(rel)
    return sorted(files)


def scan_text(name, data):
    text = data.decode("latin-1", errors="ignore")
    return [(name, label) for label, rx in SECRET_PATTERNS.items() if rx.search(text)]


def scan_for_secrets(root, files):
    findings = []
    for rel in files:
        findings += scan_text(str(rel).replace("\\", "/"), (root / rel).read_bytes())
    return findings


def build_zip(root, out_path):
    files = collect_files(root)
    findings = scan_for_secrets(root, files)
    if findings:
        raise RuntimeError("Refusing to build: possible secrets found:\n" + "\n".join(f"  {f}: {l}" for f, l in findings))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in files:
            z.write(root / rel, f"{TOP}/{str(rel).replace(chr(92), '/')}")
    return files


def verify_zip(zip_path, required=None):
    """Independent check of the finished ZIP. Returns (problems, warnings)."""
    problems, warnings = [], []
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        rels = [n.split("/", 1)[1] if "/" in n else n for n in names]
        for req in (REQUIRED if required is None else required):
            if req not in rels:
                problems.append(f"missing required file: {req}")
        for n in rels:
            p = Path(n)
            if p.name == ".env" or (p.name.startswith(".env.") and p.name != ".env.example"):
                problems.append(f"FORBIDDEN secret file in ZIP: {n}")
            if any(part in {"venv", ".venv", "__pycache__", ".git"} for part in p.parts):
                problems.append(f"should not be in ZIP: {n}")
        for n in names:
            for name, label in scan_text(n, z.read(n)):
                problems.append(f"possible secret ({label}) in {name}")
        if sum(r.startswith("data/knowledge_base/") and r.endswith(".md") for r in rels) < 10:
            problems.append("fewer than 10 knowledge-base documents")
        if not any(r.startswith("docs/") and r.lower().endswith(".pdf") for r in rels):
            warnings.append("no PDF write-up in docs/ yet (print docs/report.html to PDF and save it there)")
        if not any(r.startswith("docs/screenshots/") and r.lower().endswith((".png", ".jpg", ".jpeg")) for r in rels):
            warnings.append("no screenshots in docs/screenshots/")
    return problems, warnings


def smoke_test(zip_path):
    """Unpack in a temp folder and run the tests and the classifier from there (uses the current Python environment)."""
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        cwd = Path(tmp) / TOP
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        for label, cmd in [("pytest on the unpacked ZIP", [sys.executable, "-m", "pytest", "tests", "-q"]),
                           ("classifier loads from the ZIP's classifier.pt",
                            [sys.executable, "-m", "src.classifier.inference", "I was charged twice for my subscription"])]:
            r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
            tail = (r.stdout.strip().splitlines() or r.stderr.strip().splitlines() or [""])[-1]
            results.append((label, r.returncode == 0, tail))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="unpack the ZIP and run tests + classifier from it")
    args = parser.parse_args()
    root = config.ROOT_DIR
    out = root / "submission" / f"{TOP}.zip"

    print("1/3 Scanning for secrets and building the ZIP...")
    try:
        files = build_zip(root, out)
    except RuntimeError as exc:
        raise SystemExit(f"\nSTOP.\n{exc}\nRemove the secret, then run this again.")
    print(f"    {len(files)} files, {out.stat().st_size / 1_048_576:.1f} MB -> {out}")

    print("2/3 Verifying the ZIP...")
    problems, warnings = verify_zip(out)
    for w in warnings:
        print("    WARNING:", w)
    for p in problems:
        print("    PROBLEM:", p)
    ok = not problems
    if ok:
        print("    ZIP verified: required files present, no .env, no venv, no secrets found.")

    if args.smoke and ok:
        print("3/3 Smoke test (unpack in a temp folder, run tests + classifier)... this takes about a minute")
        for label, passed, tail in smoke_test(out):
            print(f"    [{'PASS' if passed else 'FAIL'}] {label}: {tail}")
            ok = ok and passed
    elif not args.smoke:
        print("3/3 Smoke test skipped (add --smoke to run it).")
    print("\nREADY TO SUBMIT" + (" (resolve the warnings above first if you can)" if warnings else "") if ok else "\nNOT READY: fix the problems above")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
