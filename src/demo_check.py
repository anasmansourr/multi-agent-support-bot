"""Integration check: runs every demo scenario against the REAL stack and prints PASS / FAIL.

    python -m src.demo_check --routing-only      # real classifier only; no LLM, no network; seconds
    python -m src.demo_check                     # everything real: classifier + docs index + Groq + Ollama
    python -m src.demo_check --skip-ollama       # skip the checks that need the local Ollama model

A report is written to models/integration_report.txt (use it in your write-up). LLM answers vary a little
between runs, so a single FAIL in the live part is worth re-running once before investigating.
Feature-request tickets created by this script go to a temporary file, not to data/feature_requests.jsonl.
"""
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src import config


@dataclass
class Result:
    name: str
    ok: bool
    detail: str = ""
    review: str = ""            # text a human should read (e.g. an anti-hallucination answer)


# ------------------------------------------------------------------ routing checks (no LLM)
ROUTING_CASES = [  # (label, message, expected agent, expected reason or None)
    ("billing problem", "I have a billing problem.", "billing", "ok"),
    ("owes 500 (memory demo)", "It says I owe 500", "billing", "ok"),
    ("technical: invoice page (handoff turn 2)", "the invoice page will not load", "technical", "ok"),
    ("technical: upload crash", "the app crashes when I upload a file", "technical", "ok"),
    ("feature request: new", "please add a linux app", "feature_request", "ok"),
    ("feature request: on roadmap", "please add dark mode", "feature_request", "ok"),
    ("general: greeting", "hello", "general", "ok"),
    ("general: support hours", "What are your support hours?", "general", "ok"),
    ("general: data regions", "Which data regions can I choose?", "general", "ok"),
    ("unknown input", "asdf qwerty", "general", "unknown_words"),
]
LOW_CONFIDENCE_MESSAGE = "uploading a photo gives me a red banner saying failed"


def routing_checks(router):
    results = []
    for label, message, agent, reason in ROUTING_CASES:
        d = router.route_only(message)
        ok = d.selected_agent == agent and (reason is None or d.reason == reason)
        results.append(Result(f"route: {label}", ok,
                              f"{d.predicted_intent} ({d.confidence:.3f}) -> {d.selected_agent} [{d.reason}]; expected {agent} [{reason}]"))
    d = router.route_only(LOW_CONFIDENCE_MESSAGE)
    ok = d.reason == "low_confidence" and d.selected_agent == "general" and d.predicted_intent != "general" and d.fallback_applied
    results.append(Result("route: low-confidence fallback is VISIBLE (predicted != general, sent to general)", ok,
                          f"{d.predicted_intent} ({d.confidence:.3f}) -> {d.selected_agent} [{d.reason}]"))
    return results


# ------------------------------------------------------------------ live checks (real LLMs)
def _tools(result):
    return [c["tool"] for c in result.agent.tool_calls] if result.agent else []


def _text(result):
    return (result.agent.output if result.agent else str(result.error)).lower()


def _fail_if_no_agent(name, result):
    return None if result.agent is not None else Result(name, False, f"no answer: {result.error}")


def live_checks(router, skip_ollama=False):
    from src.bot import SupportBot
    results = []
    config.FEATURE_REQUESTS_PATH = Path(tempfile.mkdtemp()) / "feature_requests.jsonl"     # do not touch the real file

    def fresh():
        return SupportBot(router=router)

    def add(name, ok, detail="", review=""):
        results.append(Result(name, bool(ok), detail, review))

    # 1. billing tool call is REAL and the answer uses its result
    r = fresh().ask("It says I owe 500")
    add("billing: check_invoice really called, answer mentions 500",
        r.agent and "check_invoice" in _tools(r) and "500" in _text(r), f"tools={_tools(r)}", _text(r))

    # 2. refund tool
    r = fresh().ask("Can I get a refund for my last payment?")
    add("billing: refund eligibility tool called", r.agent and "check_refund_eligibility" in _tools(r),
        f"agent={r.routing.selected_agent} tools={_tools(r)}", _text(r))

    # 3. technical
    r = fresh().ask("the invoice page will not load")
    add("technical agent uses a tool (service status or docs)",
        r.agent and r.routing.selected_agent == "technical" and bool(set(_tools(r)) & {"check_service_status", "search_docs"}),
        f"agent={r.routing.selected_agent} tools={_tools(r)}", _text(r))

    # 4-5. feature requests
    r = fresh().ask("please add a linux app")
    add("feature request: roadmap checked, then ticket logged (new feature)",
        r.agent and _tools(r)[:1] == ["check_roadmap"] and "log_feature_request" in _tools(r),
        f"tools={_tools(r)}", _text(r))
    r = fresh().ask("please add dark mode")
    add("feature request: already planned -> roadmap checked, NO ticket",
        r.agent and "check_roadmap" in _tools(r) and "log_feature_request" not in _tools(r),
        f"tools={_tools(r)}", _text(r))

    # 6-7. RAG
    r = fresh().ask("What are your support hours?")
    add("RAG: general agent searches the docs, answer has the real hours",
        r.agent and "search_docs" in _tools(r) and ("09:00" in _text(r) or "9:00" in _text(r)) and "18:00" in _text(r),
        f"tools={_tools(r)}", _text(r))
    r = fresh().ask("Which data regions can I choose?")
    add("RAG: answer says US and EU", r.agent and "us" in _text(r) and "eu" in _text(r), f"tools={_tools(r)}", _text(r))

    # 8. anti-hallucination (needs a human look)
    r = fresh().ask("Can I run FlowDesk on my own server?")
    t = _text(r)
    add("anti-hallucination: says the docs do not cover self-hosting (READ THE ANSWER)",
        r.agent and any(p in t for p in ("documentation", "not sure", "don't have", "do not have", "can't confirm", "cannot confirm"))
        and not any(p in t for p in ("there isn't an option", "there is no option", "not possible")),
        f"tools={_tools(r)}", t)

    # 9. low confidence, end to end
    r = fresh().ask(LOW_CONFIDENCE_MESSAGE)
    add("low confidence: routed to the general agent and still answered from the docs",
        r.agent and r.routing.reason == "low_confidence" and r.routing.selected_agent == "general" and "search_docs" in _tools(r),
        f"{r.routing.predicted_intent} ({r.routing.confidence:.3f}) -> {r.routing.selected_agent}", _text(r))

    # 10. unknown words
    r = fresh().ask("asdf qwerty")
    add("unknown input: handled by the general agent, no crash", r.agent is not None and r.routing.reason == "unknown_words",
        f"reason={r.routing.reason}", _text(r))

    # 11. memory within one agent
    bot = fresh()
    first = bot.ask("I have a billing problem.")
    second = bot.ask("It says I owe 500")
    add("memory: 'it' is understood in turn 2 (invoice looked up, 500 mentioned) and turn 1 asked a question",
        first.agent and second.agent and not first.agent.tool_calls and "check_invoice" in _tools(second) and "500" in _text(second),
        f"turn1 tools={_tools(first)} turn2 tools={_tools(second)}", f"T1: {_text(first)}\nT2: {_text(second)}")

    # 12. handoff keeps the context
    bot = fresh()
    a = bot.ask("It says I owe 500")
    b = bot.ask("the invoice page will not load")
    ok = (a.agent and b.agent and a.routing.selected_agent == "billing" and b.routing.selected_agent == "technical"
          and [t.agent for t in bot.memory.turns] == ["billing", "technical"])
    add("handoff: billing -> technical, one shared memory with both turns", ok,
        f"agents={[t.agent for t in bot.memory.turns]}", _text(b) if b.agent else "")

    # 13. Groq outage -> Ollama
    if not skip_ollama:
        r = fresh().ask("It says I owe 500", simulate_outage={"groq"})
        add("fallback: simulated Groq outage -> Ollama answers AND still calls the tool",
            r.agent and r.agent.provider == "ollama" and r.agent.used_fallback and "check_invoice" in _tools(r),
            f"provider={r.agent.provider if r.agent else None} tools={_tools(r)}", _text(r))

    # 14. both providers down -> safe message
    r = fresh().ask("It says I owe 500", simulate_outage={"groq", "ollama"})
    add("both providers down: friendly message, no traceback, routing still known",
        r.agent is None and r.error and "try again" in r.error.lower() and "traceback" not in r.error.lower()
        and r.routing.selected_agent == "billing", f"error={r.error!r}")

    # 15. empty input
    r = fresh().ask("   ")
    add("empty input rejected politely", r.agent is None and r.error == "Please type a message.")
    return results


# ------------------------------------------------------------------ output
def format_report(results, mode):
    lines = [f"Integration report - {datetime.now():%Y-%m-%d %H:%M} - mode: {mode}", ""]
    for r in results:
        lines.append(f"[{'PASS' if r.ok else 'FAIL'}] {r.name}")
        if r.detail:
            lines.append(f"        {r.detail}")
        if r.review and (not r.ok or "READ" in r.name or "anti-hallucination" in r.name):
            lines.append("        answer: " + r.review.replace("\n", "\n                "))
    passed = sum(r.ok for r in results)
    lines += ["", f"{passed}/{len(results)} checks passed"]
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-only", action="store_true", help="real classifier only, no LLM calls")
    parser.add_argument("--skip-ollama", action="store_true")
    args = parser.parse_args()

    from src.routing.router import Router
    try:
        router = Router()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    results = routing_checks(router)
    if not args.routing_only:
        results += live_checks(router, skip_ollama=args.skip_ollama)
    report = format_report(results, "routing-only" if args.routing_only else "live")
    print(report)
    (config.MODELS_DIR / "integration_report.txt").write_text(report + "\n", encoding="utf-8")
    raise SystemExit(0 if all(r.ok for r in results) else 1)


if __name__ == "__main__":
    main()
