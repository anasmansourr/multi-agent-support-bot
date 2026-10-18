import os

import pytest

from src import config
from src.demo_check import LOW_CONFIDENCE_MESSAGE, ROUTING_CASES, Result, format_report, routing_checks
from src.routing.router import decide
from tests.test_routing import FakePredictor


class ScriptedRouter:
    """route_only() backed by a dict message -> (intent, confidence, known_ratio)."""

    def __init__(self, table):
        self.table = table

    def route_only(self, message):
        intent, conf, known = self.table[message]
        return decide(message, FakePredictor(raw={"intent": intent, "confidence": conf, "known_ratio": known}))


def good_table():
    table = {m: (agent if agent != "general" or reason != "unknown_words" else "general", 0.9, 1.0)
             for _, m, agent, reason in ROUTING_CASES}
    table["asdf qwerty"] = ("billing", 0.95, 0.0)                    # confident but unknown words
    table[LOW_CONFIDENCE_MESSAGE] = ("billing", 0.32, 0.5)
    return table


def test_all_routing_checks_pass_for_a_correct_router():
    results = routing_checks(ScriptedRouter(good_table()))
    assert len(results) == len(ROUTING_CASES) + 1 and all(r.ok for r in results), [r for r in results if not r.ok]


def test_a_misrouting_is_reported_as_a_failure():
    table = good_table()
    table["It says I owe 500"] = ("general", 0.9, 1.0)
    failed = [r for r in routing_checks(ScriptedRouter(table)) if not r.ok]
    assert [r.name for r in failed] == ["route: owes 500 (memory demo)"]


def test_an_invisible_low_confidence_demo_is_flagged():
    table = good_table()
    table[LOW_CONFIDENCE_MESSAGE] = ("general", 0.32, 0.5)           # predicted == general -> fallback shows nothing
    failed = [r.name for r in routing_checks(ScriptedRouter(table)) if not r.ok]
    assert any("VISIBLE" in n for n in failed)


def test_report_format():
    report = format_report([Result("a", True, "ok detail"), Result("b", False, "why", review="the answer")], "live")
    assert "[PASS] a" in report and "[FAIL] b" in report and "1/2 checks passed" in report and "the answer" in report


# Opt-in: needs your trained models/classifier.pt.   RUN_INTEGRATION=1 python -m pytest tests/test_demo_check.py
@pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1" or not config.CLASSIFIER_PATH.exists(),
                    reason="set RUN_INTEGRATION=1 and train the classifier first")
def test_real_classifier_passes_all_routing_checks():
    from src.routing.router import Router
    results = routing_checks(Router())
    assert all(r.ok for r in results), [(r.name, r.detail) for r in results if not r.ok]
