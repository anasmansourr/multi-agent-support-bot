import json

import pytest

from src import config
from src.agents import tools
from src.agents.tools import (TOOLS_BY_AGENT, check_invoice_tool, check_refund_eligibility_tool, check_roadmap_tool,
                              check_service_status_tool, log_feature_request_tool)


def test_check_invoice_returns_the_demo_balance():
    r = check_invoice_tool.invoke({"customer_id": "demo_customer"})
    assert r["found"] and r["amount_due_usd"] == 500.00 and r["invoice_status"] == "overdue"
    assert r["due_date"] == "2026-09-15" and r["last_payment_date"] == "2026-08-16"


def test_customer_id_is_normalised_and_unknown_is_not_an_error():
    assert check_invoice_tool.invoke({"customer_id": "  DEMO_Customer "})["found"]
    r = check_invoice_tool.invoke({"customer_id": "nobody"})
    assert r["found"] is False and "customer id" in r["message"]


def test_refund_window():
    assert check_refund_eligibility_tool.invoke({"customer_id": "cust_recent"})["eligible"] is True
    r = check_refund_eligibility_tool.invoke({"customer_id": "demo_customer"})
    assert r["eligible"] is False and "outside" in r["reason"]


def test_service_status():
    assert check_service_status_tool.invoke({"service": "webhooks"})["status"] == "partial_outage"
    assert check_service_status_tool.invoke({"service": "Mobile App"})["status"] == "operational"
    assert len(check_service_status_tool.invoke({})["services"]) == 5          # default 'all'
    assert "webhooks" in check_service_status_tool.invoke({"service": "toaster"})["valid_services"]


def test_roadmap_search():
    r = check_roadmap_tool.invoke({"keyword": "dark mode"})
    assert [m["feature"] for m in r["matches"]] == ["Dark mode"]        # not "Offline mode"
    assert r["matches"][0]["status"] == "planned"
    assert check_roadmap_tool.invoke({"keyword": "okta"})["matches"][0]["feature"] == "SAML single sign-on"
    assert check_roadmap_tool.invoke({"keyword": "teleportation"})["found"] is False


def test_log_feature_request_really_writes_and_numbers_tickets(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FEATURE_REQUESTS_PATH", tmp_path / "fr.jsonl")
    a = log_feature_request_tool.invoke({"title": "Linux app", "customer_id": "demo_customer"})
    b = log_feature_request_tool.invoke({"title": "Widgets"})
    assert (a["ticket_id"], b["ticket_id"]) == ("FR-0001", "FR-0002")
    lines = [json.loads(x) for x in (tmp_path / "fr.jsonl").read_text().splitlines()]
    assert lines[0]["title"] == "Linux app" and lines[0]["customer_id"] == "demo_customer"
    assert lines[1]["customer_id"] == "anonymous"


def test_tool_errors_come_back_as_text_not_exceptions(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FEATURE_REQUESTS_PATH", tmp_path / "fr.jsonl")
    assert log_feature_request_tool.invoke({"title": "   "}) == "A feature request needs a title."
    assert "customer_id" in check_invoice_tool.invoke({})                      # names the missing field


def test_unexpected_crash_becomes_a_safe_message(monkeypatch):
    def broken(_):
        raise RuntimeError("db password=hunter2")
    monkeypatch.setitem(tools.MOCK_CUSTOMERS, "demo_customer", None)
    monkeypatch.setattr(tools, "_customer", broken)
    out = check_invoice_tool.invoke({"customer_id": "demo_customer"})
    assert "temporarily unavailable" in out and "hunter2" not in out


def test_each_agent_only_has_its_own_tools():
    names = {k: {t.name for t in v} for k, v in TOOLS_BY_AGENT.items()}
    assert names["billing"] == {"check_invoice", "check_refund_eligibility"}
    assert names["technical"] == {"check_service_status", "search_docs"}
    assert names["feature_request"] == {"check_roadmap", "log_feature_request", "search_docs"}
    assert names["general"] == {"search_docs"}
    assert "search_docs" not in names["billing"]            # billing works from account data only
    assert set(config.INTENTS) == set(TOOLS_BY_AGENT)


def test_tool_descriptions_exist_for_the_llm():
    for tool_list in TOOLS_BY_AGENT.values():
        for t in tool_list:
            assert len(t.description) > 40 and t.args_schema is not None
