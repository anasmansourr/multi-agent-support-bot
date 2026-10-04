"""Tools the agents can call (LangChain StructuredTools).

*** ALL DATA HERE IS MOCK / DEMO DATA. ***
There is no real billing system, status page or roadmap behind these tools. They return
deterministic, hard-coded records so the demo is repeatable and testable. The only real
side effect is `log_feature_request`, which appends a line to data/feature_requests.jsonl.

Rules
  * Each agent gets ONLY the tools it needs (see TOOLS_BY_AGENT).
  * Tools never raise to the agent: expected problems return a clear message
    (found=False, ...) and unexpected ones become a ToolException that LangChain hands back
    to the model as plain text (handle_tool_error=True). The app never crashes on a tool.
  * `search_docs` is backed by the RAG retriever (src/rag), which is NOT mock: it searches real documents
    (fictional FlowDesk docs written for this project in data/knowledge_base/).
"""
import functools
import json
import logging
from datetime import date, timedelta

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field

from src import config

logger = logging.getLogger(__name__)

MOCK_TODAY = date(2026, 10, 1)      # fixed "today" so mock dates never drift

# --------------------------------------------------------------- MOCK DATA
MOCK_CUSTOMERS = {
    # The one used in the demo: "It says I owe 500."
    "demo_customer": {"name": "Demo Customer", "plan": "Pro (5 seats)", "invoice_id": "INV-2041",
                      "status": "overdue", "amount_due": 500.00, "due_in_days": -16,
                      "last_payment_amount": 245.00, "last_payment_days_ago": 46, "payment_method": "Visa ending 4242"},
    "cust_paid": {"name": "Paid Up Inc.", "plan": "Starter (2 seats)", "invoice_id": "INV-2099",
                  "status": "paid", "amount_due": 0.00, "due_in_days": 14,
                  "last_payment_amount": 58.00, "last_payment_days_ago": 6, "payment_method": "Mastercard ending 1881"},
    "cust_recent": {"name": "Recent Buyer Ltd.", "plan": "Business (10 seats)", "invoice_id": "INV-2105",
                    "status": "paid", "amount_due": 0.00, "due_in_days": 28,
                    "last_payment_amount": 790.00, "last_payment_days_ago": 9, "payment_method": "Visa ending 0007"},
}
REFUND_WINDOW_DAYS = 30

MOCK_SERVICE_STATUS = {
    "api": {"status": "operational", "note": ""},
    "dashboard": {"status": "operational", "note": ""},
    "mobile_app": {"status": "operational", "note": ""},
    "email_notifications": {"status": "degraded_performance", "note": "Delivery delays of up to 30 minutes."},
    "webhooks": {"status": "partial_outage", "note": "Some deliveries are failing. Engineers are investigating."},
}

MOCK_ROADMAP = [
    {"feature": "Dark mode", "status": "planned", "eta": "Q1 2027", "votes": 128, "tags": ["dark", "theme", "mode"]},
    {"feature": "Excel export", "status": "under_review", "eta": None, "votes": 64, "tags": ["excel", "export", "xlsx"]},
    {"feature": "SAML single sign-on", "status": "in_progress", "eta": "Q4 2026", "votes": 97, "tags": ["sso", "saml", "okta", "login"]},
    {"feature": "Kanban board view", "status": "shipped", "eta": None, "votes": 85, "tags": ["kanban", "board"]},
    {"feature": "Recurring tasks", "status": "planned", "eta": "Q2 2027", "votes": 52, "tags": ["recurring", "repeat", "tasks"]},
    {"feature": "Offline mode", "status": "not_planned", "eta": None, "votes": 41, "tags": ["offline"]},
    {"feature": "Zapier integration", "status": "under_review", "eta": None, "votes": 33, "tags": ["zapier", "integration"]},
]


# --------------------------------------------------------------- helpers
def _safe(func):
    """Turn unexpected exceptions into a ToolException with a generic, secret-free message."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ToolException:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("Tool %s failed", func.__name__)
            raise ToolException(f"The {func.__name__} tool is temporarily unavailable. "
                                "Tell the customer you could not retrieve this information right now.")
    return wrapper


def _validation_message(error):
    fields = sorted({str(e["loc"][0]) for e in error.errors() if e.get("loc")})
    return "Invalid tool input - missing or invalid field(s): " + ", ".join(fields)


def _make_tool(func, schema):
    return StructuredTool.from_function(
        func=_safe(func), name=func.__name__, description=(func.__doc__ or "").strip(),
        args_schema=schema, handle_tool_error=True, handle_validation_error=_validation_message)


def _customer(customer_id):
    return MOCK_CUSTOMERS.get(str(customer_id).strip().lower())


def _not_found(customer_id):
    return {"found": False, "message": f"No customer found with id '{customer_id}'. "
                                       "Ask the customer to confirm their customer id."}


# --------------------------------------------------------------- billing tools
class CustomerInput(BaseModel):
    customer_id: str = Field(description="The customer's account id, e.g. 'demo_customer'.")


def check_invoice(customer_id: str) -> dict:
    """Look up a customer's latest invoice: status, amount due, due date, last payment and plan.
    Use this whenever the customer asks what they owe, why they were charged, or about an invoice."""
    c = _customer(customer_id)
    if c is None:
        return _not_found(customer_id)
    return {"found": True, "customer_id": customer_id.strip().lower(), "name": c["name"], "plan": c["plan"],
            "invoice_id": c["invoice_id"], "invoice_status": c["status"], "amount_due_usd": c["amount_due"],
            "due_date": str(MOCK_TODAY + timedelta(days=c["due_in_days"])),
            "last_payment_usd": c["last_payment_amount"],
            "last_payment_date": str(MOCK_TODAY - timedelta(days=c["last_payment_days_ago"])),
            "payment_method": c["payment_method"]}


def check_refund_eligibility(customer_id: str) -> dict:
    """Check whether a customer's most recent payment can still be refunded (30-day refund window).
    Use this when the customer asks for a refund or money back. This only CHECKS; it does not issue a refund."""
    c = _customer(customer_id)
    if c is None:
        return _not_found(customer_id)
    days = c["last_payment_days_ago"]
    eligible = days <= REFUND_WINDOW_DAYS
    return {"found": True, "customer_id": customer_id.strip().lower(), "eligible": eligible,
            "last_payment_usd": c["last_payment_amount"], "days_since_payment": days,
            "refund_window_days": REFUND_WINDOW_DAYS,
            "reason": (f"Payment was {days} days ago, inside the {REFUND_WINDOW_DAYS}-day window." if eligible
                       else f"Payment was {days} days ago, outside the {REFUND_WINDOW_DAYS}-day window.")}


# --------------------------------------------------------------- technical tool
class ServiceInput(BaseModel):
    service: str = Field(default="all", description="One of: api, dashboard, mobile_app, "
                                                    "email_notifications, webhooks, or 'all'.")


def check_service_status(service: str = "all") -> dict:
    """Check the current status of a product service (outage or degradation). Use this when the customer
    reports something not working, slow, or failing, before suggesting troubleshooting steps."""
    key = str(service).strip().lower().replace(" ", "_") or "all"
    if key == "all":
        return {"found": True, "services": {k: dict(v) for k, v in MOCK_SERVICE_STATUS.items()}}
    if key not in MOCK_SERVICE_STATUS:
        return {"found": False, "message": f"Unknown service '{service}'.",
                "valid_services": sorted(MOCK_SERVICE_STATUS)}
    return {"found": True, "service": key, **MOCK_SERVICE_STATUS[key]}


# --------------------------------------------------------------- feature-request tools
class RoadmapInput(BaseModel):
    keyword: str = Field(description="A word or phrase for the feature, e.g. 'dark mode' or 'excel'.")


def check_roadmap(keyword: str) -> dict:
    """Search the product roadmap for a feature. Returns its status (shipped, in_progress, planned,
    under_review, not_planned), ETA and votes. Use this BEFORE logging a new feature request."""
    words = [w for w in str(keyword).lower().replace("-", " ").split() if w]
    if not words:
        raise ToolException("Please provide a keyword for the feature.")
    # Score = how many of the query words hit the feature name or its tags; keep only the best matches,
    # so "dark mode" finds "Dark mode" and not "Offline mode" (which shares only the word "mode").
    scored = [(sum(w in item["feature"].lower() or w in item["tags"] for w in words), item) for item in MOCK_ROADMAP]
    best = max((score for score, _ in scored), default=0)
    hits = [{k: v for k, v in item.items() if k != "tags"} for score, item in scored if score == best and best > 0]
    return {"found": True, "matches": hits} if hits else \
        {"found": False, "message": f"Nothing on the roadmap matches '{keyword}'."}


class FeatureRequestInput(BaseModel):
    title: str = Field(description="Short title of the requested feature.")
    description: str = Field(default="", description="What the customer wants and why.")
    customer_id: str = Field(default="anonymous", description="The customer's account id, if known.")


def log_feature_request(title: str, description: str = "", customer_id: str = "anonymous") -> dict:
    """Record a NEW feature request from a customer and return its ticket id. Only call this after the
    customer confirmed they want it logged and check_roadmap showed it is not already planned or shipped."""
    title = str(title).strip()
    if not title:
        raise ToolException("A feature request needs a title.")
    path = config.FEATURE_REQUESTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = sum(1 for _ in open(path, encoding="utf-8")) if path.exists() else 0
    ticket = f"FR-{existing + 1:04d}"
    entry = {"ticket_id": ticket, "title": title, "description": str(description).strip(),
             "customer_id": str(customer_id).strip() or "anonymous"}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"logged": True, "ticket_id": ticket, "title": title}


# --------------------------------------------------------------- RAG tool
class DocsInput(BaseModel):
    query: str = Field(description="What to look up, as a short question or keywords, e.g. 'refund window' or 'webhook retries'.")


def search_docs(query: str) -> dict:
    """Search the FlowDesk documentation (pricing, policies, troubleshooting, API, security, support hours,
    company info). Returns the most relevant excerpts with their source file. Use it for any factual question
    about the product or company, and answer ONLY from what it returns."""
    query = str(query).strip()
    if not query:
        raise ToolException("Please provide something to search for.")
    from src.rag import retriever as rag           # imported lazily: loading the embedding model is slow
    docs = rag.get_retriever().invoke(query)
    if not docs:
        return {"found": False, "message": "No relevant documentation was found. The documentation does not "
                                           "provide enough information to answer this."}
    return {"found": True, "results": [{"source": d.metadata["source"], "score": round(d.metadata["score"], 2),
                                        "excerpt": d.page_content} for d in docs]}


# --------------------------------------------------------------- registry
check_invoice_tool = _make_tool(check_invoice, CustomerInput)
check_refund_eligibility_tool = _make_tool(check_refund_eligibility, CustomerInput)
check_service_status_tool = _make_tool(check_service_status, ServiceInput)
check_roadmap_tool = _make_tool(check_roadmap, RoadmapInput)
log_feature_request_tool = _make_tool(log_feature_request, FeatureRequestInput)
search_docs_tool = _make_tool(search_docs, DocsInput)

# Each agent sees only its own tools. Technical, feature-request and general can search the docs (RAG);
# billing works from account data only.
TOOLS_BY_AGENT = {
    "billing": [check_invoice_tool, check_refund_eligibility_tool],
    "technical": [check_service_status_tool, search_docs_tool],
    "feature_request": [check_roadmap_tool, log_feature_request_tool, search_docs_tool],
    "general": [search_docs_tool],
}


def main():
    """python -m src.agents.tools   - call every tool directly so you can see what they return."""
    demos = [
        (check_invoice_tool, {"customer_id": "demo_customer"}),
        (check_invoice_tool, {"customer_id": "nobody"}),
        (check_refund_eligibility_tool, {"customer_id": "cust_recent"}),
        (check_refund_eligibility_tool, {"customer_id": "demo_customer"}),
        (check_service_status_tool, {"service": "webhooks"}),
        (check_roadmap_tool, {"keyword": "dark mode"}),
        (check_roadmap_tool, {"keyword": "teleportation"}),
        (log_feature_request_tool, {"title": "Demo request", "description": "from the tools demo", "customer_id": "demo_customer"}),
        (check_invoice_tool, {}),
    ]
    for tool, args in demos:
        print(f"{tool.name}({args})\n   -> {tool.invoke(args)}\n")
    print(f"(the log_feature_request demo appended a line to {config.FEATURE_REQUESTS_PATH})")


if __name__ == "__main__":
    main()
