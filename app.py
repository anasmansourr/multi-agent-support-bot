"""Streamlit UI.  Run:  streamlit run app.py

Main panel : chat. Sidebar: how the last message was routed (intent, confidence, agent), the LLM provider,
the real tool calls, a usage counter, and demo controls (Groq-outage switch, threshold, customer)."""
import logging

import streamlit as st

from src import config
from src.ui_helpers import (AGENT_NAMES, REASON_TEXT, attempts_text, message_caption, new_usage, provider_label,
                            record_usage)

logger = logging.getLogger(__name__)
st.set_page_config(page_title="FlowDesk Support Assistant", layout="wide")

CUSTOMERS = {"demo_customer": "Pro, owes $500 (overdue)", "cust_recent": "Business, paid 9 days ago",
             "cust_paid": "Starter, paid 6 days ago"}
DEMO_MESSAGES = {
    "Billing": ["I have a billing problem.", "It says I owe 500"],
    "Technical (handoff after billing)": ["the invoice page will not load", "the app crashes when I upload a file"],
    "Feature request": ["please add a linux app", "please add dark mode"],
    "General / RAG": ["hello", "What are your support hours?", "Which data regions can I choose?"],
    "Low confidence": ["uploading a photo gives me a red banner saying failed"],
    "Unknown input": ["asdf qwerty"],
    "Not in the documentation": ["Can I run FlowDesk on my own server?"],
}


@st.cache_resource(show_spinner="Loading the intent classifier...")
def load_router():
    from src.routing.router import Router
    return Router()


@st.cache_resource(show_spinner="Loading the documentation search model...")
def warm_up_docs():
    try:
        from src.rag.retriever import get_retriever
        get_retriever()
        return True
    except Exception:  # noqa: BLE001  - the search tool reports its own failures later
        logger.exception("Could not pre-load the documentation index")
        return False


def create_bot():
    from src.bot import SupportBot
    bot = SupportBot(router=load_router())
    warm_up_docs()
    return bot


def init_state():
    ss = st.session_state
    if "bot" not in ss:
        try:
            ss.bot = create_bot()
        except FileNotFoundError:
            st.error("The trained classifier was not found (models/classifier.pt). "
                     "Train it first with: `python -m src.classifier.train`")
            st.stop()
        except Exception:  # noqa: BLE001
            logger.exception("Startup failed")
            st.error("Could not start the assistant. Check the terminal for details.")
            st.stop()
    ss.setdefault("messages", [])
    ss.setdefault("last", None)
    ss.setdefault("usage", new_usage())
    ss.setdefault("queued", None)


def queue_message(text):
    st.session_state.queued = text


def reset_conversation():
    ss = st.session_state
    ss.bot.reset()
    ss.messages, ss.last, ss.usage = [], None, new_usage()


# ------------------------------------------------------------------ page
init_state()
ss = st.session_state

st.title("FlowDesk Support Assistant")
st.caption("A custom Bi-LSTM intent classifier routes each message to a specialised agent "
           "(tools + document search). FlowDesk is fictional; account, status and roadmap data are mock data.")

# --- sidebar: reserve the top for the routing panel (filled after the message is processed), controls below
route_box = st.sidebar.container()
with st.sidebar:
    st.divider()
    st.subheader("Controls")
    customer = st.selectbox("Signed-in customer", list(CUSTOMERS), format_func=lambda c: f"{c} - {CUSTOMERS[c]}")
    threshold = st.slider("Confidence threshold", 0.0, 1.0, float(config.CONFIDENCE_THRESHOLD), 0.01,
                          help="Below this, the message goes to the General agent.")
    outage = st.toggle("Simulate Groq outage", help="Forces the Ollama fallback, for the demo.")
    if outage:
        st.warning("Groq outage simulation is ON: answers come from Ollama.")
    st.button("Reset conversation", on_click=reset_conversation, use_container_width=True)
    with st.expander("Demo messages"):
        for group, messages in DEMO_MESSAGES.items():
            st.caption(group)
            for i, text in enumerate(messages):
                st.button(text, key=f"demo_{group}_{i}", on_click=queue_message, args=(text,), use_container_width=True)

# --- chat history
for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("caption"):
            st.caption(m["caption"])

# --- new message (typed or from a demo button)
prompt = st.chat_input("Type your message...") or ss.queued
ss.queued = None
if prompt:
    ss.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ss.bot.ask(prompt, customer_id=customer, threshold=threshold,
                                    simulate_outage={"groq"} if outage else ())
            except Exception:  # noqa: BLE001  - the bot already guards itself; this is the UI's own last resort
                logger.exception("Unexpected UI error")
                from src.routing.router import GENERIC_ERROR_MESSAGE, RouterResult, RoutingDecision
                result = RouterResult(RoutingDecision("unknown", 0.0, "general", False, "classifier_error"),
                                      None, GENERIC_ERROR_MESSAGE)
        text = result.agent.output if result.agent else result.error
        caption = message_caption(result)
        st.markdown(text)
        st.caption(caption)
    ss.messages.append({"role": "assistant", "content": text, "caption": caption})
    ss.last = result
    record_usage(ss.usage, result)

# --- sidebar: routing panel for the last message
with route_box:
    st.subheader("Routing (last message)")
    last = ss.last
    if last is None:
        st.info("Send a message to see how it is routed.")
    else:
        d = last.routing
        c1, c2 = st.columns(2)
        c1.metric("Predicted intent", d.predicted_intent)
        c2.metric("Confidence", f"{d.confidence:.3f}")
        st.progress(min(max(d.confidence, 0.0), 1.0), text=f"threshold {d.threshold:.2f}")
        st.metric("Selected agent", AGENT_NAMES.get(d.selected_agent, d.selected_agent))
        st.caption(REASON_TEXT.get(d.reason, d.reason))
        if d.known_ratio is not None:
            st.caption(f"Known words: {d.known_ratio:.0%}")
        if last.agent is None:
            st.error(last.error)
        else:
            a = last.agent
            st.markdown(f"**LLM:** {provider_label(a)}")
            if a.used_fallback:
                st.warning(f"Fallback used ({attempts_text(a.attempts)})")
            st.markdown("**Tool calls**")
            if not a.tool_calls:
                st.caption("none")
            for call in a.tool_calls:
                st.markdown(f"`{call['tool']}`")
                st.json({"args": call["args"], "result": call["result"]}, expanded=False)
    st.divider()
    st.subheader("Usage (this session)")
    u = ss.usage
    st.markdown(f"Messages **{u['messages']}** · Groq **{u['groq']}** · Ollama **{u['ollama']}**  \n"
                f"Fallbacks **{u['fallbacks']}** · Tool calls **{u['tool_calls']}** · Errors **{u['errors']}**")
