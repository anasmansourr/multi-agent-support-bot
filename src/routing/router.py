"""Routing: the custom classifier decides the intent; this module picks the agent. No LLM is involved
in the decision.

    message -> IntentPredictor (LSTM) -> decide() -> RunnableBranch -> run_agent(<that intent>)

  decide()       pure function: (intent, confidence) -> which agent. Holds the ONE low-confidence rule.
  build_chain()  LangChain composition: RunnableLambda (classify) | RunnableBranch (one branch per agent)
  Router.handle  the call the UI makes; never raises on provider failure, returns RouterResult.error instead

Probe routing without calling any LLM (free, instant):
    python -m src.routing.router --route-only "it says i owe 500" "please add dark mode"
Full pipeline:
    python -m src.routing.router "it says i owe 500"
"""
import math
from dataclasses import dataclass, field
from functools import partial

from langchain_core.runnables import RunnableBranch, RunnableLambda

from src import config
from src.agents.agents import AgentResult, run_agent
from src.llm.providers import AllProvidersFailed

FALLBACK_AGENT = "general"


@dataclass
class RoutingDecision:
    predicted_intent: str          # what the classifier said (kept even when we override it)
    confidence: float
    selected_agent: str            # which agent will actually run
    fallback_applied: bool         # True when selected_agent != predicted_intent
    reason: str                    # ok | low_confidence | classifier_error | invalid_output
    scores: dict = field(default_factory=dict)
    threshold: float = config.CONFIDENCE_THRESHOLD


@dataclass
class RouterResult:
    routing: RoutingDecision
    agent: AgentResult | None = None   # None when every LLM provider failed
    error: str | None = None           # user-safe message, set when agent is None
    attempts: list = field(default_factory=list)


def decide(message, predictor, threshold=None, active_agent=None, sticky=None):
    """Classifier -> routing decision. Defensive: bad classifier output never crashes the app,
    it just routes to the general agent with a reason.

    Low confidence normally means "general agent". If `sticky` (config.FOLLOWUP_STICKY) and a specialist agent
    is already handling the conversation (`active_agent`), a low-confidence follow-up such as "yes please"
    stays with that agent instead. A CONFIDENT classification always wins, which is how handoffs happen."""
    threshold = config.CONFIDENCE_THRESHOLD if threshold is None else threshold
    sticky = config.FOLLOWUP_STICKY if sticky is None else sticky
    try:
        out = predictor.predict(message)
        intent, confidence = out["intent"], float(out["confidence"])
        scores = dict(out.get("scores", {}))
    except Exception:  # noqa: BLE001
        return RoutingDecision("unknown", 0.0, FALLBACK_AGENT, True, "classifier_error", threshold=threshold)

    if intent not in config.INTENTS or math.isnan(confidence) or not 0.0 <= confidence <= 1.0:
        return RoutingDecision(str(intent), 0.0, FALLBACK_AGENT, True, "invalid_output", scores, threshold)
    if confidence < threshold:
        if sticky and active_agent in config.INTENTS and active_agent != FALLBACK_AGENT:
            return RoutingDecision(intent, confidence, active_agent, intent != active_agent,
                                   "low_confidence_kept_agent", scores, threshold)
        return RoutingDecision(intent, confidence, FALLBACK_AGENT, intent != FALLBACK_AGENT,
                               "low_confidence", scores, threshold)
    return RoutingDecision(intent, confidence, intent, False, "ok", scores, threshold)


# ------------------------------------------------------------------ LangChain chain
def _run(intent, state):
    """One branch: run exactly the agent for `intent`."""
    return run_agent(intent, state["input"], state.get("chat_history"), state.get("customer_id", config.DEFAULT_CUSTOMER_ID),
                     providers=state.get("providers"), simulate_outage=state.get("simulate_outage", ()))


def build_chain(predictor):
    """RunnableLambda(classify) | RunnableBranch(one branch per agent, general = default).
    Input dict: input, chat_history, customer_id, [threshold, providers, simulate_outage].
    Output dict: {"routing": RoutingDecision, "agent": AgentResult}."""
    # Classify unless the caller already did (Router.handle does, so it can still report the routing if the LLM fails).
    classify = RunnableLambda(lambda s: s if "routing" in s
                              else {**s, "routing": decide(s["input"], predictor, s.get("threshold"), s.get("active_agent"))})
    branches = [(lambda s, i=intent: s["routing"].selected_agent == i, RunnableLambda(partial(_run, intent)))
                for intent in config.INTENTS if intent != FALLBACK_AGENT]
    branch = RunnableBranch(*branches, RunnableLambda(partial(_run, FALLBACK_AGENT)))
    run_selected = RunnableLambda(lambda s: {"routing": s["routing"], "agent": branch.invoke(s)})
    return classify | run_selected


class Router:
    def __init__(self, predictor=None):
        if predictor is None:
            from src.classifier.inference import IntentPredictor
            predictor = IntentPredictor()
        self.predictor = predictor
        self.chain = build_chain(predictor)

    def route_only(self, message, threshold=None):
        return decide(message, self.predictor, threshold)

    def handle(self, message, chat_history=None, customer_id=config.DEFAULT_CUSTOMER_ID, threshold=None,
               providers=None, simulate_outage=(), active_agent=None):
        routing = decide(message, self.predictor, threshold, active_agent)    # the classifier runs exactly once per message
        state = {"input": message, "chat_history": list(chat_history or []), "customer_id": customer_id,
                 "routing": routing, "providers": providers, "simulate_outage": simulate_outage}
        try:
            out = self.chain.invoke(state)
            return RouterResult(out["routing"], out["agent"])
        except AllProvidersFailed as exc:
            return RouterResult(routing, None, exc.user_message, exc.attempts)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("messages", nargs="+")
    parser.add_argument("--route-only", action="store_true", help="show routing decisions only (no LLM call)")
    parser.add_argument("--simulate-groq-outage", action="store_true")
    args = parser.parse_args()
    router = Router()

    for message in args.messages:
        if args.route_only:
            d = router.route_only(message)
            print(f"{message!r}\n   predicted={d.predicted_intent} ({d.confidence:.2f}) -> agent={d.selected_agent}"
                  f"  [{d.reason}]" + ("  FALLBACK" if d.fallback_applied else ""))
            continue
        r = router.handle(message, simulate_outage={"groq"} if args.simulate_groq_outage else ())
        d = r.routing
        print(f"\nMESSAGE: {message}\nClassifier: {d.predicted_intent} ({d.confidence:.2f}) -> agent: {d.selected_agent}"
              f" [{d.reason}, threshold {d.threshold:.2f}]")
        if r.agent is None:
            print(f"ERROR (user-safe): {r.error}")
            continue
        print(f"LLM: {r.agent.provider} ({r.agent.model})" + (" [fallback]" if r.agent.used_fallback else ""))
        for c in r.agent.tool_calls:
            print(f"TOOL CALL: {c['tool']}({c['args']}) -> {c['result']}")
        print(f"ANSWER: {r.agent.output}")


if __name__ == "__main__":
    main()
