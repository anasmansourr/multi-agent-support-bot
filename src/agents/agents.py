"""One tool-calling agent per intent, plus `run_agent`, the single entry point the router calls.

    classifier intent -> run_agent(intent, message, history) -> AgentExecutor (own prompt + own tools)
                                                              -> LLM via call_with_fallback (Groq, else Ollama)
Try it:   python -m src.agents.agents --intent billing "It says I owe 500"
"""
from dataclasses import dataclass, field

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src import config
from src.agents.prompts import build_system_prompt
from src.agents.tools import TOOLS_BY_AGENT
from src.llm.providers import AllProvidersFailed, call_with_fallback


@dataclass
class AgentResult:
    output: str
    intent: str                                   # which agent actually ran
    tool_calls: list = field(default_factory=list)  # [{"tool", "args", "result"}] - REAL executions
    provider: str = ""
    model: str = ""
    used_fallback: bool = False
    attempts: list = field(default_factory=list)  # failed providers before the one that answered


class _NoToolAgent:
    """Used only when an agent has no tools (the General agent until search_docs arrives in Phase 13):
    an empty tool list cannot be bound to the model, so run prompt -> LLM directly with the same output shape."""

    def __init__(self, prompt, llm):
        self._chain = prompt.partial(agent_scratchpad=[]) | llm | StrOutputParser()

    def invoke(self, inputs):
        return {"output": self._chain.invoke(inputs), "intermediate_steps": []}


def build_agent(intent, llm):
    """Build the executor for one intent. Each agent gets ONLY its own tools and its own prompt."""
    tools = TOOLS_BY_AGENT[intent]
    system = build_system_prompt(intent, [t.name for t in tools])
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    if not tools:
        return _NoToolAgent(prompt, llm)
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, max_iterations=config.AGENT_MAX_ITERATIONS,
                         handle_parsing_errors=True, return_intermediate_steps=True)


def _text(output):
    """Some models return a list of content blocks instead of a string."""
    if isinstance(output, list):
        output = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in output)
    return str(output).strip()


def run_agent(intent, user_message, chat_history=None, customer_id=config.DEFAULT_CUSTOMER_ID,
              providers=None, simulate_outage=()):
    """Run the agent for `intent`. Unknown intents go to the general agent.
    Raises AllProvidersFailed (user-safe message) if neither Groq nor Ollama can answer."""
    if intent not in TOOLS_BY_AGENT:
        intent = "general"
    inputs = {"input": user_message, "chat_history": list(chat_history or []), "customer_id": customer_id}
    fb = call_with_fallback(lambda llm: build_agent(intent, llm).invoke(inputs), providers, simulate_outage)
    tool_calls = [{"tool": action.tool, "args": action.tool_input, "result": observation}
                  for action, observation in fb.value.get("intermediate_steps", [])]
    return AgentResult(output=_text(fb.value.get("output", "")) or "Sorry, I could not produce an answer.",
                       intent=intent, tool_calls=tool_calls, provider=fb.provider, model=fb.model,
                       used_fallback=fb.used_fallback, attempts=fb.attempts)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("message")
    parser.add_argument("--intent", required=True, choices=list(TOOLS_BY_AGENT))
    parser.add_argument("--customer", default=config.DEFAULT_CUSTOMER_ID)
    parser.add_argument("--simulate-groq-outage", action="store_true")
    args = parser.parse_args()
    try:
        r = run_agent(args.intent, args.message, customer_id=args.customer,
                      simulate_outage={"groq"} if args.simulate_groq_outage else ())
    except AllProvidersFailed as exc:
        print(f"\n{exc.user_message}")
        for a in exc.attempts:
            print(f"  {a['provider']}: [{a['category']}] {a['message']}")
        raise SystemExit(1)
    print(f"\nAgent: {r.intent} | LLM: {r.provider} ({r.model})" + (" [fallback]" if r.used_fallback else ""))
    if r.tool_calls:
        for c in r.tool_calls:
            print(f"TOOL CALL: {c['tool']}({c['args']})\n   -> {c['result']}")
    else:
        print("TOOL CALL: (none)")
    print(f"\nANSWER: {r.output}")


if __name__ == "__main__":
    main()
