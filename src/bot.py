"""SupportBot = Router + ConversationMemory. This is what the Streamlit app (and the terminal chat) uses.

    bot = SupportBot()
    result = bot.ask("I have a billing problem")      # remembers the turn
    result = bot.ask("It says I owe 500")             # 'it' is understood from memory

Terminal chat:   python -m src.bot          commands: /reset  /memory  /outage  /quit
"""
from src import config
from src.memory.history import ConversationMemory
from src.routing.router import Router


class SupportBot:
    def __init__(self, router=None, memory=None):
        self.router = router or Router()
        self.memory = memory or ConversationMemory()

    def ask(self, message, customer_id=config.DEFAULT_CUSTOMER_ID, threshold=None, providers=None, simulate_outage=()):
        result = self.router.handle(message, chat_history=self.memory.messages(), customer_id=customer_id,
                                    threshold=threshold, providers=providers, simulate_outage=simulate_outage,
                                    active_agent=self.memory.last_agent())
        if result.agent is not None:                       # a failed turn is not remembered
            self.memory.add(message, result.agent.output, agent=result.routing.selected_agent)
        return result

    def reset(self):
        self.memory.clear()


def _show(result):
    d = result.routing
    note = {"ok": "", "low_confidence": "  [LOW CONFIDENCE -> general]",
            "low_confidence_kept_agent": "  [LOW CONFIDENCE -> kept current agent]"}.get(d.reason, f"  [{d.reason}]")
    print(f"  classifier: {d.predicted_intent} ({d.confidence:.2f}) -> agent: {d.selected_agent}{note}")
    if result.agent is None:
        print(f"  ERROR: {result.error}")
        return
    a = result.agent
    print(f"  llm: {a.provider} ({a.model})" + (" [FALLBACK]" if a.used_fallback else ""))
    for c in a.tool_calls:
        print(f"  TOOL CALL: {c['tool']}({c['args']}) -> {str(c['result'])[:160]}")
    print(f"\nBot: {a.output}\n")


def main():
    bot = SupportBot()
    outage = False
    print("FlowDesk support chat (demo). Commands: /reset  /memory  /outage (toggle Groq outage)  /quit\n")
    while True:
        try:
            text = input(f"You{' [Groq outage ON]' if outage else ''}: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/reset":
            bot.reset()
            print("  (memory cleared)\n")
        elif text == "/memory":
            for t in bot.memory.turns:
                print(f"  [{t.agent}] You: {t.user}\n           Bot: {t.assistant[:100]}")
            print()
        elif text == "/outage":
            outage = not outage
            print(f"  (simulated Groq outage: {'ON' if outage else 'OFF'})\n")
        else:
            _show(bot.ask(text, simulate_outage={"groq"} if outage else ()))


if __name__ == "__main__":
    main()
