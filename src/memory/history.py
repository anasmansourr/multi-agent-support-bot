"""Conversation memory: one shared, bounded history for the whole conversation.

It is NOT stored per agent. Whichever agent the router picks receives the same recent history,
so context survives a handoff (billing -> technical keeps what the customer already said).

Summarization is designed in but off by default: pass `summarizer(old_summary, dropped_turns) -> str`
(for example an LLM call) and turns that fall out of the window are condensed instead of forgotten.
"""
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage

from src import config


@dataclass
class Turn:
    user: str
    assistant: str
    agent: str          # which agent answered this turn


class ConversationMemory:
    def __init__(self, max_turns=None, max_chars=None, summarizer=None):
        self.max_turns = config.MEMORY_MAX_TURNS if max_turns is None else max_turns
        self.max_chars = config.MEMORY_MAX_CHARS if max_chars is None else max_chars
        self.summarizer = summarizer
        self.turns = []
        self.summary = ""
        self.dropped = 0

    def _clip(self, text):
        text = str(text).strip()
        return text if len(text) <= self.max_chars else text[: self.max_chars].rstrip() + " ..."

    def add(self, user, assistant, agent):
        self.turns.append(Turn(self._clip(user), self._clip(assistant), agent))
        overflow = len(self.turns) - self.max_turns
        if overflow > 0:
            old, self.turns = self.turns[:overflow], self.turns[overflow:]
            self.dropped += len(old)
            if self.summarizer is not None:
                try:
                    self.summary = self.summarizer(self.summary, old) or self.summary
                except Exception:  # noqa: BLE001  - a failing summarizer must never break the chat
                    pass

    def messages(self):
        """LangChain messages for the agents' `chat_history`. Alternates human/AI, summary first."""
        out = []
        if self.summary:
            out += [HumanMessage(f"Summary of the earlier conversation: {self.summary}"), AIMessage("Understood.")]
        for t in self.turns:
            out += [HumanMessage(t.user), AIMessage(t.assistant)]
        return out

    def last_agent(self):
        return self.turns[-1].agent if self.turns else None

    def clear(self):
        self.turns, self.summary, self.dropped = [], "", 0

    def __len__(self):
        return len(self.turns)
