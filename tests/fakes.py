"""A scripted chat model for tests: no network, but it behaves like a tool-calling LLM."""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr


def tool_call_msg(name, args, call_id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


class ScriptedChatModel(BaseChatModel):
    """Returns the scripted AIMessages in order (the last one repeats). Records every prompt it saw."""
    script: list = Field(default_factory=list)
    seen: list = Field(default_factory=list)
    seen_tool_names: list = Field(default_factory=list)
    _i: int = PrivateAttr(default=0)

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kwargs):          # required by create_tool_calling_agent
        self.seen_tool_names = [t.name for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        msg = self.script[min(self._i, len(self.script) - 1)]
        self._i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])
