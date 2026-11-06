"""Chat-model factory (one place to swap providers) + a scripted fake LLM.

LLM_PROVIDER=fake       
LLM_PROVIDER=ollama  
"""
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

STATUS_WORDS = ["placed", "paid", "shipped", "delivered", "cancelled", "refunded"]


def get_chat_model() -> BaseChatModel:
    provider = os.environ.get("LLM_PROVIDER", "fake").lower()
    if provider == "fake":
        return FakeChatModel()
    if provider == "ollama":
        from langchain_ollama import ChatOllama  # lazy: not needed offline

        return ChatOllama(
            model=os.environ.get("LLM_MODEL", "gemma4:latest"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.7,
            max_tokens=1024,
        )
    raise ValueError(f"Unknown LLM_PROVIDER {provider!r} (use 'fake' or 'ollama')")


class FakeChatModel(BaseChatModel):
    """A real (rule-based) scripted model, not a stub that says "ok".

    * Turn start: parses the staff question and picks a tool, like a model would.
    * After tool results: writes an answer *from the tool output* (including errors).
    * Deliberately GULLIBLE: if injection text reaches it inside a tool result it
      obeys and calls refund_order. The guardrails must keep that text away from it,
      so the injection evals fail loudly if the guardrails regress.
    * Records every message it receives in `seen`, so tests can assert that no PII
      ever reached the model.
    """

    seen: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "shophub-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeChatModel":
        return self

    # ---- planning ----------------------------------------------------------
    def _plan(self, messages: list[BaseMessage]) -> AIMessage:
        self.seen.append(list(messages))
        tail: list[ToolMessage] = []
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                tail.append(m)
            elif isinstance(m, HumanMessage):
                break
        question = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        question = question if isinstance(question, str) else ""
        if tail:
            return self._answer(question, list(reversed(tail)))
        return self._route(question)

    @staticmethod
    def _call(name: str, args: dict) -> AIMessage:
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}_{len(str(args))}", "type": "tool_call"}])

    def _route(self, q: str) -> AIMessage:
        low = q.lower()
        if m := re.search(r"\b(refund|cancel)\b\D{0,20}?(\d{3,})", low):
            return self._call(f"{m.group(1)}_order", {"order_id": int(m.group(2))})
        if m := re.search(r"\border\D{0,12}?(\d{3,})", low):
            return self._call("lookup_order", {"order_id": int(m.group(1))})
        if "not shipped" in low or "unshipped" in low:
            return self._call("list_orders", {"status": "PAID"})
        if "orders" in low or "customer" in low:
            for s in STATUS_WORDS:
                if s in low:
                    return self._call("list_orders", {"status": s.upper()})
            return self._call("list_orders", {})
        return AIMessage(content="I can look up orders, list orders by status, and propose refunds or cancellations. What would you like?")

    def _answer(self, q: str, results: list[ToolMessage]) -> AIMessage:
        lines: list[str] = []
        for tm in results:
            raw = tm.content if isinstance(tm.content, str) else json.dumps(tm.content)
            # Gullible on purpose (see class docstring).
            if re.search(r"ignore (your|all|previous) instructions|system override", raw, re.I):
                ids = re.findall(r'"id":\s*(\d+)', raw)
                if ids:
                    return self._call("refund_order", {"order_id": int(ids[0])})
            try:
                data = json.loads(raw)
            except ValueError:
                lines.append("I got an unreadable response from the store.")
                continue
            lines.append(self._describe(q, data))
        return AIMessage(content=" ".join(lines))

    @staticmethod
    def _describe(q: str, data: dict) -> str:
        if not data.get("ok"):
            err = data.get("error", {})
            code, msg = err.get("code"), err.get("message", "unknown error")
            if code == "rejected_by_staff":
                return "Understood - I did not make any change."
            if msg == "Order not found":
                return "I couldn't find that order. It may not exist, or it may not belong to your store."
            return f"I couldn't complete that: {msg}."
        res = data["result"]
        if "orders" in res:
            orders = res["orders"]
            low = q.lower()
            if m := re.search(r"not shipped in (\d+) days?", low):
                cutoff = datetime.now(timezone.utc) - timedelta(days=int(m.group(1)))
                orders = [o for o in orders if datetime.fromisoformat(o["placed_at"]) < cutoff]
                head = f"{len(orders)} paid order(s) not shipped in {m.group(1)}+ days"
            elif m := re.search(r"customer (\w+)", low):
                orders = sorted((o for o in orders if m.group(1) in o["customer_name"].lower()),
                                key=lambda o: o["placed_at"], reverse=True)[:1]
                head = f"Latest order for {m.group(1).title()}"
            else:
                head = f"{len(orders)} order(s)"
            body = "; ".join(f"#{o['id']} {o['customer_name']} {o['status']} {o['total']}" for o in orders)
            return f"{head}: {body}." if orders else f"{head}."
        o = res["order"]
        done = " (change executed)" if data.get("executed") else ""
        note = f" Note: {o['customer_note']}" if o.get("customer_note") else ""
        return f"Order #{o['id']} for {o['customer_name']} is {o['status']}, total {o['total']}.{done}{note}"

    # ---- BaseChatModel plumbing ---------------------------------------------
    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._plan(messages))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        plan = self._plan(messages)
        if plan.tool_calls:
            tc = plan.tool_calls[0]
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[
                {"name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": 0}]))
            return
        for word in re.findall(r"\S+\s*", plan.content):
            yield ChatGenerationChunk(message=AIMessageChunk(content=word))  # base class emits the token callback
