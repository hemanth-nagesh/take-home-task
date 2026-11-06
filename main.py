"""ShopHub orchestrator HTTP API: SSE chat + human-in-the-loop approval endpoints.

The LangGraph agent, the A2A sub-agent client and the checkpointer factory live
in ``orchestrator/main.py``. This module is only the transport layer — FastAPI
endpoints, request/response models, authentication, thread routing and the SSE
stream that carries tokens and tool events to the client.

Run:  uvicorn main:app --port 8000
"""
import hashlib
import json
import logging
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from common.guardrails import redact_pii
from orchestrator.llm import get_chat_model
import orchestrator.main as agent

log = logging.getLogger("orchestrator.http")


# --------------------------------------------------------------------------- request models
class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")


class DecisionIn(BaseModel):
    order_id: int | None = None  # optional guard: must match the pending proposal


# --------------------------------------------------------------------------- SSE helpers
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict)) if isinstance(content, list) else ""


def _pending_proposal(snapshot: Any) -> dict | None:
    return next((i.value for t in snapshot.tasks for i in t.interrupts), None)


# --------------------------------------------------------------------------- app factory
def create_app(*, model: BaseChatModel | None = None, checkpointer: Any = None,
               store_http: httpx.AsyncClient | None = None,
               agent_http: httpx.AsyncClient | None = None) -> FastAPI:
    """Dependencies are injectable so tests run fully in-process.

    Config constants (STORE_API_URL, ORDERS_AGENT_URL) are read from the agent
    module at call time so monkeypatching ``orchestrator.main`` in tests takes
    effect here too.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with AsyncExitStack() as stack:
            saver = checkpointer or await agent.make_checkpointer(stack)
            app.state.store = store_http or httpx.AsyncClient(base_url=agent.STORE_API_URL, timeout=10.0)
            ah = agent_http or httpx.AsyncClient(timeout=20.0)
            for own, c in ((store_http, app.state.store), (agent_http, ah)):
                if own is None:
                    stack.push_async_callback(c.aclose)
            orders_agent = agent.OrdersAgent(agent.ORDERS_AGENT_URL, ah)
            app.state.graph = agent.build_graph(model or get_chat_model(), orders_agent, saver)
            try:
                await orders_agent.ensure()
            except Exception as exc:  # agent may still be starting; retried lazily
                log.warning("agent card not available yet: %s", type(exc).__name__)
            yield

    app = FastAPI(title="ShopHub orchestrator", lifespan=lifespan)

    async def authenticate(authorization: str | None) -> agent.Caller:
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise HTTPException(401, "Missing bearer token")
        token = token.strip()
        try:
            resp = await app.state.store.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        except httpx.HTTPError:
            raise HTTPException(502, "Store API unreachable")
        if resp.status_code == 401:
            raise HTTPException(401, "Invalid token")
        if resp.status_code != 200:
            raise HTTPException(502, "Store API error")
        me = resp.json()
        if not me.get("merchant"):                      # FAIL CLOSED (R4c)
            raise HTTPException(403, "This account is not associated with a merchant")
        return agent.Caller(token=token, user=me["user"], merchant=me["merchant"])

    def thread_config(caller: agent.Caller, conversation_id: str) -> dict:
        # Threads are namespaced by (merchant, user): knowing someone's conversation id is useless.
        tid = hashlib.sha256(f"{caller.merchant}\x00{caller.user}\x00{conversation_id}".encode()).hexdigest()[:40]
        return {"configurable": {"thread_id": tid}, "recursion_limit": 16}   # NB: no token in here

    async def stream_turn(caller: agent.Caller, conversation_id: str, graph_input: Any, config: dict) -> AsyncIterator[str]:
        agent._caller.set(caller)
        graph = app.state.graph
        try:
            yield _sse("conversation", {"conversation_id": conversation_id})
            async for mode, payload in graph.astream(graph_input, config, stream_mode=["messages", "custom"]):
                if mode == "messages":
                    chunk, meta = payload
                    if meta.get("langgraph_node") == "assistant" and isinstance(chunk, AIMessageChunk):
                        if text := _text(chunk.content):
                            yield _sse("token", {"text": text})
                elif mode == "custom" and isinstance(payload, dict) and "event" in payload:
                    yield _sse(payload["event"], {k: v for k, v in payload.items() if k != "event"})
            proposal = _pending_proposal(await graph.aget_state(config))
            if proposal:
                yield _sse("proposal", proposal)
            yield _sse("done", {"pending_approval": proposal is not None})
        except Exception:
            log.exception("turn failed")
            yield _sse("error", {"message": "Internal error while processing the turn"})

    def sse_response(gen: AsyncIterator[str]) -> StreamingResponse:
        return StreamingResponse(gen, media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/chat")
    async def chat(body: ChatIn, authorization: str | None = Header(default=None)):
        caller = await authenticate(authorization)
        cid = body.conversation_id or uuid.uuid4().hex
        config = thread_config(caller, cid)
        if (await app.state.graph.aget_state(config)).next:
            raise HTTPException(409, "A proposal is awaiting your decision: approve or reject it first")
        msg = {"messages": [HumanMessage(content=redact_pii(body.message))]}      # R4b at ingress
        return sse_response(stream_turn(caller, cid, msg, config))

    async def decide(cid: str, approve: bool, body: DecisionIn, authorization: str | None):
        caller = await authenticate(authorization)
        config = thread_config(caller, cid)
        proposal = _pending_proposal(await app.state.graph.aget_state(config))
        if proposal is None:
            raise HTTPException(409, "No proposal is pending for this conversation")
        if body.order_id is not None and body.order_id != proposal["order_id"]:
            raise HTTPException(409, "order_id does not match the pending proposal")
        return sse_response(stream_turn(caller, cid, Command(resume={"approve": approve}), config))

    @app.post("/chat/{conversation_id}/approve")
    async def approve(conversation_id: str, body: DecisionIn | None = None,
                      authorization: str | None = Header(default=None)):
        return await decide(conversation_id, True, body or DecisionIn(), authorization)

    @app.post("/chat/{conversation_id}/reject")
    async def reject(conversation_id: str, body: DecisionIn | None = None,
                     authorization: str | None = Header(default=None)):
        return await decide(conversation_id, False, body or DecisionIn(), authorization)

    return app


app = create_app()
