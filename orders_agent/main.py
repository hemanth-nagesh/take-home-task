"""Orders sub-agent (R2): official google/a2a-sdk server + A2A protocol."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI

# --- a2a-sdk server primitives ---
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCard
from a2a.helpers.proto_helpers import new_text_part, new_task
from a2a.types.a2a_pb2 import TaskState
from a2a.client.card_resolver import parse_agent_card

from common.guardrails import sanitize_data

log = logging.getLogger("orders_agent")

STORE_API_URL = os.environ.get("STORE_API_URL", "http://127.0.0.1:8001")
PUBLIC_URL = os.environ.get("ORDERS_AGENT_PUBLIC_URL", "http://127.0.0.1:8002")
STATUSES = ["PLACED", "PAID", "SHIPPED", "DELIVERED", "CANCELLED", "REFUNDED"]

# Agent card source of truth. The JSON document ships as a static asset inside
# the package (orders_agent/.well-known/agent-card.json) and is served verbatim
# at GET /.well-known/agent-card.json — the same path A2A clients discover it on.
# Override the file location with AGENT_CARD_PATH.
_DEFAULT_CARD_PATH = Path(__file__).resolve().parent / ".well-known" / "agent-card.json"
AGENT_CARD_PATH = Path(_DEFAULT_CARD_PATH)


# ── Error helpers ────────────────────────────────────────────────────────────

class AgentError(Exception):
    """Domain error raised by skill handlers; caught inside the executor."""
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


def _bearer_from_context(context: RequestContext) -> str:
    """Extract the raw Bearer token forwarded by the orchestrator.

    The SDK's DefaultServerCallContextBuilder places every HTTP request
    header in context.call_context.state['headers'] (lowercase keys).
    """
    headers: dict[str, str] = context.call_context.state.get("headers", {})
    auth = headers.get("authorization", "")
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AgentError("unauthorized", "Missing bearer token")
    return token.strip()


# ── Store API helpers ────────────────────────────────────────────────────────

async def _store_request(
    client: httpx.AsyncClient,
    token: str,
    method: str,
    path: str,
    **kw: Any,
) -> Any:
    """Call the Store API, forwarding the caller's Bearer token unchanged."""
    try:
        resp = await client.request(
            method, path,
            headers={"Authorization": f"Bearer {token}"},
            **kw,
        )
    except httpx.HTTPError as exc:
        raise AgentError("upstream_error", f"Store API unreachable: {type(exc).__name__}")

    if resp.status_code == 401:
        raise AgentError("unauthorized", "Token rejected by store")
    if resp.status_code == 403:
        raise AgentError("no_merchant", "Account is not associated with a merchant")
    if resp.status_code == 404:
        raise AgentError("not_found", "Order not found")
    if resp.status_code == 409:
        raise AgentError("conflict", resp.json().get("detail", "Invalid state transition"))
    if resp.status_code >= 400:
        raise AgentError("upstream_error", f"Store API error {resp.status_code}")
    return resp.json()


async def _require_merchant(client: httpx.AsyncClient, token: str) -> str:
    me = await _store_request(client, token, "GET", "/auth/me")
    merchant = me.get("merchant")
    if not merchant:
        raise AgentError("no_merchant", "Account is not associated with a merchant")
    return merchant


def _order_id_of(params: dict) -> int:
    oid = params.get("order_id")
    if type(oid) is not int:  # rejects booleans too
        raise AgentError("invalid_params", "order_id must be an integer")
    return oid


# ── Skill handlers ───────────────────────────────────────────────────────────

async def _list_orders(client: httpx.AsyncClient, token: str, params: dict) -> dict:
    status = params.get("status")
    if status is not None and status not in STATUSES:
        raise AgentError("invalid_params", f"status must be one of {STATUSES}")
    q = {"status": status} if status else None
    return {"orders": await _store_request(client, token, "GET", "/orders", params=q)}


async def _lookup_order(client: httpx.AsyncClient, token: str, params: dict) -> dict:
    body = await _store_request(
        client, token, "GET", "/orders/lookup",
        params={"id": _order_id_of(params)},
    )
    # Legacy store envelope: HTTP 200 does NOT always mean success
    if not isinstance(body, dict) or body.get("status") is not True or "order" not in body:
        raise AgentError("not_found", "Order not found")
    return {"order": body["order"]}


async def _mutate(
    client: httpx.AsyncClient, path: str, token: str, params: dict,
) -> dict:
    return {"order": await _store_request(client, token, "POST", path, json={"id": _order_id_of(params)})}


_HANDLERS = {
    "list_orders":  _list_orders,
    "lookup_order": _lookup_order,
    "refund_order": lambda c, t, p: _mutate(c, "/orders/refund", t, p),
    "cancel_order": lambda c, t, p: _mutate(c, "/orders/cancel", t, p),
}


def _parse_natural_language_intent(text: str) -> tuple[str | None, dict]:
    """Parse natural language prompts into a target skill and parameters.

    Enables direct natural-language interaction over A2A (e.g. from A2A CLI,
    testing tools, or notebooks) while structured callers pass JSON.
    """
    t = text.strip().lower()
    m_id = re.search(r"\b(?:order\s*#?|#)\s*(\d+)\b", t) or re.search(r"\b(\d{4,5})\b", t)
    order_id = int(m_id.group(1)) if m_id else None

    if "refund" in t:
        if order_id is not None:
            return "refund_order", {"order_id": order_id}
        raise AgentError("invalid_params", "Missing order id for refund")

    if "cancel" in t:
        if order_id is not None:
            return "cancel_order", {"order_id": order_id}
        raise AgentError("invalid_params", "Missing order id for cancellation")

    if any(k in t for k in ("look up", "lookup", "show order", "status of order", "find order", "get order")) or (order_id and "list" not in t):
        if order_id is not None:
            return "lookup_order", {"order_id": order_id}

    if any(k in t for k in ("list", "all orders", "orders")):
        for s in STATUSES:
            if s.lower() in t:
                return "list_orders", {"status": s}
        return "list_orders", {}

    return None, {}


# ── A2A AgentExecutor ────────────────────────────────────────────────────────

class OrdersAgentExecutor(AgentExecutor):
    """Processes A2A SendMessage requests using the official a2a-sdk.

    Supports both:
      1. Structured JSON: ``{"method": "<skill_id>", "params": {...}}``
      2. Natural language: e.g. ``"What is the status of order 1001?"``

    Execution model: **streaming Task**. The A2A spec lets an AgentExecutor
    either enqueue a single ``Message`` (immediate response) or run a ``Task``
    and stream ``TaskStatusUpdateEvent`` / ``TaskArtifactUpdateEvent`` events to
    a terminal state. We use the Task model so the agent advertises and uses
    ``capabilities.streaming`` — the JSON-RPC ``message/stream`` (SSE) transport
    is engaged when the orchestrator streams, while ``message/send``
    (non-streaming) still works by collecting the same events into the final
    completed Task.

    The skill result (or error) is carried as a text **artifact** so it is
    extractable from both a streamed ``artifact_update`` event and the final
    completed/failed ``Task`` snapshot:
        {"ok": true,  "result": {...}}   — on success
        {"ok": false, "error": {...}}    — on any domain/auth error
    """

    def __init__(self, store_http: httpx.AsyncClient) -> None:
        self._client = store_http

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id
        context_id = context.context_id
        updater = TaskUpdater(event_queue, task_id, context_id)

        # Enter task mode: the first event MUST be a Task, after which only
        # status/artifact update events are valid (enforced by ActiveTask).
        await event_queue.enqueue_event(
            new_task(task_id, context_id, TaskState.TASK_STATE_SUBMITTED,
                     history=[context.message] if context.message else None)
        )
        await updater.start_work()

        try:
            # ── 1. Parse incoming request (structured JSON or natural language) ─
            raw_text = context.get_user_input().strip()
            method: str | None = None
            params: dict = {}

            if raw_text.startswith("{") and raw_text.endswith("}"):
                try:
                    envelope = json.loads(raw_text)
                    if not isinstance(envelope, dict):
                        raise AgentError("invalid_request", "Expected a JSON object")
                    method = envelope.get("method")
                    params = envelope.get("params") or {}
                    if not isinstance(params, dict):
                        raise AgentError("invalid_params", "params must be an object")
                except json.JSONDecodeError:
                    raise AgentError("parse_error", "Request body must be valid JSON")
            else:
                method, params = _parse_natural_language_intent(raw_text)

            if not method or method not in _HANDLERS:
                raise AgentError("method_not_found", f"Unknown method or query {method or raw_text[:80]!r}")

            # ── 2. Auth — fail closed ──────────────────────────────────
            token = _bearer_from_context(context)
            await _require_merchant(self._client, token)   # R4c

            # ── 3. Dispatch skill ──────────────────────────────────────
            result = await _HANDLERS[method](self._client, token, params)
            safe_result = sanitize_data(result)            # R4a + R4b
            payload = json.dumps({"ok": True, "result": safe_result}, ensure_ascii=False)

            # ── 4. Stream the result as an artifact, then complete ──────
            await updater.add_artifact(parts=[new_text_part(payload)], name="result", last_chunk=True)
            await updater.complete()

        except AgentError as err:
            payload = json.dumps(
                {"ok": False, "error": {"code": err.code, "message": err.message}},
                ensure_ascii=False,
            )
            await updater.add_artifact(parts=[new_text_part(payload)], name="result", last_chunk=True)
            await updater.failed()

        except Exception:
            log.exception("Unexpected error in OrdersAgentExecutor.execute")
            payload = json.dumps(
                {"ok": False, "error": {"code": "internal_error", "message": "Internal agent error"}},
                ensure_ascii=False,
            )
            await updater.add_artifact(parts=[new_text_part(payload)], name="result", last_chunk=True)
            await updater.failed()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel transitions the in-flight task to CANCELED (best effort)."""
        try:
            await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()
        except Exception:
            log.warning("could not cancel task %s", context.task_id)


# ── A2A AgentCard (loaded from JSON) ─────────────────────────────────────────

def _card_dict(path: Path | None = None) -> dict[str, Any]:
    """Read the agent-card JSON file and apply the one runtime field."""
    card_path = path or AGENT_CARD_PATH
    data: dict[str, Any] = json.loads(card_path.read_text(encoding="utf-8"))
    if data.get("supportedInterfaces"):
        data["supportedInterfaces"][0]["url"] = f"{PUBLIC_URL}/rpc"
    return data


def render_agent_card(path: Path | None = None) -> dict[str, Any]:
    """Return the agent card as a plain dict for serving at the well-known URL. """
    return _card_dict(path)


def load_agent_card(path: Path | None = None) -> AgentCard:
    """Parse the agent-card JSON into an ``AgentCard`` proto for internal use.

    The request handler needs the proto to validate input modes and read
    capability flags (streaming, push notifications). The wire document seen by
    clients is served directly from the file by :func:`render_agent_card`, so
    the two never drift — both read the same source file.
    """
    return parse_agent_card(_card_dict(path))


# ── App factory ──────────────────────────────────────────────────────────────

def create_app(store_http: httpx.AsyncClient | None = None) -> FastAPI:
    """`store_http` is injectable so tests can route to an in-process store."""
    client = store_http or httpx.AsyncClient(base_url=STORE_API_URL, timeout=10.0)

    agent_card = load_agent_card()
    executor = OrdersAgentExecutor(client)
    task_store = InMemoryTaskStore()
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
        agent_card=agent_card,
    )

    app = FastAPI(title="ShopHub Orders Agent (A2A)")

    # ── Standard health check (outside A2A routing) ────────────────
    @app.get("/health")
    def health():
        return {"ok": True}

    # ── Agent card: served verbatim from the static JSON file ──────
    # GET /.well-known/agent-card.json — A2A discovery endpoint.
    # The file lives at orders_agent/.well-known/agent-card.json; only the
    # RPC URL is injected at request time (env-dependent).
    @app.get("/.well-known/agent-card.json", include_in_schema=False)
    async def get_agent_card():
        from fastapi.responses import JSONResponse
        return JSONResponse(render_agent_card())

    # Backward compatibility alias for v0.3 / legacy discovery path
    @app.get("/.well-known/agent.json", include_in_schema=False)
    async def legacy_agent_card():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/.well-known/agent-card.json")

    # ── Attach official A2A routes ─────────────────────────────────
    # • POST /rpc  — JSON-RPC 2.0 (send_message, etc.)
    # (agent card routes are served above directly from the static file)
    add_a2a_routes_to_fastapi(
        app,
        jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url="/rpc"),
    )

    return app


app = create_app()
