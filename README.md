# ShopHub Support Assistant

A multi-tenant, tool-using AI support agent for ShopHub merchant staff. Built with FastAPI, LangGraph, Google Agent2Agent (A2A) protocol, JSON-RPC 2.0, and PostgreSQL.

## Architecture

```
Client (chat_client.py / curl)
           |  Bearer token (forwarded, never persisted)
           v
   Orchestrator (:8000)
     ├── Ingress: PII redaction + Auth verification (fail closed)
     ├── LangGraph Agent (assistant -> tools -> approval)
     └── Checkpointer (PostgreSQL / SQLite)
           |
           | JSON-RPC 2.0 (Google A2A agent card discovery)
           v
   Orders Sub-Agent (:8002)
     ├── Guardrails (PII scrub, injection neutralisation)
     └── JSON-RPC Tool Executor
           |
           | HTTP
           v
   Store API (:8001) [Mock e-commerce backend]
```

## Quick Start with Docker (R10)

Start the database, mock store API, orders sub-agent, and orchestrator with one command:

```bash
docker compose up --build
```

Services exposed:
- **Orchestrator**: `http://localhost:8000` (`/chat`, `/health`)
- **Orders Sub-Agent**: `http://localhost:8002` (`/.well-known/agent-card.json`, `/.well-known/agent.json`, `/rpc`, `/health`)
- **Store API**: `http://localhost:8001` (`/orders`, `/auth/me`, `/health`)
- **Postgres**: `localhost:5432` (`shophub/shophub`)

## Running Locally

### 1. Install Dependencies

Using `uv` :
```bash
uv sync
```


### 2. Start Services Locally

In separate terminal windows:
```bash
# Terminal 1: Store API
uvicorn store_api.main:app --port 8001

# Terminal 2: Orders Sub-Agent
STORE_API_URL=http://127.0.0.1:8001 ORDERS_AGENT_PUBLIC_URL=http://127.0.0.1:8002 uvicorn orders_agent.main:app --port 8002

# Terminal 3: Orchestrator (defaults to offline FakeChatModel)
STORE_API_URL=http://127.0.0.1:8001 ORDERS_AGENT_URL=http://127.0.0.1:8002 uvicorn main:app --port 8000
```

## LLM Configuration

The project supports offline fake LLM mode (default, no API keys or internet needed) and local Ollama:

- **Fake LLM (default, offline)**:
  ```bash
  LLM_PROVIDER=fake uvicorn orchestrator.main:app --port 8000
  ```
- **Ollama**:
  ```bash
  LLM_PROVIDER=ollama LLM_MODEL=gemma3:latest OLLAMA_BASE_URL=http://localhost:11434 uvicorn orchestrator.main:app --port 8000
  ```

## Running Tests (R8)

All tests run fully offline without any API keys or network access:

```bash
pytest -v
```

## Running the Evaluation Harness (R9)

Runs 7 scripted multi-turn end-to-end evaluation cases reporting per-case PASS/FAIL results:

```bash
python eval.py
```

To run against a local Ollama model:
```bash
LLM_PROVIDER=ollama python eval.py
```

### Evaluation Cases Covered:
1. **C1**: Tool routing and execution (`lookup_order` returns correct status)
2. **C2**: Filtered listing (`list_orders` excludes foreign merchants and unshipped orders)
3. **C3**: Human-in-the-loop proposal generation and approval execution
4. **C4**: Adversarial prompt injection in customer notes neutralised
5. **C5**: Multi-tenant isolation (cross-merchant lookups refused)
6. **C6**: Fail-closed auth (valid token with null merchant rejected with 403)
7. **C7**: Human-in-the-loop proposal rejection (no mutations made)

## Interacting with the Streaming Chat (R6)

A streaming CLI client is provided:

```bash
# Ask about an order
python chat_client.py chat "What is the status of order 1001?" --conv demo

# Propose a refund
python chat_client.py chat "Refund order 1001" --conv demo

# Approve the pending proposal
python chat_client.py approve --conv demo

# Or reject
python chat_client.py reject --conv demo
```

Development tokens:
- `tok-acme-staff` (merchant: acme)
- `tok-zenith-staff` (merchant: zenith)
- `tok-orphan` (merchant: None — test fail-closed)
