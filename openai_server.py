# openai_server.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from lib.perplexity import Client
from concurrent.futures import ThreadPoolExecutor
import asyncio, json, uuid, time

app = FastAPI()

# Load and normalize cookies
with open("perplexity_cookies.json") as f:
    cookies = json.load(f)

if isinstance(cookies, list):
    cookies = {c["name"]: c["value"] for c in cookies}

client = Client(cookies)
executor = ThreadPoolExecutor(max_workers=4)

MODE_MAP = {
    "auto":               "auto",
    "pro":                "pro",
    "reasoning":          "reasoning",
    "deep research":      "deep research",
    "deep_research":      "deep research",
    "claude-sonnet-4-6":  "pro",
    "claude-sonnet-4-5":  "pro",
    "claude-3-5-sonnet":  "pro",
    "claude-3-7-sonnet":  "pro",
    "claude":             "pro",
    "sonnet":             "pro",
    "gpt-4o":             "auto",
    "gpt-4":              "auto",
    "gpt-4-turbo":        "auto",
    "gpt-3.5-turbo":      "auto",
    "gpt-5":              "auto",
    "sonar":              "pro",
    "sonar-pro":          "pro",
    "coding":             "pro",
    "research":           "deep research",
    "writing":            "auto",
    "default":            "auto",
    "o1":                 "reasoning",
    "o3-mini":            "reasoning",
}


def resolve_mode(model: str) -> str:
    return MODE_MAP.get(model, "auto")


def extract_answer(result) -> str:
    if not isinstance(result, dict):
        return str(result)

    # 1. Best path: blocks -> markdown_block -> answer
    blocks = result.get("blocks", [])
    for block in blocks:
        mb = block.get("markdown_block")
        if mb and mb.get("answer"):
            return mb["answer"]

    # 2. Fallback: text steps -> FINAL step -> content -> answer (JSON string)
    text_steps = result.get("text", [])
    if isinstance(text_steps, list):
        for step in reversed(text_steps):
            if isinstance(step, dict) and step.get("step_type") == "FINAL":
                content = step.get("content", {})
                raw_answer = content.get("answer", "")
                if raw_answer:
                    try:
                        parsed = json.loads(raw_answer)
                        return parsed.get("answer", raw_answer)
                    except Exception:
                        return raw_answer

    # 3. Last resort
    return json.dumps(result)


def build_query(messages: list, tools: list = None) -> str:
    parts = []

    # Mention available tools so Perplexity is aware of context
    if tools:
        tool_names = [t.get("function", {}).get("name", "") for t in tools if "function" in t]
        if tool_names:
            parts.append(f"[Available tools: {', '.join(tool_names)}]")
            parts.append("[Use these tools by describing what to call and with what arguments.]")

    for m in messages:
        role    = m.get("role", "")
        content = m.get("content", "")
        if not content:
            continue
        if role == "system":
            parts.append(f"[System: {content}]")
        elif role == "user":
            parts.append(content)
        elif role == "assistant":
            parts.append(f"[Assistant: {content}]")
        elif role == "tool":
            parts.append(f"[Tool result ({m.get('tool_call_id', '')}): {content}]")

    return "\n".join(parts)


def make_chunk(content: str, model: str, finish: bool = False) -> str:
    return json.dumps({
        "id":      f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object":  "chat.completion.chunk",
        "created": int(time.time()),
        "model":   model,
        "choices": [{
            "index":         0,
            "delta":         {"content": content} if not finish else {},
            "finish_reason": "stop" if finish else None
        }]
    })


def timeout_response(model: str) -> dict:
    return {
        "id":      f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   model,
        "choices": [{
            "index": 0,
            "message": {
                "role":    "assistant",
                "content": "⚠️ Request timed out. Perplexity took too long to respond. Please try again."
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "message": "Perplexity OpenAI-compatible server is running"}


@app.get("/v1/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [
            {"id": "auto",             "object": "model", "owned_by": "perplexity"},
            {"id": "pro",              "object": "model", "owned_by": "perplexity"},
            {"id": "reasoning",        "object": "model", "owned_by": "perplexity"},
            {"id": "deep research",    "object": "model", "owned_by": "perplexity"},
            {"id": "claude-sonnet-4-6","object": "model", "owned_by": "perplexity"},
        ]
    })


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    messages = body.get("messages", [])
    model    = body.get("model", "auto")
    stream   = body.get("stream", False)
    tools    = body.get("tools", [])

    if not messages:
        return JSONResponse({"error": "messages field is required"}, status_code=400)

    query = build_query(messages, tools or None)
    mode  = resolve_mode(model)

    # Run blocking client.search in thread pool with timeout
    try:
        loop   = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                lambda: client.search(query, mode=mode)
            ),
            timeout=90.0
        )
    except asyncio.TimeoutError:
        return JSONResponse(timeout_response(model))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    answer = extract_answer(result)

    # ── Streaming response ──
    if stream:
        def event_stream():
            words = answer.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"data: {make_chunk(chunk, model)}\n\n"
            yield f"data: {make_chunk('', model, finish=True)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ── Standard JSON response ──
    return JSONResponse({
        "id":      f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   model,
        "choices": [{
            "index": 0,
            "message": {
                "role":    "assistant",
                "content": answer
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens":      len(query.split()),
            "completion_tokens":  len(answer.split()),
            "total_tokens":       len(query.split()) + len(answer.split())
        }
    })
