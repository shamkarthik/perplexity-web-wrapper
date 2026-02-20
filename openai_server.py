# openai_server.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from lib.perplexity import Client
from concurrent.futures import ThreadPoolExecutor
import asyncio, json, uuid, time

app = FastAPI()

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

# Cline agent system prompt injected into every request
CLINE_SYSTEM_PROMPT = """You are Cline, an expert software engineer and coding agent.

CRITICAL RULES — follow exactly:
1. When you need to read a file, respond ONLY with:
<read_file>
<path>path/to/file</path>
</read_file>

2. When you need to write/create a file, respond ONLY with:
<write_to_file>
<path>path/to/file</path>
<content>
full file content here
</content>
</write_to_file>

3. When you need to run a terminal command, respond ONLY with:
<execute_command>
<command>the command</command>
</execute_command>

4. When you need to search/replace code in a file, respond ONLY with:
<replace_in_file>
<path>path/to/file</path>
<diff>
<<<<<<< SEARCH
old code
=======
new code
>>>>>>> REPLACE
</diff>
</replace_in_file>

5. When you need to ask the user a question, respond ONLY with:
<ask_followup_question>
<question>your question</question>
</ask_followup_question>

6. When the task is complete, respond ONLY with:
<attempt_completion>
<result>description of what was done</result>
</attempt_completion>

7. NEVER add markdown, explanations, or text before/after a tool call block.
8. ONE tool call per response. Wait for result before next tool call.
9. Think step by step, use tools to gather info before making changes.
10. Always read a file before editing it."""


def resolve_mode(model: str) -> str:
    return MODE_MAP.get(model, "auto")


def content_to_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    parts.append("[image]")
                else:
                    parts.append(str(item))
        return " ".join(parts)
    return str(content)


def build_query(messages: list, tools: list = None) -> str:
    parts = []

    # Always inject Cline system prompt first
    parts.append(CLINE_SYSTEM_PROMPT)

    # Extract and append original system message if any
    for m in messages:
        role    = m.get("role", "")
        content = content_to_str(m.get("content", ""))
        if not content:
            continue
        if role == "system":
            parts.append(f"[Additional instructions: {content}]")

    # Add tool definitions as context
    if tools:
        tool_names = [t.get("function", {}).get("name", "") for t in tools if "function" in t]
        if tool_names:
            parts.append(f"[Available tools: {', '.join(tool_names)}]")

    # Add conversation history
    for m in messages:
        role    = m.get("role", "")
        content = content_to_str(m.get("content", ""))
        if not content:
            continue
        if role == "user":
            parts.append(f"<user_message>\n{content}\n</user_message>")
        elif role == "assistant":
            parts.append(f"<assistant_response>\n{content}\n</assistant_response>")
        elif role == "tool":
            parts.append(f"<tool_result tool_call_id='{m.get('tool_call_id', '')}'>\n{content}\n</tool_result>")

    return "\n\n".join(parts)


def extract_answer(result) -> str:
    if not isinstance(result, dict):
        return str(result)

    blocks = result.get("blocks", [])
    for block in blocks:
        mb = block.get("markdown_block")
        if mb and mb.get("answer"):
            return mb["answer"]

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

    return json.dumps(result)


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


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

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

    # ── Streaming ──
    if stream:
        def event_stream():
            words = answer.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"data: {make_chunk(chunk, model)}\n\n"
            yield f"data: {make_chunk('', model, finish=True)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ── Standard ──
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
            "prompt_tokens":     len(query.split()),
            "completion_tokens": len(answer.split()),
            "total_tokens":      len(query.split()) + len(answer.split())
        }
    })
