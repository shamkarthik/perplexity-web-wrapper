# openai_server.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from lib.perplexity import Client
from concurrent.futures import ThreadPoolExecutor
import asyncio, json, uuid, time, re

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

# Exact Cline system prompt based on official Cline source
CLINE_SYSTEM_PROMPT = """You are Cline, a highly skilled software engineer with extensive knowledge in many programming languages, frameworks, design patterns, and best practices.

====

TOOL USE

You have access to a set of tools that are executed upon the user's approval. You can use one tool per message, and will receive the result of that tool use in the user's response. You use tools step-by-step to accomplish a given task, with each tool use informed by the result of the previous tool use.

# Tool Use Formatting

Tool use is formatted using XML-style tags. The tool name is enclosed in opening and closing tags, and each parameter is similarly enclosed within its own set of tags. Here's the structure:

<tool_name>
<parameter1_name>value1</parameter1_name>
<parameter2_name>value2</parameter2_name>
</tool_name>

Always adhere to this format for the tool use to ensure proper parsing and execution.

# Tools

## execute_command
Description: Execute a CLI command on the system.
Parameters:
- command: (required) The CLI command to execute.
- requires_approval: (required) A boolean (true or false) indicating whether this command requires explicit user approval. Set to 'false' for safe read-only operations. Set to 'true' for installs, deletes, or destructive operations.
Usage:
<execute_command>
<command>your command here</command>
<requires_approval>false</requires_approval>
</execute_command>

## read_file
Description: Read the contents of a file at the specified path.
Parameters:
- path: (required) The path of the file to read.
Usage:
<read_file>
<path>path/to/file</path>
</read_file>

## write_to_file
Description: Write content to a file. Creates file if it doesn't exist.
Parameters:
- path: (required) The path of the file to write to.
- content: (required) The COMPLETE file content — never truncate.
Usage:
<write_to_file>
<path>path/to/file</path>
<content>
full file content here
</content>
</write_to_file>

## replace_in_file
Description: Make targeted edits to specific parts of an existing file.
Parameters:
- path: (required) The path of the file to modify.
- diff: (required) SEARCH/REPLACE blocks.
Usage:
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

## search_files
Description: Regex search across files in a directory.
Parameters:
- path: (required) Directory to search in.
- regex: (required) Regular expression pattern.
- file_pattern: (optional) Glob pattern to filter files.
Usage:
<search_files>
<path>.</path>
<regex>def .*</regex>
<file_pattern>*.py</file_pattern>
</search_files>

## list_files
Description: List files and directories within a specified directory.
Parameters:
- path: (required) The directory to list.
- recursive: (optional) true or false.
Usage:
<list_files>
<path>.</path>
<recursive>false</recursive>
</list_files>

## list_code_definition_names
Description: List classes, functions, methods in source files at the top level of a directory.
Parameters:
- path: (required) Directory path.
Usage:
<list_code_definition_names>
<path>.</path>
</list_code_definition_names>

## ask_followup_question
Description: Ask the user a question to gather additional information.
Parameters:
- question: (required) A clear, specific question.
Usage:
<ask_followup_question>
<question>Your question here</question>
</ask_followup_question>

## attempt_completion
Description: Present the final result once the task is complete.
Parameters:
- result: (required) Final description of what was done.
- command: (optional) CLI command to demo the result.
Usage:
<attempt_completion>
<result>
Your final result description here
</result>
</attempt_completion>

====

CRITICAL RULES:
1. Output ONE tool call per response — NOTHING else before or after it
2. NEVER wrap tool calls in markdown code blocks or backticks
3. NEVER add citation numbers like [1][2][3] inside tool calls
4. ALWAYS include ALL required parameters — especially <path> and <requires_approval>
5. Use <thinking></thinking> tags internally to reason, but output only the tool call
6. Read a file before editing it
7. Wait for tool result before next tool call
8. For execute_command, ALWAYS include <requires_approval>false</requires_approval> for safe commands"""


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


def fix_cline_tool_calls(answer: str, messages: list) -> str:
    """Post-process Perplexity response to fix malformed Cline XML tool calls."""

    # Strip markdown fences around tool calls
    answer = re.sub(r"```(?:xml)?\s*\n?(<[a-z_]+>)", r"\1", answer)
    answer = re.sub(r"(</[a-z_]+>)\s*\n?```", r"\1", answer)

    # Strip citation numbers [1][2] that break XML
    answer = re.sub(r"\[\d+\]", "", answer)

    # Get last user message for filename inference
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = content_to_str(m.get("content", ""))
            break

    def infer_filename(fallback="output.md") -> str:
        patterns = [
            r"create\s+(?:a\s+)?(\S+\.\w+)",
            r"write\s+(?:to\s+)?(\S+\.\w+)",
            r"make\s+(?:a\s+)?(\S+\.\w+)",
            r"file\s+(?:called\s+|named\s+)?(\S+\.\w+)",
            r"(\S+\.(?:md|txt|py|json|yaml|yml|js|ts|dart|sh|env|toml|cfg))",
        ]
        for pattern in patterns:
            match = re.search(pattern, last_user_msg, re.IGNORECASE)
            if match:
                return match.group(1)
        return fallback

    # Fix write_to_file missing <path>
    if "<write_to_file>" in answer and "<path>" not in answer:
        answer = answer.replace(
            "<write_to_file>",
            f"<write_to_file>\n<path>{infer_filename('output.md')}</path>"
        )

    # Fix read_file missing <path>
    if "<read_file>" in answer and "<path>" not in answer:
        answer = answer.replace(
            "<read_file>",
            "<read_file>\n<path>openai_server.py</path>"
        )

    # Fix replace_in_file missing <path>
    if "<replace_in_file>" in answer and "<path>" not in answer:
        answer = answer.replace(
            "<replace_in_file>",
            f"<replace_in_file>\n<path>{infer_filename('openai_server.py')}</path>"
        )

    # Fix execute_command missing <requires_approval>
    if "<execute_command>" in answer and "<requires_approval>" not in answer:
        # Inject before closing tag
        answer = answer.replace(
            "</execute_command>",
            "<requires_approval>false</requires_approval>\n</execute_command>"
        )

    # Fix search_files missing <path>
    if "<search_files>" in answer and "<path>" not in answer:
        answer = answer.replace(
            "<search_files>",
            "<search_files>\n<path>.</path>"
        )

    # Fix list_files missing <path>
    if "<list_files>" in answer and "<path>" not in answer:
        answer = answer.replace(
            "<list_files>",
            "<list_files>\n<path>.</path>"
        )

    return answer.strip()


def build_query(messages: list, tools: list = None) -> str:
    parts = []

    parts.append(CLINE_SYSTEM_PROMPT)

    for m in messages:
        if m.get("role") == "system":
            content = content_to_str(m.get("content", ""))
            if content:
                parts.append(f"[Additional instructions: {content}]")

    if tools:
        tool_names = [t.get("function", {}).get("name", "") for t in tools if "function" in t]
        if tool_names:
            parts.append(f"[Available tools: {', '.join(tool_names)}]")

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
                "content": "⚠️ Request timed out. Perplexity took too long. Please try again."
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
    answer = fix_cline_tool_calls(answer, messages)

    if stream:
        def event_stream():
            words = answer.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"data: {make_chunk(chunk, model)}\n\n"
            yield f"data: {make_chunk('', model, finish=True)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

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
