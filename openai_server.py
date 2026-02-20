# openai_server.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from lib.perplexity import Client
from concurrent.futures import ThreadPoolExecutor
import asyncio, json, uuid, time, re, base64, httpx

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

CLINE_SYSTEM_PROMPT = """You are Cline, a highly skilled software engineer with extensive knowledge in many programming languages, frameworks, design patterns, and best practices.

====

TOOL USE

You have access to a set of tools that are executed upon the user's approval. You can use one tool per message, and will receive the result of that tool use in the user's response. You use tools step-by-step to accomplish a given task, with each tool use informed by the result of the previous tool use.

# Tool Use Formatting

Tool use is formatted using XML-style tags. The tool name is enclosed in opening and closing tags, and each parameter is similarly enclosed within its own set of tags:

<tool_name>
<parameter1_name>value1</parameter1_name>
<parameter2_name>value2</parameter2_name>
</tool_name>

Always adhere to this format. Output ONE tool call per response — nothing else before or after it.

# Tools

## execute_command
Description: Execute a CLI command on the system.
Parameters:
- command: (required) The CLI command to execute.
- requires_approval: (required) true or false. Use false for safe/read-only commands. Use true for installs, deletes, destructive ops.
Usage:
<execute_command>
<command>your command here</command>
<requires_approval>false</requires_approval>
</execute_command>

## read_file
Description: Read the contents of a file at the specified path.
Parameters:
- path: (required) Path of file to read.
Usage:
<read_file>
<path>path/to/file</path>
</read_file>

## write_to_file
Description: Write content to a file at the specified path. Creates file if it doesn't exist. Always provide COMPLETE file content.
Parameters:
- path: (required) Path of file to write.
- content: (required) Complete file content — never truncate or use placeholders.
Usage:
<write_to_file>
<path>path/to/file</path>
<content>
full file content here
</content>
</write_to_file>

## replace_in_file
Description: Make targeted edits to specific parts of a file using SEARCH/REPLACE blocks.
Parameters:
- path: (required) Path of file to modify.
- diff: (required) One or more SEARCH/REPLACE blocks.
Usage:
<replace_in_file>
<path>path/to/file</path>
<diff>
<<<<<<< SEARCH
old code to find
=======
new replacement code
>>>>>>> REPLACE
</diff>
</replace_in_file>

## search_files
Description: Perform a regex search across files in a directory.
Parameters:
- path: (required) Directory to search in.
- regex: (required) Regex pattern to search for.
- file_pattern: (optional) Glob pattern to filter files (e.g. *.py).
Usage:
<search_files>
<path>.</path>
<regex>pattern</regex>
<file_pattern>*.py</file_pattern>
</search_files>

## list_files
Description: List files and directories within a path.
Parameters:
- path: (required) Directory to list.
- recursive: (optional) true or false.
Usage:
<list_files>
<path>.</path>
<recursive>false</recursive>
</list_files>

## list_code_definition_names
Description: List top-level code definitions (classes, functions, methods) in a directory.
Parameters:
- path: (required) Directory path.
Usage:
<list_code_definition_names>
<path>.</path>
</list_code_definition_names>

## plan_mode_respond
Description: Respond in Plan Mode with a detailed plan or response. NEVER leave response empty.
Parameters:
- response: (required) Your detailed plan or response. Must not be empty.
- options: (optional) Array of options for user to select.
Usage:
<plan_mode_respond>
<response>
Your detailed response or plan here. Must never be empty.
</response>
<options>
[]
</options>
</plan_mode_respond>

## ask_followup_question
Description: Ask the user a clarifying question when more information is needed.
Parameters:
- question: (required) A clear, specific question.
- options: (optional) Array of answer options.
Usage:
<ask_followup_question>
<question>Your question here?</question>
<options>["Option A", "Option B"]</options>
</ask_followup_question>

## attempt_completion
Description: Present the final result to the user when the task is complete.
Parameters:
- result: (required) Description of what was accomplished.
- command: (optional) CLI command to demo the result.
Usage:
<attempt_completion>
<result>
Description of completed task.
</result>
</attempt_completion>

====

CRITICAL RULES:
1. ONE tool call per response — NOTHING before or after it
2. NEVER wrap tool calls in markdown code blocks or backticks
3. NEVER add citation numbers [1][2][3] inside tool calls
4. ALWAYS include ALL required parameters
5. execute_command ALWAYS needs <requires_approval>false</requires_approval> (or true)
6. write_to_file ALWAYS needs <path> and complete <content>
7. plan_mode_respond ALWAYS needs non-empty <response>
8. Read files before editing them
9. Wait for tool result before issuing next tool call"""


def resolve_mode(model: str) -> str:
    return MODE_MAP.get(model, "auto")


async def fetch_image_as_base64(url: str) -> tuple[str, str]:
    """Download image from URL and return (base64_data, mime_type)."""
    try:
        async with httpx.AsyncClient(timeout=10) as hc:
            resp = await hc.get(url)
            mime = resp.headers.get("content-type", "image/png").split(";")[0]
            b64  = base64.b64encode(resp.content).decode()
            return b64, mime
    except Exception:
        return None, None


async def content_to_str_async(content, include_images: bool = True) -> tuple[str, list]:
    """
    Convert content (str or list) to (text_string, image_urls_list).
    Handles multimodal OpenAI content format with text + image_url blocks.
    """
    if isinstance(content, str):
        return content, []

    if isinstance(content, list):
        text_parts  = []
        image_items = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                t = item.get("type", "")
                if t == "text":
                    text_parts.append(item.get("text", ""))
                elif t == "image_url":
                    img = item.get("image_url", {})
                    url = img.get("url", "") if isinstance(img, dict) else img

                    if url.startswith("data:"):
                        # Already base64 — extract mime and data
                        header, data = url.split(",", 1)
                        mime = header.split(":")[1].split(";")[0]
                        image_items.append({"type": "base64", "mime": mime, "data": data})
                        text_parts.append("[image attached]")
                    elif url.startswith("http"):
                        if include_images:
                            b64, mime = await fetch_image_as_base64(url)
                            if b64:
                                image_items.append({"type": "base64", "mime": mime, "data": b64})
                        text_parts.append(f"[image: {url}]")

        return " ".join(text_parts), image_items

    return str(content), []


def content_to_str(content) -> str:
    """Sync version — text only, no images."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("type", "")
                if t == "text":
                    parts.append(item.get("text", ""))
                elif t == "image_url":
                    parts.append("[image attached]")
        return " ".join(parts)
    return str(content)


def fix_cline_tool_calls(answer: str, messages: list) -> str:
    """Fix malformed Cline XML tool calls from Perplexity responses."""

    # Strip markdown fences wrapping tool calls
    answer = re.sub(r"```(?:xml)?\s*\n?(<[a-z_]+>)", r"\1", answer)
    answer = re.sub(r"(</[a-z_]+>)\s*\n?```", r"\1", answer)

    # Strip citation numbers [1][2]
    answer = re.sub(r"\[\d+\]", "", answer)

    # Get last user message for context
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
            r"(\S+\.(?:md|txt|py|json|yaml|yml|js|ts|dart|sh|env|toml|cfg|html|css))",
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
        answer = answer.replace(
            "</execute_command>",
            "<requires_approval>false</requires_approval>\n</execute_command>"
        )

    # Fix search_files missing <path>
    if "<search_files>" in answer and "<path>" not in answer:
        answer = answer.replace("<search_files>", "<search_files>\n<path>.</path>")

    # Fix list_files missing <path>
    if "<list_files>" in answer and "<path>" not in answer:
        answer = answer.replace("<list_files>", "<list_files>\n<path>.</path>")

    # Fix plan_mode_respond with empty or missing <response>
    if "<plan_mode_respond>" in answer:
        if "<response>" not in answer:
            text = re.sub(r"<[^>]+>", "", answer).strip()
            if not text:
                text = "I'll analyze the task and create a detailed plan."
            answer = answer.replace(
                "<plan_mode_respond>",
                f"<plan_mode_respond>\n<response>{text}</response>"
            )
        # Fix empty response tag
        answer = re.sub(
            r"<response>\s*</response>",
            "<response>I'll analyze the task and create a detailed plan.</response>",
            answer
        )

    # Fix ask_followup_question missing <question>
    if "<ask_followup_question>" in answer and "<question>" not in answer:
        text = re.sub(r"<[^>]+>", "", answer).strip()
        if not text:
            text = "Could you provide more details about what you'd like me to do?"
        answer = answer.replace(
            "<ask_followup_question>",
            f"<ask_followup_question>\n<question>{text}</question>"
        )

    # Fix attempt_completion missing <result>
    if "<attempt_completion>" in answer and "<result>" not in answer:
        text = re.sub(r"<[^>]+>", "", answer).strip()
        if not text:
            text = "Task completed successfully."
        answer = answer.replace(
            "<attempt_completion>",
            f"<attempt_completion>\n<result>{text}</result>"
        )

    return answer.strip()


async def build_query_async(messages: list, tools: list = None) -> str:
    """Build query string, extracting text + handling images from messages."""
    parts = []
    all_images = []

    parts.append(CLINE_SYSTEM_PROMPT)

    # System messages
    for m in messages:
        if m.get("role") == "system":
            text, _ = await content_to_str_async(m.get("content", ""), include_images=False)
            if text:
                parts.append(f"[Additional instructions: {text}]")

    # Tool names
    if tools:
        tool_names = [t.get("function", {}).get("name", "") for t in tools if "function" in t]
        if tool_names:
            parts.append(f"[Available tools: {', '.join(tool_names)}]")

    # Conversation messages
    for m in messages:
        role = m.get("role", "")
        if role == "system":
            continue
        text, images = await content_to_str_async(m.get("content", ""))
        all_images.extend(images)
        if not text and not images:
            continue
        if role == "user":
            parts.append(f"<user_message>\n{text}\n</user_message>")
        elif role == "assistant":
            parts.append(f"<assistant_response>\n{text}\n</assistant_response>")
        elif role == "tool":
            parts.append(f"<tool_result tool_call_id='{m.get('tool_call_id', '')}'>\n{text}\n</tool_result>")

    query = "\n\n".join(parts)

    # Append image descriptions if any
    if all_images:
        query += f"\n\n[Note: {len(all_images)} image(s) attached by user. Analyze and describe them as part of your response.]"

    return query


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

    # Build query asynchronously (handles image fetching)
    query = await build_query_async(messages, tools or None)
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
