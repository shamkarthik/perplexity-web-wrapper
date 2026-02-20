# openai_server.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from lib.perplexity import Client
from concurrent.futures import ThreadPoolExecutor
import asyncio, json, uuid, time, re, base64, httpx, os, tempfile, traceback

app = FastAPI()

with open("perplexity_cookies.json") as f:
    cookies = json.load(f)

if isinstance(cookies, list):
    cookies = {c["name"]: c["value"] for c in cookies}

client = Client(cookies)
executor = ThreadPoolExecutor(max_workers=4)

TMP_IMAGE_DIR = os.path.join(os.environ.get("TEMP", tempfile.gettempdir()), "cline_images")
os.makedirs(TMP_IMAGE_DIR, exist_ok=True)

# ─── Mode + Model Maps ────────────────────────────────────────────────────────

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
    "gpt-4o":             "pro",
    "gpt-4":              "pro",
    "gpt-4-turbo":        "pro",
    "gpt-3.5-turbo":      "auto",
    "gpt-5":              "pro",
    "sonar":              "pro",
    "sonar-pro":          "pro",
    "coding":             "pro",
    "research":           "deep research",
    "writing":            "auto",
    "default":            "pro",
    "o1":                 "reasoning",
    "o3-mini":            "reasoning",
    "r1":                 "reasoning",
}

WRAPPER_MODEL_MAP = {
    "claude-sonnet-4-6":  "claude 3.7 sonnet",
    "claude-sonnet-4-5":  "claude 3.7 sonnet",
    "claude-3-7-sonnet":  "claude 3.7 sonnet",
    "claude-3-5-sonnet":  "claude 3.7 sonnet",
    "claude":             "claude 3.7 sonnet",
    "sonnet":             "claude 3.7 sonnet",
    "pro":                "claude 3.7 sonnet",
    "default":            "claude 3.7 sonnet",
    "coding":             "claude 3.7 sonnet",
    "gpt-4o":             "gpt-4o",
    "gpt-4":              "gpt-4o",
    "gpt-4-turbo":        "gpt-4o",
    "gpt-5":              "gpt-4o",
    "gpt-4.5":            "gpt-4.5",
    "sonar":              "sonar",
    "sonar-pro":          "sonar",
    "grok-2":             "grok-2",
    "gemini":             "gemini 2.0 flash",
    "o3-mini":            "o3-mini",
    "r1":                 "r1",
    "auto":               None,
    "deep research":      None,
    "deep_research":      None,
    "reasoning":          None,
    "writing":            None,
    "research":           None,
}

MODE_ALLOWED_MODELS = {
    "auto":          [None],
    "pro":           [None, "sonar", "gpt-4.5", "gpt-4o", "claude 3.7 sonnet", "gemini 2.0 flash", "grok-2"],
    "reasoning":     [None, "r1", "o3-mini", "claude 3.7 sonnet"],
    "deep research": [None],
}

KNOWN_TOOLS = [
    "<execute_command>", "<read_file>", "<write_to_file>", "<replace_in_file>",
    "<search_files>", "<list_files>", "<list_code_definition_names>",
    "<plan_mode_respond>", "<ask_followup_question>", "<attempt_completion>",
]


def resolve_mode(model: str) -> str:
    return MODE_MAP.get(model, "pro")


def resolve_wrapper_model(model: str, mode: str):
    mapped  = WRAPPER_MODEL_MAP.get(model, "claude 3.7 sonnet")
    allowed = MODE_ALLOWED_MODELS.get(mode, [None])
    return mapped if mapped in allowed else None


# ─── System Prompt ────────────────────────────────────────────────────────────

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
- requires_approval: (required) true or false. false for safe/read-only, true for installs/deletes/destructive.
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
Description: Write content to a file. Creates file if it doesn't exist. Always provide COMPLETE file content.
Parameters:
- path: (required) Path of file to write.
- content: (required) Complete file content — never truncate.
Usage:
<write_to_file>
<path>path/to/file</path>
<content>
full file content here
</content>
</write_to_file>

## replace_in_file
Description: Make targeted edits to specific parts of a file.
Parameters:
- path: (required) Path of file to modify.
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
- regex: (required) Regex pattern.
- file_pattern: (optional) Glob pattern (e.g. *.py).
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
Description: List top-level code definitions in a directory.
Parameters:
- path: (required) Directory path.
Usage:
<list_code_definition_names>
<path>.</path>
</list_code_definition_names>

## ask_followup_question
Description: Ask the user a clarifying question when you need more information.
Parameters:
- question: (required) A clear, specific question.
- options: (optional) Array of answer options.
Usage:
<ask_followup_question>
<question>Your question here?</question>
<options>["Option A", "Option B"]</options>
</ask_followup_question>

## attempt_completion
Description: Present the final result ONLY when all work is fully done and verified.
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
5. execute_command ALWAYS needs <requires_approval>
6. write_to_file ALWAYS needs <path> and complete <content>
7. You are ALWAYS in Act mode — NEVER use plan_mode_respond
8. NEVER describe what you are going to do — just DO it with a tool call
9. NEVER use attempt_completion until ALL actual work is fully done and verified
10. Read files before editing them
11. Wait for tool result before next tool call
12. Your FIRST response to any task must be a concrete tool call — not a description or plan
13. If a file is NOT FOUND, do NOT retry the same path — search for it first using execute_command
14. If images are provided, describe them inside attempt_completion result"""


# ─── Image Extraction (latest user message only) ─────────────────────────────

def extract_files_from_last_user_message(messages: list) -> dict:
    """Extract base64 images ONLY from the most recent user message."""
    last_user_msg = None
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = m
            break

    if not last_user_msg:
        return {}

    content = last_user_msg.get("content", "")
    if not isinstance(content, list):
        return {}

    files = {}
    idx = 0
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "image_url":
            continue
        img  = item.get("image_url", {})
        url  = img.get("url", "") if isinstance(img, dict) else img
        if not url.startswith("data:"):
            continue
        header, b64_data = url.split(",", 1)
        mime     = header.split(":")[1].split(";")[0]
        ext      = mime.split("/")[-1].replace("jpeg", "jpg")
        filename = f"cline_image_{idx}.{ext}"
        try:
            files[filename] = base64.b64decode(b64_data)
            print(f"[Image] Extracted {filename} ({len(files[filename])} bytes)")
        except Exception as e:
            print(f"[Image extract failed] {e}")
        idx += 1

    return files


# ─── Content Processing ───────────────────────────────────────────────────────

async def fetch_remote_image(url: str) -> tuple[str, str] | tuple[None, None]:
    try:
        async with httpx.AsyncClient(timeout=10) as hc:
            r    = await hc.get(url)
            mime = r.headers.get("content-type", "image/png").split(";")[0]
            return base64.b64encode(r.content).decode(), mime
    except Exception as e:
        print(f"[Image fetch failed] {e}")
        return None, None


async def save_image_locally(b64_data: str, mime: str) -> str | None:
    try:
        ext      = mime.split("/")[-1].replace("jpeg", "jpg")
        filename = os.path.join(TMP_IMAGE_DIR, f"img_{uuid.uuid4().hex[:8]}.{ext}")
        with open(filename, "wb") as f:
            f.write(base64.b64decode(b64_data))
        return filename
    except Exception as e:
        print(f"[Image save failed] {e}")
        return None


async def process_message_content(content) -> tuple[str, list[str]]:
    """Process content → (text, [image_paths])."""
    if isinstance(content, str):
        return content, []
    if isinstance(content, list):
        text_parts  = []
        image_paths = []
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
                        header, b64_data = url.split(",", 1)
                        mime       = header.split(":")[1].split(";")[0]
                        local_path = await save_image_locally(b64_data, mime)
                        if local_path:
                            image_paths.append(local_path)
                            text_parts.append(f"[IMAGE: {local_path}]")
                        else:
                            text_parts.append("[image could not be saved]")
                    elif url.startswith("http"):
                        b64_data, mime = await fetch_remote_image(url)
                        if b64_data:
                            local_path = await save_image_locally(b64_data, mime)
                            if local_path:
                                image_paths.append(local_path)
                                text_parts.append(f"[IMAGE: {local_path}]")
                            else:
                                text_parts.append(f"[image url: {url}]")
                        else:
                            text_parts.append(f"[image url: {url}]")
        return " ".join(text_parts), image_paths
    return str(content), []


def content_to_str(content) -> str:
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
                    parts.append("[image]")
        return " ".join(parts)
    return str(content)


# ─── Tool Result Interceptor ──────────────────────────────────────────────────

def fix_tool_result_in_messages(messages: list) -> list:
    """
    Scan tool result messages. If a 'File not found' error exists,
    inject corrective hint with the right path to search.
    """
    fixed = []
    for m in messages:
        if m.get("role") == "tool":
            content = m.get("content", "")
            if isinstance(content, str) and "file not found" in content.lower():
                bad_path_match = re.search(r"not found[:\s]+(.+?)(?:\n|$)", content, re.IGNORECASE)
                bad_path = bad_path_match.group(1).strip() if bad_path_match else "unknown path"
                corrected = (
                    f"{content}\n\n"
                    f"[SYSTEM NOTE: '{bad_path}' does not exist. "
                    f"Do NOT retry this path. "
                    f"Find the correct path with: "
                    f"powershell \"Get-ChildItem -Path $env:APPDATA -Recurse "
                    f"-Filter cline_mcp_settings.json -ErrorAction SilentlyContinue "
                    f"| Select-Object FullName\" "
                    f"OR check: %APPDATA%\\Code\\User\\globalStorage\\"
                    f"saoudrizwan.claude-dev\\settings\\cline_mcp_settings.json]"
                )
                fixed.append({**m, "content": corrected})
                continue
        fixed.append(m)
    return fixed


# ─── Tool Call Fixer ──────────────────────────────────────────────────────────

def fix_cline_tool_calls(answer: str, messages: list) -> str:
    # Strip markdown fences around tool calls
    answer = re.sub(r"```(?:xml)?\s*\n?(<[a-z_]+>)", r"\1", answer)
    answer = re.sub(r"(</[a-z_]+>)\s*\n?```", r"\1", answer)
    # Strip citation numbers
    answer = re.sub(r"\[\d+\]", "", answer)

    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = content_to_str(m.get("content", ""))
            break

    def infer_filename(fallback="output.md") -> str:
        for pattern in [
            r"create\s+(?:a\s+)?(\S+\.\w+)",
            r"write\s+(?:to\s+)?(\S+\.\w+)",
            r"make\s+(?:a\s+)?(\S+\.\w+)",
            r"file\s+(?:called\s+|named\s+)?(\S+\.\w+)",
            r"(\S+\.(?:md|txt|py|json|yaml|yml|js|ts|dart|sh|env|toml|cfg|html|css))",
        ]:
            match = re.search(pattern, last_user_msg, re.IGNORECASE)
            if match:
                return match.group(1)
        return fallback

    # ── Fix missing required params ───────────────────────────────────────────
    if "<write_to_file>" in answer and "<path>" not in answer:
        answer = answer.replace("<write_to_file>",
            f"<write_to_file>\n<path>{infer_filename('output.md')}</path>")
    if "<read_file>" in answer and "<path>" not in answer:
        answer = answer.replace("<read_file>",
            "<read_file>\n<path>openai_server.py</path>")
    if "<replace_in_file>" in answer and "<path>" not in answer:
        answer = answer.replace("<replace_in_file>",
            f"<replace_in_file>\n<path>{infer_filename('openai_server.py')}</path>")
    if "<execute_command>" in answer and "<requires_approval>" not in answer:
        answer = answer.replace("</execute_command>",
            "<requires_approval>false</requires_approval>\n</execute_command>")
    if "<search_files>" in answer and "<path>" not in answer:
        answer = answer.replace("<search_files>", "<search_files>\n<path>.</path>")
    if "<list_files>" in answer and "<path>" not in answer:
        answer = answer.replace("<list_files>", "<list_files>\n<path>.</path>")
    if "<ask_followup_question>" in answer and "<question>" not in answer:
        text = re.sub(r"<[^>]+>", "", answer).strip() or "Could you provide more details?"
        answer = answer.replace("<ask_followup_question>",
            f"<ask_followup_question>\n<question>{text}</question>")
    if "<attempt_completion>" in answer and "<result>" not in answer:
        text = re.sub(r"<[^>]+>", "", answer).strip() or "Task completed successfully."
        answer = answer.replace("<attempt_completion>",
            f"<attempt_completion>\n<result>{text}</result>")
    if "<plan_mode_respond>" in answer:
        if "<response>" not in answer:
            text = re.sub(r"<[^>]+>", "", answer).strip() or "Here is my analysis."
            answer = answer.replace("<plan_mode_respond>",
                f"<plan_mode_respond>\n<response>{text}</response>")
        answer = re.sub(r"<response>\s*</response>",
            "<response>Here is my analysis.</response>", answer)

    # ── Detect repeated read_file on same failing path → redirect to find it ──
    if "<read_file>" in answer:
        path_match = re.search(r"<path>(.*?)</path>", answer)
        if path_match:
            attempted_path = path_match.group(1).strip()
            fail_count = sum(
                1 for m in messages
                if m.get("role") == "tool"
                and "not found" in str(m.get("content", "")).lower()
                and attempted_path in str(m.get("content", ""))
            )
            if fail_count >= 1:
                # Already failed once — search for it instead of retrying
                answer = (
                    "<execute_command>\n"
                    "<command>powershell \"Get-ChildItem -Path $env:APPDATA -Recurse "
                    "-Filter cline_mcp_settings.json -ErrorAction SilentlyContinue "
                    "| Select-Object FullName\"</command>\n"
                    "<requires_approval>false</requires_approval>\n"
                    "</execute_command>"
                )

    # ── No tool tag → extract inline command or use smart default ─────────────
    if not any(tag in answer for tag in KNOWN_TOOLS):
        cmd_match = re.search(
            r"(?:run|execute|use|try|command)[:\s`]+([^\n`]{5,200})",
            answer, re.IGNORECASE
        )
        if cmd_match:
            cmd = cmd_match.group(1).strip().strip("`").strip('"').strip("'")
            answer = (
                f"<execute_command>\n"
                f"<command>{cmd}</command>\n"
                f"<requires_approval>false</requires_approval>\n"
                f"</execute_command>"
            )
        else:
            # Smart default: find cline settings file rather than blind list_files
            answer = (
                "<execute_command>\n"
                "<command>powershell \"Get-ChildItem -Path $env:APPDATA -Recurse "
                "-Filter cline_mcp_settings.json -ErrorAction SilentlyContinue "
                "| Select-Object FullName\"</command>\n"
                "<requires_approval>false</requires_approval>\n"
                "</execute_command>"
            )

    return answer.strip()


# ─── Query Builder ────────────────────────────────────────────────────────────

async def build_query_async(messages: list, tools: list = None) -> str:
    # Intercept and fix "file not found" tool results before sending to model
    messages = fix_tool_result_in_messages(messages)

    parts = [CLINE_SYSTEM_PROMPT]

    for m in messages:
        if m.get("role") == "system":
            text, _ = await process_message_content(m.get("content", ""))
            if text:
                parts.append(f"[Additional instructions: {text}]")

    if tools:
        tool_names = [t.get("function", {}).get("name", "") for t in tools if "function" in t]
        if tool_names:
            parts.append(f"[Available tools: {', '.join(tool_names)}]")

    for m in messages:
        role = m.get("role", "")
        if role == "system":
            continue
        text, _ = await process_message_content(m.get("content", ""))
        if not text:
            continue
        if role == "user":
            parts.append(f"<user_message>\n{text}\n</user_message>")
        elif role == "assistant":
            parts.append(f"<assistant_response>\n{text}\n</assistant_response>")
        elif role == "tool":
            parts.append(f"<tool_result tool_call_id='{m.get('tool_call_id', '')}'>\n{text}\n</tool_result>")

    return "\n\n".join(parts)


# ─── Response Extractor ───────────────────────────────────────────────────────

def extract_answer(result) -> str:
    if not isinstance(result, dict):
        return str(result)

    # SSE REST format: result["text"] is already a parsed dict
    text = result.get("text")
    if isinstance(text, dict):
        answer = text.get("answer", "")
        if answer:
            return answer

    # Fallback: blocks format
    for block in result.get("blocks", []):
        mb = block.get("markdown_block")
        if mb and mb.get("answer"):
            return mb["answer"]

    # Fallback: text steps list (old websocket format)
    text_steps = result.get("text", [])
    if isinstance(text_steps, list):
        for step in reversed(text_steps):
            if isinstance(step, dict) and step.get("step_type") == "FINAL":
                raw = step.get("content", {}).get("answer", "")
                if raw:
                    try:
                        return json.loads(raw).get("answer", raw)
                    except Exception:
                        return raw

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
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "object": "chat.completion",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": {
            "role": "assistant",
            "content": "⚠️ Request timed out. Please try again."
        }, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "message": "Perplexity OpenAI-compatible server is running"}


@app.get("/v1/models")
async def list_models():
    return JSONResponse({"object": "list", "data": [
        {"id": "claude-sonnet-4-6", "object": "model", "owned_by": "perplexity"},
        {"id": "auto",              "object": "model", "owned_by": "perplexity"},
        {"id": "pro",               "object": "model", "owned_by": "perplexity"},
        {"id": "reasoning",         "object": "model", "owned_by": "perplexity"},
        {"id": "deep research",     "object": "model", "owned_by": "perplexity"},
    ]})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    messages = body.get("messages", [])
    model    = body.get("model", "claude-sonnet-4-6")
    stream   = body.get("stream", False)
    tools    = body.get("tools", [])

    if not messages:
        return JSONResponse({"error": "messages field is required"}, status_code=400)

    # ── Resolve mode + wrapper model ──────────────────────────────────────────
    mode          = resolve_mode(model)
    wrapper_model = resolve_wrapper_model(model, mode)
    print(f"[Request] model={model} → mode={mode}, wrapper_model={wrapper_model}")

    # ── Extract images from LATEST user message only ──────────────────────────
    files = extract_files_from_last_user_message(messages)
    if files:
        print(f"[Images] {len(files)} image(s) from latest message → uploading via S3")

    # ── Build text query ──────────────────────────────────────────────────────
    query = await build_query_async(messages, tools or None)

    # ── Call wrapper ──────────────────────────────────────────────────────────
    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                lambda: client.search(
                    query,
                    mode=mode,
                    model=wrapper_model,
                    files=files,
                    stream=False,
                )
            ),
            timeout=120.0
        )
    except asyncio.TimeoutError:
        return JSONResponse(timeout_response(model))
    except Exception as e:
        traceback.print_exc()
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
        "choices": [{"index": 0, "message": {
            "role": "assistant", "content": answer
        }, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens":     len(query.split()),
            "completion_tokens": len(answer.split()),
            "total_tokens":      len(query.split()) + len(answer.split())
        }
    })
