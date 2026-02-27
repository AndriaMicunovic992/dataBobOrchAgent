"""
orchestrator_engine.py  (v3 — project-aware + git + data layer)
================================================================
What's new vs v2:
  ✅ Project context injected into every Opus prompt
  ✅ Git commit/push after each coding session
  ✅ Specialised DATA AGENT for DAX/SQL queries
  ✅ Decisions logged permanently
  ✅ Per-project workspace isolation
  ✅ Tech stack remembered across sessions
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Callable, Awaitable

log = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

from anthropic import AsyncAnthropic

import git_tools
from git_tools import GIT_TOOLS
from project_memory import Project, get_active, set_active, slugify, list_projects
from rate_limiter import RateLimitedClient

# ── Config ────────────────────────────────────────────────────────────────────
ORCHESTRATOR_MODEL = "claude-opus-4-6"
SUBAGENT_MODEL     = "claude-sonnet-4-6"
CHEAP_MODEL        = "claude-haiku-4-5-20251001"

_raw_client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
client = RateLimitedClient(_raw_client, max_tpm=30_000)

# ── Persistent conversation memory ───────────────────────────────────────────
CONV_DIR = Path("./data/conversations")
CONV_DIR.mkdir(parents=True, exist_ok=True)
MAX_CONVERSATION_MESSAGES = 80  # ~40 user/assistant turns

_conversations: dict[int, list[dict]] = {}
_conversations_loaded: set[int] = set()
_session_resumed: set[int] = set()  # users whose conversations were loaded from disk


def _serialize_content(content):
    """Convert Anthropic SDK ContentBlock objects to JSON-serializable dicts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        result = []
        for item in content:
            if isinstance(item, dict):
                result.append(item)
            elif hasattr(item, "type"):
                if item.type == "text":
                    result.append({"type": "text", "text": item.text})
                elif item.type == "tool_use":
                    result.append({
                        "type": "tool_use",
                        "id": item.id,
                        "name": item.name,
                        "input": item.input,
                    })
                elif item.type == "tool_result":
                    result.append({
                        "type": "tool_result",
                        "tool_use_id": getattr(item, "tool_use_id", ""),
                        "content": getattr(item, "content", ""),
                    })
                else:
                    try:
                        result.append(item.model_dump())
                    except Exception:
                        result.append({"type": "text", "text": str(item)})
            else:
                result.append(item)
        return result
    return content


def _serialize_message(msg: dict) -> dict:
    """Serialize a single conversation message for JSON storage."""
    return {"role": msg["role"], "content": _serialize_content(msg.get("content", ""))}


def _save_conversation(user_id: int):
    """Persist a user's conversation to disk."""
    try:
        conv = _conversations.get(user_id, [])
        serialized = [_serialize_message(m) for m in conv]
        fp = CONV_DIR / f"{user_id}.json"
        fp.write_text(json.dumps(serialized, ensure_ascii=False, default=str), encoding="utf-8")
    except Exception as e:
        log.warning("Failed to save conversation for user %s: %s", user_id, e)


def _load_conversation(user_id: int) -> list[dict]:
    """Load a user's conversation from disk."""
    fp = CONV_DIR / f"{user_id}.json"
    if not fp.exists():
        return []
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception as e:
        log.warning("Failed to load conversation for user %s: %s", user_id, e)
    return []


def _extract_text_from_message(msg: dict) -> str:
    """Pull readable text out of a conversation message (for summarization)."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "tool_use":
                    parts.append(f"[called {item.get('name', '?')}]")
                elif item.get("type") == "tool_result":
                    c = item.get("content", "")
                    if isinstance(c, str):
                        parts.append(c[:200])
            elif hasattr(item, "type"):
                if item.type == "text":
                    parts.append(item.text)
                elif item.type == "tool_use":
                    parts.append(f"[called {item.name}]")
        return " ".join(parts)
    return str(content)[:300]


async def _summarize_old_messages(old_messages: list[dict], project_name: str = "") -> str:
    """Use Haiku to compress a chunk of conversation into a short summary."""
    # Build a readable transcript from the messages being dropped
    transcript_lines = []
    if project_name:
        transcript_lines.append(f"[Project: {project_name}]")
    for msg in old_messages:
        role = msg.get("role", "?").upper()
        text = _extract_text_from_message(msg)
        if text.strip():
            # Cap each message to avoid huge transcripts
            transcript_lines.append(f"{role}: {text[:500]}")

    transcript = "\n".join(transcript_lines)
    # Cap the total transcript sent to Haiku
    if len(transcript) > 12000:
        transcript = transcript[:12000] + "\n... (truncated)"

    try:
        response = await client.messages.create(
            model=CHEAP_MODEL,
            max_tokens=600,
            system=(
                "You are a conversation summarizer. Given an excerpt from a "
                "conversation between a user and an AI development assistant, "
                "produce a concise bullet-point summary capturing:\n"
                "- The project name and what is being built\n"
                "- Key decisions made\n"
                "- Tasks completed or in progress\n"
                "- Important technical details (file names, tools, errors)\n"
                "- Any open questions or pending items\n\n"
                "Be brief and factual. Max 10 bullet points. "
                "Always start with the project name/identity."
            ),
            messages=[{"role": "user", "content": transcript}],
        )
        summary = " ".join(b.text for b in response.content if b.type == "text")
        return summary.strip()
    except Exception as e:
        log.warning("Summarization failed, falling back to simple trim: %s", e)
        return ""


async def _summarize_and_trim(conversation: list[dict], project: "Project | None" = None) -> list[dict]:
    """Trim conversation to size limit, summarizing the dropped messages first.
    Returns the trimmed conversation (may be unchanged if under the limit)."""
    if len(conversation) <= MAX_CONVERSATION_MESSAGES:
        return conversation

    # Messages to keep: first 2 (initial context) + most recent ones
    keep_recent = MAX_CONVERSATION_MESSAGES - 3  # -3 to leave room for summary msg
    old_messages = conversation[2:-keep_recent]  # the ones being dropped
    recent_messages = conversation[-keep_recent:]

    # Summarize the old messages, including project name for context
    project_name = ""
    if project:
        data = project.load()
        project_name = data.get("name", project.slug)
    summary = await _summarize_old_messages(old_messages, project_name=project_name)

    if summary:
        header = "[CONVERSATION SUMMARY - earlier discussion condensed]"
        if project_name:
            header = f"[CONVERSATION SUMMARY for project '{project_name}' - earlier discussion condensed]"
        summary_msg = {
            "role": "user",
            "content": f"{header}\n{summary}\n[END SUMMARY - conversation continues below]",
        }
        return conversation[:2] + [summary_msg] + recent_messages
    else:
        # Summarization failed, fall back to simple trim
        return conversation[:2] + recent_messages


def get_conversation(user_id: int) -> list[dict]:
    if user_id not in _conversations_loaded:
        loaded = _load_conversation(user_id)
        _conversations[user_id] = loaded
        _conversations_loaded.add(user_id)
        if loaded:
            _session_resumed.add(user_id)  # mark: this session was restored from disk
    return _conversations.setdefault(user_id, [])


def clear_conversation(user_id: int):
    _conversations[user_id] = []
    _save_conversation(user_id)

def _sanitize_conversation(conversation: list[dict]) -> list[dict]:
    """Remove orphaned tool_use messages that lack matching tool_result.
    This prevents API 400 errors from corrupted conversation history."""
    if len(conversation) < 2:
        return conversation
    sanitized = []
    i = 0
    while i < len(conversation):
        msg = conversation[i]
        if msg.get("role") == "assistant":
            content = msg.get("content", [])
            has_tool_use = False
            if isinstance(content, list):
                has_tool_use = any(
                    (isinstance(b, dict) and b.get("type") == "tool_use") or
                    (hasattr(b, "type") and b.type == "tool_use")
                    for b in content
                )
            if has_tool_use:
                # Check if the next message has matching tool_results
                if i + 1 < len(conversation):
                    next_msg = conversation[i + 1]
                    next_content = next_msg.get("content", [])
                    has_tool_result = (
                        next_msg.get("role") == "user" and
                        isinstance(next_content, list) and
                        any(isinstance(b, dict) and b.get("type") == "tool_result" for b in next_content)
                    )
                    if has_tool_result:
                        sanitized.append(msg)
                    else:
                        # Orphaned tool_use — skip it
                        log.warning("Dropping orphaned tool_use message at index %d", i)
                        i += 1
                        continue
                else:
                    # tool_use at end of conversation with no result — skip
                    log.warning("Dropping trailing tool_use message at index %d", i)
                    i += 1
                    continue
            else:
                sanitized.append(msg)
        else:
            sanitized.append(msg)
        i += 1
    return sanitized


# ── File Tools ────────────────────────────────────────────────────────────────
SUBAGENT_FILE_TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file from the agent's workspace.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
    {
        "name": "write_file",
        "description": "Write or overwrite a file in the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "append_file",
        "description": "Append content to a file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": "List files and folders in a workspace directory.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
]

def handle_file_tool(name: str, inputs: dict, workspace: Path) -> str:
    workspace.mkdir(parents=True, exist_ok=True)
    if name == "read_file":
        fp = workspace / inputs["path"]
        return fp.read_text(encoding="utf-8") if fp.exists() else f"ERROR: not found: {inputs['path']}"
    elif name == "write_file":
        fp = workspace / inputs["path"]
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(inputs["content"], encoding="utf-8")
        return f"OK: wrote {len(inputs['content'])} chars → {inputs['path']}"
    elif name == "append_file":
        fp = workspace / inputs["path"]
        fp.parent.mkdir(parents=True, exist_ok=True)
        with open(fp, "a", encoding="utf-8") as f:
            f.write(inputs["content"])
        return f"OK: appended to {inputs['path']}"
    elif name == "list_dir":
        dp = workspace / inputs.get("path", ".")
        if not dp.exists():
            return "ERROR: dir not found"
        entries = sorted(dp.iterdir(), key=lambda p: (p.is_file(), p.name))
        return "\n".join(("📁 " if e.is_dir() else "📄 ") + e.name for e in entries) or "(empty)"
    return f"Unknown tool: {name}"


# ── Specialised Data Agent (DAX / SQL) ────────────────────────────────────────
DATA_AGENT_SYSTEM = """You are an expert data query subagent specialising in:
- DAX (Data Analysis Expressions) for Power BI / Analysis Services semantic models
- SQL for relational databases (PostgreSQL, SQL Server, SQLite, etc.)
- Query optimisation and best practices

Given a task, you will:
1. Write the correct query (DAX or SQL depending on the target)
2. Add clear comments explaining each section
3. Include performance considerations where relevant
4. Write the query to a .dax or .sql file using write_file

Reply format:
  QUERY_TYPE: DAX | SQL
  TARGET: (semantic model name / table name / DB name)
  SUMMARY: one-line description
  FILES_WRITTEN: filename(s)
  NOTES: optimisation tips or caveats"""

async def run_data_agent(
    task: str,
    context: str = "",
    workspace: Path = Path("./workspace"),
    label: str = "data-agent",
    status_cb: Callable[[str], Awaitable[None]] | None = None,
) -> str:
    messages = [{"role": "user", "content": f"TASK:\n{task}\n\nCONTEXT:\n{context}"}]
    workspace.mkdir(parents=True, exist_ok=True)
    if status_cb:
        await status_cb(f"📊 `[{label}]` data agent started…")

    while True:
        response = await client.messages.create(
            model=SUBAGENT_MODEL,
            max_tokens=4096,
            system=DATA_AGENT_SYSTEM,
            tools=SUBAGENT_FILE_TOOLS,
            messages=messages,
        )
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses or response.stop_reason == "end_turn":
            if status_cb:
                await status_cb(f"✅ `[{label}]` data agent done.")
            return " ".join(b.text for b in response.content if b.type == "text")

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tool in tool_uses:
            result = handle_file_tool(tool.name, tool.input, workspace)
            tool_results.append({"type": "tool_result", "tool_use_id": tool.id, "content": result})
        messages.append({"role": "user", "content": tool_results})


# ── SDK Coding Subagent ───────────────────────────────────────────────────────
def build_subagent_system(project: Project | None) -> str:
    base = """You are an expert coding subagent with file tools.
Complete the task fully and autonomously — do not ask questions.
Use read_file to check existing code before writing. Use write_file to persist results.
Add a SUMMARY comment at the top of every file you create.

Reply format:
  SUMMARY: one-line description
  FILES_WRITTEN: comma-separated list
  NOTES: caveats or follow-up suggestions"""

    if project:
        data = project.load()
        return base + f"""

PROJECT CONTEXT:
  Name     : {data.get('name', '')}
  Backend  : {data.get('backend', 'TBD')}
  Frontend : {data.get('frontend', 'TBD')}
  Data     : {data.get('data_layer', 'TBD')}
  Database : {data.get('database', 'TBD')}
Conventions: {data.get('conventions', 'follow project patterns')}
"""
    return base

async def run_sdk_subagent(
    task: str,
    context: str = "",
    workspace: Path = Path("./workspace"),
    cheap: bool = False,
    label: str = "agent",
    status_cb: Callable[[str], Awaitable[None]] | None = None,
    project: Project | None = None,
) -> str:
    model = CHEAP_MODEL if cheap else SUBAGENT_MODEL
    messages = [{"role": "user", "content": f"TASK:\n{task}\n\nCONTEXT:\n{context}"}]
    workspace.mkdir(parents=True, exist_ok=True)
    if status_cb:
        await status_cb(f"🔧 `[{label}]` started…")

    while True:
        response = await client.messages.create(
            model=model, max_tokens=4096,
            system=build_subagent_system(project),
            tools=SUBAGENT_FILE_TOOLS,
            messages=messages,
        )
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses or response.stop_reason == "end_turn":
            if status_cb:
                await status_cb(f"✅ `[{label}]` finished.")
            return " ".join(b.text for b in response.content if b.type == "text")

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tool in tool_uses:
            if status_cb:
                await status_cb(f"  📂 `[{label}]` {tool.name}")
            result = handle_file_tool(tool.name, tool.input, workspace)
            tool_results.append({"type": "tool_result", "tool_use_id": tool.id, "content": result})
        messages.append({"role": "user", "content": tool_results})


# ── Claude Code CLI subagent (Windows-safe, falls back to SDK) ───────────────
async def run_claude_code_subagent(
    task: str, working_dir: Path, label: str = "cc-agent",
    status_cb: Callable[[str], Awaitable[None]] | None = None,
    project: "Project | None" = None,
) -> str:
    working_dir.mkdir(parents=True, exist_ok=True)
    if status_cb:
        await status_cb(f"🖥️ `[{label}]` Claude Code CLI running…")

    result = await git_tools.run_claude_code(task, working_dir)

    # If Claude Code isn't installed, fall back to SDK subagent automatically
    if "Claude Code CLI not found" in result:
        if status_cb:
            await status_cb(
                "⚠️ Claude Code CLI not installed — falling back to SDK agent.\n"
                "_(Install Node.js + `npm install -g @anthropic-ai/claude-code` for full shell access)_"
            )
        return await run_sdk_subagent(
            task=task, workspace=working_dir,
            label=label, status_cb=status_cb, project=project,
        )

    if status_cb:
        await status_cb(f"✅ `[{label}]` done.")
    return result


# ── Parallel runner ───────────────────────────────────────────────────────────
async def run_parallel_agents(
    tasks: list[dict], project: Project,
    status_cb: Callable[[str], Awaitable[None]] | None = None,
) -> list[dict]:
    async def _one(t: dict, idx: int) -> dict:
        label   = t.get("label") or f"agent-{idx+1}"
        task_id = project.add_task(t["task"][:80])
        project.update_task(task_id, "running")
        try:
            result = await run_sdk_subagent(
                task=t["task"], context=t.get("context", ""),
                workspace=project.workspace / label,
                cheap=t.get("cheap", False), label=label,
                status_cb=status_cb, project=project,
            )
            project.update_task(task_id, "done", result[:500])
            return {"label": label, "task_id": task_id, "result": result}
        except Exception as e:
            project.update_task(task_id, "failed", str(e))
            return {"label": label, "task_id": task_id, "result": f"FAILED: {e}"}

    if status_cb:
        await status_cb(f"⚡ Launching {len(tasks)} agents in parallel…")
    return list(await asyncio.gather(*[_one(t, i) for i, t in enumerate(tasks)]))


# ── Agent routing & dependency context ────────────────────────────────────────

def _build_dependency_context(task_id: int, group_id: int, project: Project) -> str:
    """Build context string from the outputs of completed dependency tasks."""
    db = project.get_db()
    row = db.execute(
        "SELECT depends_on FROM enhanced_tasks WHERE id=?", (task_id,)
    ).fetchone()
    if not row or not row[0]:
        return ""
    dep_ids = json.loads(row[0])
    if not dep_ids:
        return ""
    parts = ["=== OUTPUTS FROM DEPENDENCY TASKS ==="]
    for dep_id in dep_ids:
        dep = db.execute(
            "SELECT title, agent_type, result, files_written FROM enhanced_tasks WHERE id=?",
            (dep_id,),
        ).fetchone()
        if dep:
            parts.append(f"\n--- Task #{dep_id}: {dep[0]} ({dep[1]}) ---")
            parts.append(f"Files: {dep[3] or 'none'}")
            parts.append(f"Output: {(dep[2] or '')[:800]}")
    return "\n".join(parts)


async def route_to_agent(
    task: dict, project: Project, group_id: int,
    status_cb: Callable[[str], Awaitable[None]] | None = None,
) -> str:
    """Route a task to the appropriate specialised agent based on agent_type."""
    agent_type = task.get("agent_type", "backend")
    label = f"{agent_type}-{task['id']}"
    ws = project.workspace / label
    context = _build_dependency_context(task["id"], group_id, project)
    full_task = f"{task['title']}\n\n{task.get('description', '')}"

    if agent_type == "data":
        return await run_data_agent(
            task=full_task, context=context, workspace=ws,
            label=label, status_cb=status_cb,
        )
    elif agent_type == "review":
        return await run_review_subagent(
            task=full_task, context=context, workspace=project.workspace,
            label=label, status_cb=status_cb, project=project,
        )
    else:
        specialisations = {
            "backend": "You are a BACKEND specialist. Focus on API routes, business logic, database models, and auth.",
            "frontend": "You are a FRONTEND specialist. Focus on components, UI, state management, and styling.",
            "test": "You are a TEST specialist. Write thorough tests with edge cases. Use the project's test framework.",
        }
        extra = specialisations.get(agent_type, "")
        if extra:
            context = f"{context}\n\n{extra}" if context else extra
        return await run_sdk_subagent(
            task=full_task, context=context, workspace=ws,
            cheap=False, label=label, status_cb=status_cb, project=project,
        )


# ── Task group execution engine ──────────────────────────────────────────────

def _extract_files_written(result: str) -> str:
    for line in result.split("\n"):
        if line.strip().upper().startswith("FILES_WRITTEN:"):
            return line.split(":", 1)[1].strip()
    return ""


def _format_group_results(status: dict) -> str:
    lines = [f"Execution complete. {status['counts'].get('done', 0)}/{status['total']} tasks done."]
    if status["has_failures"]:
        lines.append(f"FAILURES: {status['counts'].get('failed', 0)} tasks failed")
    for t in status["tasks"]:
        icon = {"done": "done", "failed": "FAIL", "blocked": "BLOCKED"}.get(t["status"], t["status"])
        summary = (t.get("summary") or "")[:100]
        files = t.get("files") or ""
        lines.append(f"  [{icon}] #{t['id']} ({t['agent_type']}) {t['title']}")
        if summary:
            lines.append(f"       {summary}")
        if files:
            lines.append(f"       Files: {files}")
    return "\n".join(lines)


async def execute_task_group_loop(
    group_id: int, project: Project,
    status_cb: Callable[[str], Awaitable[None]] | None = None,
) -> str:
    """Execute a task group in waves, respecting dependencies."""
    wave = 0
    while True:
        wave += 1
        ready_tasks = project.get_ready_tasks(group_id)

        if not ready_tasks:
            status = project.get_group_status(group_id)
            if status["all_done"]:
                if status_cb:
                    await status_cb(f"All {status['total']} tasks complete.")
                return _format_group_results(status)
            elif status["counts"].get("blocked", 0) > 0:
                return (
                    f"DEADLOCK: {status['counts']['blocked']} tasks blocked "
                    "but no tasks are ready. Check dependencies."
                )
            else:
                return _format_group_results(status)

        if status_cb:
            names = ", ".join(t["title"][:30] for t in ready_tasks)
            await status_cb(f"Wave {wave}: launching {len(ready_tasks)} tasks ({names})")

        async def _execute_one(task: dict) -> dict:
            project.mark_task_running(task["id"])
            try:
                result = await route_to_agent(
                    task=task, project=project,
                    group_id=group_id, status_cb=status_cb,
                )
                files = _extract_files_written(result)
                project.mark_task_done(task["id"], result, files)
                return {"id": task["id"], "status": "done", "result": result}
            except Exception as e:
                project.mark_task_failed(task["id"], str(e))
                return {"id": task["id"], "status": "failed", "error": str(e)}

        results = await asyncio.gather(*[_execute_one(t) for t in ready_tasks])

        if status_cb:
            done = sum(1 for r in results if r["status"] == "done")
            failed = sum(1 for r in results if r["status"] == "failed")
            await status_cb(f"Wave {wave} complete: {done} done, {failed} failed")

        newly_ready = project.resolve_dependencies(group_id)
        if status_cb and newly_ready > 0:
            await status_cb(f"{newly_ready} tasks unblocked for next wave")


# ── Merge subagent outputs into main workspace ──────────────────────────────

def merge_subagent_outputs(project: Project) -> list[str]:
    """Copy files written by parallel agents (in subdirectories) into the main workspace.
    Returns list of files merged."""
    import shutil
    ws = project.workspace
    merged = []
    skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv"}

    for entry in ws.iterdir():
        # Only process agent subdirectories (not regular repo files)
        if not entry.is_dir():
            continue
        if entry.name in skip_dirs:
            continue
        if not entry.name.startswith(("backend-", "frontend-", "data-", "test-", "review-", "agent-")):
            continue

        # Copy all files from agent subdir into main workspace
        for src_file in entry.rglob("*"):
            if src_file.is_file() and ".git" not in src_file.parts:
                rel = src_file.relative_to(entry)
                dest = ws / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src_file), str(dest))
                merged.append(str(rel))

        # Clean up the agent subdirectory
        shutil.rmtree(str(entry), ignore_errors=True)

    return merged


# ── Review agent ─────────────────────────────────────────────────────────────

REVIEW_AGENT_SYSTEM = """You are an expert code review agent. Your job is to review the outputs of multiple parallel coding agents and check for:

1. FILE CONFLICTS: Did two agents write to the same file? Do their outputs clash?
2. INTEGRATION ISSUES: Do imports, API contracts, and type definitions align across modules?
3. MISSING PIECES: Are there gaps — e.g., a route references a model that wasn't created?
4. CONSISTENCY: Do naming conventions, error handling, and patterns match across files?
5. QUALITY: Are there obvious bugs, security issues, or anti-patterns?

Read all relevant files in the workspace using read_file and list_dir.

Reply format:
  STATUS: CLEAN | ISSUES_FOUND
  ISSUES: (numbered list of issues, or "none")
  FIXES_NEEDED: (numbered list of specific fixes, or "none")
  FILES_REVIEWED: comma-separated list"""


async def run_review_subagent(
    task: str, context: str = "",
    workspace: Path = Path("./workspace"),
    label: str = "reviewer",
    status_cb: Callable[[str], Awaitable[None]] | None = None,
    project: Project | None = None,
) -> str:
    """Run a review agent that checks integration across agent outputs."""
    messages = [{"role": "user", "content": f"REVIEW TASK:\n{task}\n\nCONTEXT:\n{context}"}]
    if status_cb:
        await status_cb(f"🔍 `[{label}]` reviewing code…")

    while True:
        response = await client.messages.create(
            model=SUBAGENT_MODEL, max_tokens=4096,
            system=REVIEW_AGENT_SYSTEM,
            tools=SUBAGENT_FILE_TOOLS,
            messages=messages,
        )
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses or response.stop_reason == "end_turn":
            if status_cb:
                await status_cb(f"✅ `[{label}]` review complete.")
            return " ".join(b.text for b in response.content if b.type == "text")

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tool in tool_uses:
            result = handle_file_tool(tool.name, tool.input, workspace)
            tool_results.append({"type": "tool_result", "tool_use_id": tool.id, "content": result})
        messages.append({"role": "user", "content": tool_results})


async def run_review_agent(
    group_id: int, project: Project,
    focus: str = "",
    status_cb: Callable[[str], Awaitable[None]] | None = None,
) -> str:
    """Top-level review: gather all task results and spawn review subagent."""
    completed = project.get_task_results_for_review(group_id)
    parts = [f"Feature group #{group_id} — {len(completed)} tasks completed."]
    for t in completed:
        parts.append(f"\n--- {t['title']} ({t['agent_type']}) ---")
        parts.append(f"Files written: {t.get('files') or 'unknown'}")
        parts.append(f"Result summary: {(t.get('result') or '')[:600]}")

    review_task = (
        "Review all outputs from the parallel agents listed in CONTEXT. "
        "Check for file conflicts, integration issues, missing pieces, and consistency. "
        "Read the actual files from the workspace to verify."
    )
    if focus:
        review_task += f"\n\nSpecific focus: {focus}"

    return await run_review_subagent(
        task=review_task, context="\n".join(parts),
        workspace=project.workspace, label="reviewer",
        status_cb=status_cb, project=project,
    )


# ── Orchestrator tools ────────────────────────────────────────────────────────
def build_orchestrator_tools() -> list[dict]:
    coding_tools = [
        {
            "name": "spawn_coding_agent",
            "description": "Dispatch one focused coding task (backend, frontend, config) to a Sonnet subagent with file tools.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task":    {"type": "string"},
                    "context": {"type": "string"},
                    "cheap":   {"type": "boolean"},
                    "label":   {"type": "string", "description": "Short name, becomes a subfolder"},
                },
                "required": ["task"],
            },
        },
        {
            "name": "spawn_parallel_agents",
            "description": "Run MULTIPLE independent coding tasks simultaneously. Use for independent modules.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string"}, "context": {"type": "string"},
                                "cheap": {"type": "boolean"}, "label": {"type": "string"},
                            },
                            "required": ["task"],
                        },
                    }
                },
                "required": ["tasks"],
            },
        },
        {
            "name": "spawn_data_agent",
            "description": (
                "Dispatch a DAX or SQL query writing task to a specialised data agent. "
                "Use for: Power BI measures, calculated columns, semantic model queries, "
                "SQL stored procedures, views, or complex analytical queries."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "task":        {"type": "string", "description": "What query/measure to build"},
                    "query_type":  {"type": "string", "enum": ["DAX", "SQL", "auto"], "description": "Query language to use"},
                    "target":      {"type": "string", "description": "Target model/table/DB name"},
                    "context":     {"type": "string", "description": "Schema, existing measures, or sample data"},
                    "label":       {"type": "string"},
                },
                "required": ["task"],
            },
        },
        {
            "name": "spawn_claude_code_agent",
            "description": "Dispatch to Claude Code CLI for complex multi-file work needing shell/test/install.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "label": {"type": "string"},
                },
                "required": ["task"],
            },
        },
        {
            "name": "read_workspace_file",
            "description": "Read a project file to review or pass as context to a subagent.",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
        {
            "name": "list_workspace",
            "description": "List files in the project workspace.",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
        {
            "name": "update_project_meta",
            "description": (
                "Update the project's persistent metadata (tech stack, conventions, GitHub repo, etc.). "
                "Call this when the stack is confirmed or conventions are established."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "backend":      {"type": "string"},
                    "frontend":     {"type": "string"},
                    "data_layer":   {"type": "string", "description": "e.g. Power BI semantic models, PostgreSQL, etc."},
                    "database":     {"type": "string"},
                    "auth":         {"type": "string"},
                    "github_repo":  {"type": "string"},
                    "conventions":  {"type": "string", "description": "Coding conventions and patterns to follow"},
                    "description":  {"type": "string"},
                },
            },
        },
        {
            "name": "create_task",
            "description": "Log a task as pending in the project task DB.",
            "input_schema": {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]},
        },
        {
            "name": "list_tasks",
            "description": "Show recent tasks and their statuses for this project.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "create_new_project",
            "description": (
                "Create a new project and set it as active. Call this FIRST when setting up a new project. "
                "This works even when no project is active yet. After calling this, all other tools become available."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Human-readable project name"},
                    "description": {"type": "string", "description": "Brief project description"},
                    "backend": {"type": "string", "description": "Backend tech (e.g. FastAPI, Django, Node.js)"},
                    "frontend": {"type": "string", "description": "Frontend tech (e.g. React, Vue, Streamlit)"},
                    "data_layer": {"type": "string", "description": "Data layer (e.g. PostgreSQL, Power BI)"},
                    "database": {"type": "string", "description": "Database type"},
                    "auth": {"type": "string", "description": "Auth method (e.g. JWT, OAuth)"},
                    "github_repo": {"type": "string", "description": "GitHub repo URL (optional)"},
                },
                "required": ["name"],
            },
        },
    ]
    planning_tools = [
        {
            "name": "create_task_group",
            "description": (
                "Create a named group of tasks with dependencies for a feature. "
                "Use in Phase 3 (DECOMPOSE) to break a feature into atomic tasks. "
                "Each task specifies an agent_type and which other tasks (by index) it depends on."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short name for this task group (e.g. 'auth-system')"},
                    "feature": {"type": "string", "description": "Description of the feature being built"},
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string", "description": "Short task title"},
                                "description": {"type": "string", "description": "Detailed instructions for the agent"},
                                "agent_type": {
                                    "type": "string",
                                    "enum": ["backend", "frontend", "data", "test", "review"],
                                    "description": "Which specialised agent should handle this",
                                },
                                "depends_on_indices": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": "Indices (0-based) of tasks in THIS array that must complete first",
                                },
                            },
                            "required": ["title", "description", "agent_type"],
                        },
                    },
                },
                "required": ["name", "feature", "tasks"],
            },
        },
        {
            "name": "execute_task_group",
            "description": (
                "Execute a task group created by create_task_group. "
                "Runs tasks in waves: all READY tasks run in parallel, then dependencies are resolved, "
                "then the next wave runs. Continues until all tasks are done or failed."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "integer", "description": "The group ID returned by create_task_group"},
                },
                "required": ["group_id"],
            },
        },
        {
            "name": "review_results",
            "description": (
                "Spawn a review agent to check all completed task outputs for a group. "
                "Checks for: file conflicts, integration issues, missing imports, "
                "API contract mismatches, and overall consistency. Use in Phase 5."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "integer", "description": "The group ID to review"},
                    "focus": {"type": "string", "description": "Specific concerns to check (optional)"},
                },
                "required": ["group_id"],
            },
        },
        {
            "name": "get_group_status",
            "description": "Get the current execution status of a task group — how many done, running, blocked, failed.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "integer"},
                },
                "required": ["group_id"],
            },
        },
    ]
    return coding_tools + planning_tools + GIT_TOOLS


ORCHESTRATOR_TOOLS = build_orchestrator_tools()


# ── Orchestrator system prompt ────────────────────────────────────────────────
ORCHESTRATOR_SYSTEM_BASE = """You are an expert AI architect and orchestrator managing a team of specialised coding, data, and review subagents.
You are talking to your user via Telegram — keep replies concise and well-formatted for mobile.
Use Telegram-compatible Markdown (bold with *, code with `backticks`, code blocks with ```).

You manage a full-stack project with these agent specialisations:
  BACKEND   — API, business logic, auth, database schemas
  FRONTEND  — UI, components, state management, styling
  DATA      — DAX measures, SQL queries, semantic models
  TEST      — Unit tests, integration tests, test fixtures
  REVIEW    — Code review, integration check, conflict detection

=== MANDATORY WORKFLOW ===

For EVERY feature request or significant change, you MUST follow these five phases IN ORDER.
Do NOT skip phases. Do NOT start coding before planning.

**PHASE 1: ANALYSE** (understand before acting)
- Read the project context injected above
- If the project has a github_repo set, call git_pull first to sync latest code
- If the workspace has existing code, use read_workspace_file and list_workspace to examine it
- If the workspace is empty (new project), that is NORMAL — skip file reading and proceed
- If the repo hasn't been cloned yet, call git_init with the repo URL first
- Identify what already exists, what needs to change, what is new
- Summarise your understanding to the user in 2-3 sentences

**PHASE 2: ARCHITECT** (design before building)
- Design the technical approach: what components, what data flow, what APIs
- Call log_decision for every significant architectural choice
- Call update_project_meta if the stack or conventions change
- Present the plan to the user with a brief summary

**PHASE 3: DECOMPOSE** (break down into tasks with dependencies)
- Call create_task_group to create a named group of tasks
- Each task must specify: agent_type (backend/frontend/data/test/review), dependencies (list of task indices that must complete first)
- Tasks with no dependencies are READY and can run in parallel
- Tasks with dependencies are BLOCKED until their dependencies complete
- Present the task breakdown to the user

**PHASE 4: EXECUTE** (run tasks respecting dependencies)
- Call execute_task_group to begin execution
- The system will automatically:
  - Run all READY tasks in parallel using the correct agent type
  - Wait for completion, then unblock dependent tasks
  - Continue until all tasks are done or failed
- Monitor progress via status callbacks

**PHASE 5: REVIEW & INTEGRATE** (verify before committing)
- Call review_results to spawn a review agent that checks:
  - All task outputs for consistency
  - File conflicts between parallel agents
  - Integration issues (imports, API contracts, type mismatches)
- If issues found: fix them by spawning targeted agents
- If clean: call git_commit_push with a meaningful message
- Summarise what was built to the user

=== RULES ===
- ALWAYS read project context at the top of this prompt before acting
- ALWAYS follow the five phases — never skip straight to coding
- ALWAYS call git_init (if repo not cloned) or git_pull (if already cloned) before making changes to an existing repo
- PREFER parallel execution for independent tasks
- Use spawn_data_agent for ALL DAX/SQL work — never give data tasks to coding agents
- After completing a feature, always git_commit_push
- Use cheap=true for boilerplate, simple config, or documentation tasks
- Keep Telegram replies short — user is on mobile
- Mention file paths and commit hashes in summaries
- For small questions or clarifications, respond directly without the full workflow
- You still have spawn_coding_agent and spawn_parallel_agents for simple ad-hoc tasks
- spawn_coding_agent works in the main workspace — agents can read/modify existing repo files directly
"""

def build_system_prompt(project: Project | None, session_resumed: bool = False) -> str:
    resumed_hint = ""
    if session_resumed:
        resumed_hint = (
            "\n\n=== SESSION RESUMED ===\n"
            "This is a RESUMED session. The conversation history below was loaded from "
            "a previous session. Read it carefully to understand what was being discussed "
            "before continuing. Do NOT ask the user to remind you — the context is in the "
            "conversation history. Pick up where you left off.\n"
            "=== END SESSION NOTE ===\n"
        )

    if not project:
        return (
            ORCHESTRATOR_SYSTEM_BASE
            + "\n\nNo active project. Call create_new_project to set one up. "
            "Ask the user what they want to build, then call create_new_project with the details. "
            "You can still answer questions without an active project, but you need one to use coding/git tools."
            + resumed_hint
        )
    return ORCHESTRATOR_SYSTEM_BASE + "\n" + project.build_enhanced_context_prompt() + resumed_hint


# ── Tool handler ──────────────────────────────────────────────────────────────
async def handle_tool(
    name: str, inputs: dict,
    project: Project,
    status_cb: Callable[[str], Awaitable[None]] | None,
) -> str:
    ws = project.workspace

    if name == "spawn_coding_agent":
        label   = inputs.get("label", "agent")
        task_id = project.add_task(inputs["task"][:80])
        project.update_task(task_id, "running")
        # Single agents work directly in the main workspace (where the cloned repo lives)
        # so they can read and modify existing files
        result  = await run_sdk_subagent(
            task=inputs["task"], context=inputs.get("context", ""),
            workspace=ws, cheap=inputs.get("cheap", False),
            label=label, status_cb=status_cb, project=project,
        )
        project.update_task(task_id, "done", result[:500])
        return result

    elif name == "spawn_parallel_agents":
        results = await run_parallel_agents(inputs["tasks"], project, status_cb)
        # Merge outputs from agent subdirectories into main workspace
        merged = merge_subagent_outputs(project)
        if merged and status_cb:
            await status_cb(f"Merged {len(merged)} files from parallel agents into workspace")
        return "\n\n---\n\n".join(f"*[{r['label']}]*\n{r['result']}" for r in results)

    elif name == "spawn_data_agent":
        label   = inputs.get("label", "data-agent")
        target  = inputs.get("target", "")
        qtype   = inputs.get("query_type", "auto")
        context = inputs.get("context", "")
        full_context = f"Query type: {qtype}\nTarget: {target}\n{context}"
        task_id = project.add_task(f"[DATA] {inputs['task'][:70]}")
        project.update_task(task_id, "running")
        # Data agents also work in main workspace to access existing files
        result = await run_data_agent(
            task=inputs["task"], context=full_context,
            workspace=ws, label=label, status_cb=status_cb,
        )
        project.update_task(task_id, "done", result[:500])
        return result

    elif name == "spawn_claude_code_agent":
        label   = inputs.get("label", "cc-agent")
        task_id = project.add_task(inputs["task"][:80])
        project.update_task(task_id, "running")
        # Claude Code agent works in main workspace
        result = await run_claude_code_subagent(
            task=inputs["task"], working_dir=ws,
            label=label, status_cb=status_cb,
        )
        project.update_task(task_id, "done", result[:500])
        return result

    elif name == "read_workspace_file":
        return handle_file_tool("read_file", inputs, ws)

    elif name == "list_workspace":
        return handle_file_tool("list_dir", inputs, ws)

    elif name == "update_project_meta":
        project.update(**inputs)
        return "✅ Project metadata updated."

    elif name == "create_task":
        tid = project.add_task(inputs["title"])
        return f"Task #{tid} created."

    elif name == "list_tasks":
        tasks = project.list_tasks()
        if not tasks:
            return "No tasks yet."
        icons = {"done": "✅", "running": "⏳", "failed": "❌", "pending": "🕐"}
        return "\n".join(f"{icons.get(t['status'], '•')} #{t['id']} {t['title']}" for t in tasks)

    elif name == "create_new_project":
        # When called with an existing project — just update metadata
        project.update(**{k: v for k, v in inputs.items() if k != "name" and v})
        return f"Project already active: {project.slug}. Metadata updated."

    # ── Planning & execution tools ────────────────────────────────────────────
    elif name == "create_task_group":
        result = project.create_task_group(
            name=inputs["name"], feature=inputs["feature"], tasks=inputs["tasks"],
        )
        lines = [f"Task group #{result['group_id']} created: {inputs['name']}"]
        for t in result["tasks"]:
            icon = "🟢" if t["status"] == "ready" else "🔒"
            lines.append(f"  {icon} #{t['id']} {t['title']} ({t['status']})")
        return "\n".join(lines)

    elif name == "execute_task_group":
        result = await execute_task_group_loop(inputs["group_id"], project, status_cb)
        # Merge subagent output subdirectories into main workspace
        merged = merge_subagent_outputs(project)
        if merged and status_cb:
            await status_cb(f"Merged {len(merged)} files from agent subdirs into workspace")
        return result

    elif name == "review_results":
        return await run_review_agent(
            inputs["group_id"], project,
            focus=inputs.get("focus", ""), status_cb=status_cb,
        )

    elif name == "get_group_status":
        status = project.get_group_status(inputs["group_id"])
        icons = {"done": "done", "running": "run", "ready": "rdy",
                 "blocked": "blk", "failed": "FAIL", "pending": "pnd"}
        lines = [f"Group #{status['group_id']}: {status['counts']}"]
        for t in status["tasks"]:
            lines.append(f"  [{icons.get(t['status'], '?')}] #{t['id']} ({t['agent_type']}) {t['title']}")
        if status["all_done"]:
            lines.append("ALL TASKS COMPLETE")
        if status["has_failures"]:
            lines.append("WARNING: Some tasks failed")
        return "\n".join(lines)

    # ── Git tools ─────────────────────────────────────────────────────────────
    elif name == "git_init":
        result = await git_tools.git_init(ws, inputs["repo_url"], inputs.get("branch", "main"))
        project.update(github_repo=inputs["repo_url"])
        return result

    elif name == "git_commit_push":
        # Merge any remaining subagent outputs before committing
        merged = merge_subagent_outputs(project)
        if merged and status_cb:
            await status_cb(f"Merged {len(merged)} files before commit")
        return await git_tools.git_commit_push(
            ws, inputs["message"], inputs.get("branch", "main")
        )

    elif name == "git_pull":
        return await git_tools.git_pull(ws, inputs.get("branch", "main"))

    elif name == "git_status":
        return await git_tools.git_status(ws)

    elif name == "git_log":
        return await git_tools.git_log(ws, inputs.get("n", 5))

    elif name == "log_decision":
        project.log_decision(inputs["title"], inputs["reasoning"])
        return f"✅ Decision logged: {inputs['title']}"

    return f"Unknown tool: {name}"


# ── Main chat entry point ─────────────────────────────────────────────────────
async def chat(
    user_id: int,
    message: str,
    project: Project | None = None,
    status_cb: Callable[[str], Awaitable[None]] | None = None,
) -> str:
    conversation = get_conversation(user_id)
    conversation.append({"role": "user", "content": message})

    # ── Project recovery: ensure we don't lose project context ──
    if project is None:
        project = get_active(user_id)  # retry from persistent storage
    if project is None:
        # Auto-select if there's exactly one project on disk
        projects = list_projects()
        if len(projects) == 1:
            project = Project(projects[0]["slug"])
            set_active(user_id, projects[0]["slug"])
            log.info("Auto-selected sole project '%s' for user %s", projects[0]["slug"], user_id)

    # ── Check if this is a resumed session (loaded from disk after restart) ──
    is_resumed = user_id in _session_resumed
    if is_resumed:
        _session_resumed.discard(user_id)  # only show hint once

    # Summarize & trim conversation if it's getting too long
    trimmed = await _summarize_and_trim(conversation, project=project)
    if len(trimmed) != len(conversation):
        _conversations[user_id] = trimmed
        conversation = trimmed
        _save_conversation(user_id)  # persist the trimmed version

    system = build_system_prompt(project, session_resumed=is_resumed)

    while True:
        # Sanitize conversation to fix any orphaned tool_use messages
        clean = _sanitize_conversation(conversation)
        if len(clean) != len(conversation):
            _conversations[user_id] = clean
            conversation = clean
        try:
            response = await client.messages.create(
                model=ORCHESTRATOR_MODEL,
                max_tokens=4096,
                system=system,
                tools=ORCHESTRATOR_TOOLS,
                messages=conversation,
            )
        except Exception as api_err:
            # If conversation is corrupted beyond repair, reset and retry
            err_str = str(api_err)
            if "400" in err_str or "invalid" in err_str.lower():
                log.warning("API error with conversation, resetting: %s", api_err)
                _conversations[user_id] = [{"role": "user", "content": message}]
                conversation = _conversations[user_id]
                response = await client.messages.create(
                    model=ORCHESTRATOR_MODEL,
                    max_tokens=4096,
                    system=system,
                    tools=ORCHESTRATOR_TOOLS,
                    messages=conversation,
                )
            else:
                raise

        text_parts = [b.text for b in response.content if b.type == "text"]
        tool_uses  = [b for b in response.content if b.type == "tool_use"]

        if not tool_uses or response.stop_reason == "end_turn":
            conversation.append({"role": "assistant", "content": response.content})
            # ── Auto-save conversation to disk ──
            _save_conversation(user_id)
            return "".join(text_parts)

        conversation.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tool in tool_uses:
            try:
                if status_cb:
                    await status_cb(f"⚙️ `{tool.name}`…")
                # Special case: create_new_project works without an active project
                if tool.name == "create_new_project" and not project:
                    inp = tool.input
                    slug = slugify(inp["name"])
                    project = Project(slug)
                    meta = {"name": inp["name"]}
                    for key in ("description", "backend", "frontend", "data_layer",
                                "database", "auth", "github_repo"):
                        if inp.get(key):
                            meta[key] = inp[key]
                    project.save(meta)
                    set_active(user_id, slug)
                    # Rebuild system prompt with new project context
                    system = build_system_prompt(project)
                    result = f"✅ Project '{inp['name']}' created and activated (slug: {slug})."
                    if status_cb:
                        await status_cb(result)
                elif project:
                    result = await handle_tool(tool.name, tool.input, project, status_cb)
                else:
                    result = (
                        "No active project. Call create_new_project first to set one up, "
                        "or ask the user to run /newproject."
                    )
            except Exception as e:
                log.exception("Tool execution error for %s", tool.name)
                result = f"❌ Tool '{tool.name}' failed: {e}"
            tool_results.append({"type": "tool_result", "tool_use_id": tool.id, "content": result})
        conversation.append({"role": "user", "content": tool_results})
        # ── Auto-save after each tool round too ──
        _save_conversation(user_id)
