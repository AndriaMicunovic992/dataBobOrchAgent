"""
git_tools.py  (Windows-safe)
=============================
Git operations for the orchestrator.

Uses the `gitpython` library (pure Python) as the primary driver —
no subprocess PATH issues on Windows.
Falls back to subprocess git only if gitpython is not installed.

Install: pip install gitpython
"""

import os
import asyncio
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

# ── Try to import gitpython ───────────────────────────────────────────────────
try:
    import git as gitpython
    HAS_GITPYTHON = True
except ImportError:
    HAS_GITPYTHON = False


# ── Internal helpers ──────────────────────────────────────────────────────────
def _inject_token(url: str) -> str:
    """Inject GitHub PAT into HTTPS URL for auth."""
    if not GITHUB_TOKEN:
        return url
    if "github.com" in url and "@" not in url:
        return url.replace("https://", f"https://{GITHUB_TOKEN}@")
    return url


def _check_git_available() -> str | None:
    """Return None if git is usable, or an error message if not."""
    if HAS_GITPYTHON:
        return None
    # Try subprocess git as fallback
    import shutil
    if shutil.which("git"):
        return None
    return (
        "Git not found. Fix:\n"
        "1. Install gitpython:  pip install gitpython\n"
        "   OR\n"
        "2. Install Git for Windows from git-scm.com, restart CMD/terminal"
    )


async def _run_in_thread(fn):
    """Run a blocking gitpython call in a thread so we don't block asyncio."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn)


# ── Subprocess fallback (only used if gitpython not installed) ────────────────
async def _git_subprocess(args: list[str], cwd: Path) -> tuple[int, str, str]:
    import shutil
    git_cmd = shutil.which("git") or "git"
    try:
        proc = await asyncio.create_subprocess_exec(
            git_cmd, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode, stdout.decode(errors="replace").strip(), stderr.decode(errors="replace").strip()
    except FileNotFoundError:
        return 1, "", "git executable not found. Install Git from git-scm.com or run: pip install gitpython"
    except Exception as e:
        return 1, "", str(e)


# ── Public API ────────────────────────────────────────────────────────────────
async def git_init(workspace: Path, repo_url: str, branch: str = "main") -> str:
    """Initialise a git repo in workspace and connect it to GitHub."""
    workspace.mkdir(parents=True, exist_ok=True)
    err = _check_git_available()
    if err:
        return f"ERROR: {err}"

    authed_url = _inject_token(repo_url)

    if HAS_GITPYTHON:
        def _do():
            git_dir = workspace / ".git"
            if git_dir.exists():
                repo = gitpython.Repo(str(workspace))
            else:
                repo = gitpython.Repo.init(str(workspace), initial_branch=branch)

            # Set remote
            if "origin" in [r.name for r in repo.remotes]:
                repo.remote("origin").set_url(authed_url)
            else:
                repo.create_remote("origin", authed_url)

            with repo.config_writer() as cfg:
                cfg.set_value("user", "email", "orchestrator@agent.local")
                cfg.set_value("user", "name",  "AI Orchestrator")

            return f"OK: git repo ready → {repo_url}"

        try:
            return await _run_in_thread(_do)
        except Exception as e:
            return f"ERROR git init: {e}"

    else:
        # Subprocess fallback
        git_dir = workspace / ".git"
        if not git_dir.exists():
            rc, _, err = await _git_subprocess(["init", "-b", branch], workspace)
            if rc != 0:
                # Older git doesn't support -b, try without
                rc, _, err2 = await _git_subprocess(["init"], workspace)
                if rc != 0:
                    return f"ERROR git init: {err2}"

        rc, out, _ = await _git_subprocess(["remote", "get-url", "origin"], workspace)
        if rc != 0:
            await _git_subprocess(["remote", "add", "origin", authed_url], workspace)
        else:
            await _git_subprocess(["remote", "set-url", "origin", authed_url], workspace)

        await _git_subprocess(["config", "user.email", "orchestrator@agent.local"], workspace)
        await _git_subprocess(["config", "user.name",  "AI Orchestrator"], workspace)
        return f"OK: git repo ready → {repo_url}"


async def git_status(workspace: Path) -> str:
    err = _check_git_available()
    if err:
        return f"ERROR: {err}"

    if HAS_GITPYTHON:
        def _do():
            if not (workspace / ".git").exists():
                return "No git repo initialised yet."
            repo = gitpython.Repo(str(workspace))
            changed = [item.a_path for item in repo.index.diff(None)]
            untracked = repo.untracked_files
            if not changed and not untracked:
                return "(clean — nothing to commit)"
            lines = []
            for f in changed:
                lines.append(f" M {f}")
            for f in untracked:
                lines.append(f"?? {f}")
            return "\n".join(lines)
        try:
            return await _run_in_thread(_do)
        except Exception as e:
            return f"ERROR: {e}"
    else:
        rc, out, err = await _git_subprocess(["status", "--short"], workspace)
        return out or "(clean)"


async def git_commit_push(
    workspace: Path,
    message: str,
    branch: str = "main",
    add_all: bool = True,
) -> str:
    """Stage all changes, commit, and push to GitHub."""
    err = _check_git_available()
    if err:
        return f"ERROR: {err}"

    if not (workspace / ".git").exists():
        return "ERROR: No git repo in workspace. Run git_init first or set a GitHub repo in /project settings."

    if HAS_GITPYTHON:
        def _do():
            repo = gitpython.Repo(str(workspace))
            if add_all:
                repo.git.add(A=True)

            if not repo.index.diff("HEAD") and not repo.untracked_files:
                # Check if there's a HEAD at all (first commit)
                try:
                    repo.head.commit
                    return "Nothing to commit — working tree clean."
                except ValueError:
                    pass  # No commits yet, proceed

            commit = repo.index.commit(message)

            # Push
            origin = repo.remote("origin")
            push_info = origin.push(refspec=f"HEAD:refs/heads/{branch}", set_upstream=True)

            for info in push_info:
                if info.flags & info.ERROR:
                    return f"Committed ({commit.hexsha[:7]}) but push failed: {info.summary}\nTip: check GITHUB_TOKEN in .env"

            return f"✅ Committed & pushed: \"{message}\" ({commit.hexsha[:7]})"

        try:
            return await _run_in_thread(_do)
        except Exception as e:
            return f"ERROR during commit/push: {e}\nTip: check GITHUB_TOKEN in .env"

    else:
        # Subprocess fallback
        if add_all:
            await _git_subprocess(["add", "-A"], workspace)

        rc, _, _ = await _git_subprocess(["diff", "--cached", "--quiet"], workspace)
        if rc == 0:
            return "Nothing to commit — working tree clean."

        rc, out, err = await _git_subprocess(["commit", "-m", message], workspace)
        if rc != 0:
            return f"ERROR commit: {err}"

        rc, out, err = await _git_subprocess(["push", "-u", "origin", branch], workspace)
        if rc != 0:
            return f"Committed locally but push failed: {err}\nTip: check GITHUB_TOKEN in .env"

        return f"✅ Committed & pushed: \"{message}\""


async def git_log(workspace: Path, n: int = 5) -> str:
    err = _check_git_available()
    if err:
        return f"ERROR: {err}"

    if HAS_GITPYTHON:
        def _do():
            if not (workspace / ".git").exists():
                return "(no git repo)"
            repo = gitpython.Repo(str(workspace))
            try:
                commits = list(repo.iter_commits(max_count=n))
            except Exception:
                return "(no commits yet)"
            lines = [f"{c.hexsha[:7]} {c.message.strip().splitlines()[0]}" for c in commits]
            return "\n".join(lines) or "(no commits yet)"
        try:
            return await _run_in_thread(_do)
        except Exception as e:
            return f"ERROR: {e}"
    else:
        rc, out, err = await _git_subprocess(["log", f"-{n}", "--oneline", "--decorate"], workspace)
        return out or "(no commits yet)"


async def create_gitignore(workspace: Path, stack: str = "python+node") -> str:
    """Write a .gitignore appropriate for the stack."""
    content = """# Common
.env
.env.*
!.env.example
*.pyc
__pycache__/
.DS_Store
Thumbs.db
"""
    if "python" in stack:
        content += """
# Python
venv/
.venv/
*.egg-info/
dist/
build/
.pytest_cache/
.mypy_cache/
"""
    if "node" in stack:
        content += """
# Node
node_modules/
.next/
dist/
.nuxt/
"""
    (workspace / ".gitignore").write_text(content, encoding="utf-8")
    return "OK: .gitignore written"


# ── Claude Code CLI helper (Windows-safe) ─────────────────────────────────────
async def run_claude_code(task: str, working_dir: Path) -> str:
    """
    Run the Claude Code CLI. Gives a clear error if not installed.
    Install: npm install -g @anthropic-ai/claude-code
    """
    import shutil
    claude_cmd = shutil.which("claude")
    if not claude_cmd:
        return (
            "❌ Claude Code CLI not found.\n"
            "To install:\n"
            "  1. Install Node.js from nodejs.org (LTS version)\n"
            "  2. Restart CMD / terminal\n"
            "  3. Run:  npm install -g @anthropic-ai/claude-code\n"
            "  4. Run:  claude --version  to verify\n\n"
            "Alternatively, the task will be handled by the SDK subagent instead."
        )

    working_dir.mkdir(parents=True, exist_ok=True)
    try:
        proc = await asyncio.create_subprocess_exec(
            claude_cmd, "--print", "--model", "claude-sonnet-4-6", task,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(working_dir),
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            return stdout.decode(errors="replace")
        return f"Claude Code error: {stderr.decode(errors='replace')}"
    except Exception as e:
        return f"Failed to run Claude Code: {e}"


# ── Tool definitions (passed to orchestrator) ─────────────────────────────────
GIT_TOOLS = [
    {
        "name": "git_commit_push",
        "description": (
            "Stage all changed files, commit with a meaningful message, and push to GitHub. "
            "Call this after completing any meaningful unit of work."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Commit message (imperative tense)"},
                "branch":  {"type": "string", "description": "Branch name (default: main)"},
            },
            "required": ["message"],
        },
    },
    {
        "name": "git_status",
        "description": "Show which files have changed since the last commit.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "git_log",
        "description": "Show the last N commit messages.",
        "input_schema": {
            "type": "object",
            "properties": {"n": {"type": "integer", "description": "Number of commits (default 5)"}},
        },
    },
    {
        "name": "git_init",
        "description": "Initialise the git repo and connect it to a GitHub remote URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_url": {"type": "string", "description": "GitHub HTTPS URL"},
                "branch":   {"type": "string", "description": "Default branch (default: main)"},
            },
            "required": ["repo_url"],
        },
    },
    {
        "name": "log_decision",
        "description": "Permanently record an architecture or technology decision with reasoning.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title":     {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["title", "reasoning"],
        },
    },
]
