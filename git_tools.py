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
    import shutil
    # gitpython STILL needs the git binary — check it's actually there
    if shutil.which("git"):
        return None
    if HAS_GITPYTHON:
        # gitpython imported but git binary missing — common on Railway/Docker
        try:
            gitpython.Repo.init.__module__  # basic check
            # Try to actually invoke git to verify
            gitpython.cmd.Git().version()
            return None
        except Exception:
            return (
                "gitpython installed but git binary not found.\n"
                "On Railway: add a nixpacks.toml with aptPkgs = [\"git\"]\n"
                "On Docker: add 'RUN apt-get install -y git' to Dockerfile"
            )
    return (
        "Git not found. Fix:\n"
        "1. Install gitpython:  pip install gitpython\n"
        "   OR\n"
        "2. Install git:  apt-get install git (Linux) / brew install git (Mac)"
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
    """Clone an existing GitHub repo into workspace, or connect if already cloned."""
    workspace.mkdir(parents=True, exist_ok=True)
    err = _check_git_available()
    if err:
        return f"ERROR: {err}"

    authed_url = _inject_token(repo_url)
    git_dir = workspace / ".git"

    if HAS_GITPYTHON:
        def _do():
            if git_dir.exists():
                # Already have a repo — update remote URL and pull latest
                repo = gitpython.Repo(str(workspace))
                if "origin" in [r.name for r in repo.remotes]:
                    repo.remote("origin").set_url(authed_url)
                else:
                    repo.create_remote("origin", authed_url)
                with repo.config_writer() as cfg:
                    cfg.set_value("user", "email", "orchestrator@agent.local")
                    cfg.set_value("user", "name",  "AI Orchestrator")
                # Pull latest changes
                try:
                    repo.remote("origin").fetch()
                    if branch in [ref.name.split("/")[-1] for ref in repo.remote("origin").refs]:
                        repo.git.reset("--mixed", f"origin/{branch}")
                except Exception as pull_err:
                    return f"OK: repo connected (pull skipped: {pull_err}) → {repo_url}"
                return f"OK: git repo synced → {repo_url}"
            else:
                # Clone the repo into workspace
                # gitpython clone_from needs an empty or non-existent dir,
                # but workspace may have files. Use a temp clone then move.
                import shutil
                import tempfile
                tmp_dir = Path(tempfile.mkdtemp())
                try:
                    repo = gitpython.Repo.clone_from(
                        authed_url, str(tmp_dir), branch=branch,
                    )
                    # Move .git and all repo files into workspace
                    for item in tmp_dir.iterdir():
                        dest = workspace / item.name
                        if dest.exists():
                            if dest.is_dir():
                                shutil.rmtree(str(dest))
                            else:
                                dest.unlink()
                        shutil.move(str(item), str(dest))
                finally:
                    if tmp_dir.exists():
                        shutil.rmtree(str(tmp_dir), ignore_errors=True)

                repo = gitpython.Repo(str(workspace))
                with repo.config_writer() as cfg:
                    cfg.set_value("user", "email", "orchestrator@agent.local")
                    cfg.set_value("user", "name",  "AI Orchestrator")
                file_count = len(list(workspace.rglob("*")))
                return f"OK: cloned {repo_url} ({file_count} files) → workspace"

        try:
            return await _run_in_thread(_do)
        except Exception as e:
            return f"ERROR git init/clone: {e}"

    else:
        # Subprocess fallback — use git clone
        if not git_dir.exists():
            # Clone into workspace (need empty dir for clone)
            import shutil
            import tempfile
            tmp_dir = Path(tempfile.mkdtemp())
            rc, out, clone_err = await _git_subprocess(
                ["clone", "--branch", branch, authed_url, str(tmp_dir)], workspace.parent,
            )
            if rc != 0:
                # Maybe branch doesn't exist yet, try without --branch
                rc, out, clone_err = await _git_subprocess(
                    ["clone", authed_url, str(tmp_dir)], workspace.parent,
                )
            if rc == 0:
                # Move cloned content into workspace
                for item in tmp_dir.iterdir():
                    dest = workspace / item.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(str(dest))
                        else:
                            dest.unlink()
                    shutil.move(str(item), str(dest))
                shutil.rmtree(str(tmp_dir), ignore_errors=True)
            else:
                shutil.rmtree(str(tmp_dir), ignore_errors=True)
                # Fall back to init + remote add for empty repos
                rc2, _, err2 = await _git_subprocess(["init", "-b", branch], workspace)
                if rc2 != 0:
                    await _git_subprocess(["init"], workspace)
                await _git_subprocess(["remote", "add", "origin", authed_url], workspace)
        else:
            # Already cloned — update remote and pull
            await _git_subprocess(["remote", "set-url", "origin", authed_url], workspace)
            await _git_subprocess(["fetch", "origin"], workspace)
            await _git_subprocess(["reset", "--mixed", f"origin/{branch}"], workspace)

        await _git_subprocess(["config", "user.email", "orchestrator@agent.local"], workspace)
        await _git_subprocess(["config", "user.name",  "AI Orchestrator"], workspace)
        return f"OK: git repo ready → {repo_url}"


async def git_pull(workspace: Path, branch: str = "main") -> str:
    """Pull latest changes from GitHub before starting work."""
    err = _check_git_available()
    if err:
        return f"ERROR: {err}"
    if not (workspace / ".git").exists():
        return "ERROR: No git repo. Run git_init first."

    if HAS_GITPYTHON:
        def _do():
            repo = gitpython.Repo(str(workspace))
            # Ensure token is injected in remote URL
            if GITHUB_TOKEN and "origin" in [r.name for r in repo.remotes]:
                current_url = repo.remote("origin").url
                if "github.com" in current_url and "@" not in current_url:
                    repo.remote("origin").set_url(_inject_token(current_url))
            repo.remote("origin").fetch()
            # Reset working tree to match remote (keeps untracked files)
            repo.git.reset("--mixed", f"origin/{branch}")
            repo.git.checkout("--", ".")
            return f"OK: pulled latest from origin/{branch}"
        try:
            return await _run_in_thread(_do)
        except Exception as e:
            return f"ERROR git pull: {e}"
    else:
        await _git_subprocess(["fetch", "origin"], workspace)
        rc, out, err = await _git_subprocess(["reset", "--mixed", f"origin/{branch}"], workspace)
        if rc != 0:
            return f"ERROR git pull: {err}"
        await _git_subprocess(["checkout", "--", "."], workspace)
        return f"OK: pulled latest from origin/{branch}"


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
        "description": (
            "Clone a GitHub repo into the workspace (or sync if already cloned). "
            "This MUST be called before any coding work on an existing repo. "
            "It downloads all existing files so agents can read and modify them."
        ),
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
        "name": "git_pull",
        "description": "Pull latest changes from GitHub before starting work. Use at the start of each session.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {"type": "string", "description": "Branch to pull (default: main)"},
            },
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
