"""
git_tools.py  (Windows-safe + Cloud-safe)
==========================================
Git operations for the orchestrator.

Priority order:
  1. gitpython (if installed AND git binary available)
  2. subprocess git (if git binary available)
  3. GitHub REST API fallback (always works, no git binary needed)

The GitHub API fallback ensures operations work on Railway, Docker,
and any environment where the git binary is not installed.
"""

import os
import asyncio
import base64
import json
import logging
import re
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger(__name__)

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


def _git_binary_available() -> bool:
    """Check if the git binary is actually available."""
    import shutil
    if shutil.which("git"):
        return True
    if HAS_GITPYTHON:
        try:
            gitpython.cmd.Git().version()
            return True
        except Exception:
            pass
    return False


def _check_git_available() -> str | None:
    """Return None if git (binary) is usable, or an error message if not."""
    if _git_binary_available():
        return None
    # Git binary not available -- but we have the API fallback, so only
    # return error if we also can't use the API
    return "git_binary_not_found"


async def _run_in_thread(fn):
    """Run a blocking gitpython call in a thread so we don't block asyncio."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn)


# ── Subprocess fallback ──────────────────────────────────────────────────────
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
        return 1, "", "git executable not found"
    except Exception as e:
        return 1, "", str(e)


# ══════════════════════════════════════════════════════════════════════════════
# GITHUB REST API FALLBACK
# When git binary is not available (common on Railway/cloud), these functions
# use the GitHub REST API to clone repos, pull changes, and commit+push.
# ══════════════════════════════════════════════════════════════════════════════

def _parse_github_url(url: str) -> tuple[str, str] | None:
    """Extract (owner, repo) from a GitHub URL. Returns None if not a GitHub URL."""
    # Handle various formats:
    #   https://github.com/owner/repo.git
    #   https://github.com/owner/repo
    #   https://TOKEN@github.com/owner/repo.git
    #   git@github.com:owner/repo.git
    m = re.search(r"github\.com[/:]([^/]+)/([^/\s]+?)(?:\.git)?$", url)
    if m:
        return m.group(1), m.group(2)
    return None


def _github_api(method: str, endpoint: str, body: dict | None = None, token: str = "") -> dict | list | str:
    """Make a GitHub API request. Returns parsed JSON or error string."""
    tk = token or GITHUB_TOKEN
    url = f"https://api.github.com{endpoint}" if endpoint.startswith("/") else endpoint
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "AI-Orchestrator/1.0",
    }
    if tk:
        headers["Authorization"] = f"Bearer {tk}"

    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}
    except HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        log.warning("GitHub API %s %s -> %s: %s", method, endpoint, e.code, error_body[:300])
        raise
    except URLError as e:
        log.warning("GitHub API connection error: %s", e)
        raise


def _github_download_file(owner: str, repo: str, path: str, branch: str = "main") -> bytes | None:
    """Download a single file's content via GitHub API. Returns bytes or None."""
    try:
        data = _github_api("GET", f"/repos/{owner}/{repo}/contents/{path}?ref={branch}")
        if isinstance(data, dict) and data.get("content"):
            return base64.b64decode(data["content"])
        return None
    except Exception:
        return None


async def _github_clone_repo(owner: str, repo: str, workspace: Path, branch: str = "main") -> str:
    """Clone a repo by downloading all files via GitHub API."""

    def _do():
        # Get the full tree recursively
        try:
            tree_data = _github_api("GET", f"/repos/{owner}/{repo}/git/trees/{branch}?recursive=1")
        except HTTPError as e:
            if e.code == 404:
                return f"ERROR: repo {owner}/{repo} or branch '{branch}' not found (404)"
            return f"ERROR: GitHub API error: {e.code}"
        except Exception as e:
            return f"ERROR: Cannot reach GitHub API: {e}"

        if not isinstance(tree_data, dict) or "tree" not in tree_data:
            return "ERROR: unexpected response from GitHub API"

        tree = tree_data["tree"]
        file_count = 0
        skipped = 0

        for item in tree:
            if item["type"] != "blob":
                continue

            file_path = workspace / item["path"]
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Download file content via blob API (handles large files)
            try:
                blob = _github_api("GET", f"/repos/{owner}/{repo}/git/blobs/{item['sha']}")
                if isinstance(blob, dict) and blob.get("content"):
                    content = base64.b64decode(blob["content"])
                    file_path.write_bytes(content)
                    file_count += 1
                else:
                    skipped += 1
            except Exception as e:
                log.warning("Failed to download %s: %s", item["path"], e)
                skipped += 1

        # Save repo metadata so we know what's been cloned
        meta = {
            "owner": owner,
            "repo": repo,
            "branch": branch,
            "base_tree_sha": tree_data.get("sha", ""),
            "base_commit_sha": "",
        }
        # Get the latest commit SHA
        try:
            ref_data = _github_api("GET", f"/repos/{owner}/{repo}/git/ref/heads/{branch}")
            if isinstance(ref_data, dict):
                meta["base_commit_sha"] = ref_data.get("object", {}).get("sha", "")
        except Exception:
            pass

        meta_dir = workspace / ".git_api"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        result = f"OK: downloaded {file_count} files from {owner}/{repo} (branch: {branch}) via GitHub API"
        if skipped:
            result += f" ({skipped} skipped)"
        return result

    return await _run_in_thread(_do)


async def _github_pull_repo(owner: str, repo: str, workspace: Path, branch: str = "main") -> str:
    """Pull latest changes by re-downloading changed files via GitHub API."""

    def _do():
        meta_file = workspace / ".git_api" / "meta.json"
        if not meta_file.exists():
            return "NEED_FULL_CLONE"

        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        old_commit = meta.get("base_commit_sha", "")

        # Get current commit
        try:
            ref_data = _github_api("GET", f"/repos/{owner}/{repo}/git/ref/heads/{branch}")
            new_commit = ref_data.get("object", {}).get("sha", "")
        except Exception as e:
            return f"ERROR: cannot get latest commit: {e}"

        if old_commit == new_commit:
            return f"OK: already up to date (commit {new_commit[:7]})"

        # Get the new tree and download everything (simple but reliable)
        try:
            tree_data = _github_api("GET", f"/repos/{owner}/{repo}/git/trees/{branch}?recursive=1")
        except Exception as e:
            return f"ERROR: cannot get tree: {e}"

        tree = tree_data.get("tree", [])
        updated = 0
        for item in tree:
            if item["type"] != "blob":
                continue
            file_path = workspace / item["path"]
            file_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                blob = _github_api("GET", f"/repos/{owner}/{repo}/git/blobs/{item['sha']}")
                if isinstance(blob, dict) and blob.get("content"):
                    content = base64.b64decode(blob["content"])
                    # Only write if different
                    if not file_path.exists() or file_path.read_bytes() != content:
                        file_path.write_bytes(content)
                        updated += 1
            except Exception:
                pass

        # Update metadata
        meta["base_commit_sha"] = new_commit
        meta["base_tree_sha"] = tree_data.get("sha", "")
        (workspace / ".git_api" / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        return f"OK: pulled {updated} updated files (commit {new_commit[:7]})"

    result = await _run_in_thread(_do)
    if result == "NEED_FULL_CLONE":
        return await _github_clone_repo(owner, repo, workspace, branch)
    return result


async def _github_commit_push(
    owner: str, repo: str, workspace: Path,
    message: str, branch: str = "main"
) -> str:
    """Commit and push all workspace changes via GitHub API (Git Data API)."""

    def _do():
        meta_file = workspace / ".git_api" / "meta.json"
        if not meta_file.exists():
            return "ERROR: no repo metadata. Run git_init first."

        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        base_commit = meta.get("base_commit_sha", "")
        if not base_commit:
            # Try to get it
            try:
                ref_data = _github_api("GET", f"/repos/{owner}/{repo}/git/ref/heads/{branch}")
                base_commit = ref_data.get("object", {}).get("sha", "")
            except Exception:
                pass

        if not base_commit:
            return "ERROR: cannot determine base commit SHA"

        # Get the base tree
        try:
            commit_data = _github_api("GET", f"/repos/{owner}/{repo}/git/commits/{base_commit}")
            base_tree_sha = commit_data.get("tree", {}).get("sha", "")
        except Exception as e:
            return f"ERROR: cannot get base commit data: {e}"

        # Collect all files in workspace (excluding .git_api and common ignores)
        skip_dirs = {".git", ".git_api", "__pycache__", "node_modules", ".venv", "venv", ".env"}
        skip_files = {".env", ".env.local"}
        tree_items = []

        for file_path in workspace.rglob("*"):
            if file_path.is_dir():
                continue
            rel_parts = file_path.relative_to(workspace).parts
            if any(part in skip_dirs for part in rel_parts):
                continue
            if file_path.name in skip_files:
                continue

            rel_path = str(file_path.relative_to(workspace)).replace("\\", "/")

            # Create blob for this file
            try:
                file_content = file_path.read_bytes()
                encoded = base64.b64encode(file_content).decode("ascii")
                blob = _github_api("POST", f"/repos/{owner}/{repo}/git/blobs", {
                    "content": encoded,
                    "encoding": "base64",
                })
                tree_items.append({
                    "path": rel_path,
                    "mode": "100644",
                    "type": "blob",
                    "sha": blob["sha"],
                })
            except Exception as e:
                log.warning("Failed to create blob for %s: %s", rel_path, e)

        if not tree_items:
            return "Nothing to commit (no files found in workspace)"

        # Create new tree
        try:
            new_tree = _github_api("POST", f"/repos/{owner}/{repo}/git/trees", {
                "base_tree": base_tree_sha,
                "tree": tree_items,
            })
        except Exception as e:
            return f"ERROR creating tree: {e}"

        # Create new commit
        try:
            new_commit = _github_api("POST", f"/repos/{owner}/{repo}/git/commits", {
                "message": message,
                "tree": new_tree["sha"],
                "parents": [base_commit],
                "author": {
                    "name": "AI Orchestrator",
                    "email": "orchestrator@agent.local",
                },
            })
        except Exception as e:
            return f"ERROR creating commit: {e}"

        # Update branch ref
        try:
            _github_api("PATCH", f"/repos/{owner}/{repo}/git/refs/heads/{branch}", {
                "sha": new_commit["sha"],
            })
        except Exception as e:
            return f"Commit created ({new_commit['sha'][:7]}) but ref update failed: {e}"

        # Update local metadata
        meta["base_commit_sha"] = new_commit["sha"]
        meta["base_tree_sha"] = new_tree["sha"]
        (workspace / ".git_api" / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        return f'Committed & pushed: "{message}" ({new_commit["sha"][:7]}) [{len(tree_items)} files]'

    return await _run_in_thread(_do)


async def _github_status(owner: str, repo: str, workspace: Path, branch: str = "main") -> str:
    """Compare workspace files against latest remote tree via API."""

    def _do():
        try:
            tree_data = _github_api("GET", f"/repos/{owner}/{repo}/git/trees/{branch}?recursive=1")
        except Exception as e:
            return f"ERROR: cannot get remote tree: {e}"

        remote_files = {}
        for item in tree_data.get("tree", []):
            if item["type"] == "blob":
                remote_files[item["path"]] = item["sha"]

        skip_dirs = {".git", ".git_api", "__pycache__", "node_modules", ".venv", "venv"}
        skip_files = {".env", ".env.local"}
        changes = []

        # Check local files
        local_files = set()
        for file_path in workspace.rglob("*"):
            if file_path.is_dir():
                continue
            rel_parts = file_path.relative_to(workspace).parts
            if any(part in skip_dirs for part in rel_parts):
                continue
            if file_path.name in skip_files:
                continue
            rel = str(file_path.relative_to(workspace)).replace("\\", "/")
            local_files.add(rel)
            if rel not in remote_files:
                changes.append(f"?? {rel}")
            # Note: we can't easily compare SHA without computing git blob SHA

        # Check for deleted files
        for remote_path in remote_files:
            if remote_path not in local_files:
                changes.append(f" D {remote_path}")

        if not changes:
            return "(clean or unchanged -- detailed diff requires git binary)"
        return "\n".join(changes)

    return await _run_in_thread(_do)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# Each function tries: gitpython -> subprocess -> GitHub API fallback
# ══════════════════════════════════════════════════════════════════════════════

async def git_init(workspace: Path, repo_url: str, branch: str = "main") -> str:
    """Clone an existing GitHub repo into workspace, or connect if already cloned."""
    workspace.mkdir(parents=True, exist_ok=True)
    authed_url = _inject_token(repo_url)
    git_dir = workspace / ".git"

    # ── Try gitpython first ──
    if _git_binary_available() and HAS_GITPYTHON:
        def _do():
            if git_dir.exists():
                repo = gitpython.Repo(str(workspace))
                if "origin" in [r.name for r in repo.remotes]:
                    repo.remote("origin").set_url(authed_url)
                else:
                    repo.create_remote("origin", authed_url)
                with repo.config_writer() as cfg:
                    cfg.set_value("user", "email", "orchestrator@agent.local")
                    cfg.set_value("user", "name",  "AI Orchestrator")
                try:
                    repo.remote("origin").fetch()
                    if branch in [ref.name.split("/")[-1] for ref in repo.remote("origin").refs]:
                        repo.git.reset("--mixed", f"origin/{branch}")
                except Exception as pull_err:
                    return f"OK: repo connected (pull skipped: {pull_err})"
                return f"OK: git repo synced -> {repo_url}"
            else:
                import shutil
                import tempfile
                tmp_dir = Path(tempfile.mkdtemp())
                try:
                    repo = gitpython.Repo.clone_from(
                        authed_url, str(tmp_dir), branch=branch,
                    )
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
                return f"OK: cloned {repo_url} ({file_count} files)"

        try:
            return await _run_in_thread(_do)
        except Exception as e:
            log.warning("gitpython clone failed, trying API fallback: %s", e)

    # ── Try subprocess git ──
    elif _git_binary_available():
        if not git_dir.exists():
            import shutil
            import tempfile
            tmp_dir = Path(tempfile.mkdtemp())
            rc, out, clone_err = await _git_subprocess(
                ["clone", "--branch", branch, authed_url, str(tmp_dir)], workspace.parent,
            )
            if rc != 0:
                rc, out, clone_err = await _git_subprocess(
                    ["clone", authed_url, str(tmp_dir)], workspace.parent,
                )
            if rc == 0:
                for item in tmp_dir.iterdir():
                    dest = workspace / item.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(str(dest))
                        else:
                            dest.unlink()
                    shutil.move(str(item), str(dest))
                shutil.rmtree(str(tmp_dir), ignore_errors=True)
                await _git_subprocess(["config", "user.email", "orchestrator@agent.local"], workspace)
                await _git_subprocess(["config", "user.name",  "AI Orchestrator"], workspace)
                return f"OK: cloned {repo_url}"
            else:
                shutil.rmtree(str(tmp_dir), ignore_errors=True)
                log.warning("subprocess clone failed: %s", clone_err)
        else:
            await _git_subprocess(["remote", "set-url", "origin", authed_url], workspace)
            await _git_subprocess(["fetch", "origin"], workspace)
            await _git_subprocess(["reset", "--mixed", f"origin/{branch}"], workspace)
            return f"OK: git repo synced -> {repo_url}"

    # ── GitHub API fallback ──
    parsed = _parse_github_url(repo_url)
    if not parsed:
        return f"ERROR: not a GitHub URL, and git binary not available: {repo_url}"

    owner, repo_name = parsed
    log.info("Using GitHub API fallback for clone: %s/%s", owner, repo_name)

    # Check if we already have an API clone
    api_meta = workspace / ".git_api" / "meta.json"
    if api_meta.exists() or git_dir.exists():
        # Already cloned (via API or git) -- just pull latest
        return await _github_pull_repo(owner, repo_name, workspace, branch)

    result = await _github_clone_repo(owner, repo_name, workspace, branch)
    return result


async def git_pull(workspace: Path, branch: str = "main") -> str:
    """Pull latest changes from GitHub before starting work."""
    git_dir = workspace / ".git"
    api_meta = workspace / ".git_api" / "meta.json"

    # ── If we have a real git repo and git binary, use it ──
    if git_dir.exists() and _git_binary_available():
        if HAS_GITPYTHON:
            def _do():
                repo = gitpython.Repo(str(workspace))
                if GITHUB_TOKEN and "origin" in [r.name for r in repo.remotes]:
                    current_url = repo.remote("origin").url
                    if "github.com" in current_url and "@" not in current_url:
                        repo.remote("origin").set_url(_inject_token(current_url))
                repo.remote("origin").fetch()
                repo.git.reset("--mixed", f"origin/{branch}")
                repo.git.checkout("--", ".")
                return f"OK: pulled latest from origin/{branch}"
            try:
                return await _run_in_thread(_do)
            except Exception as e:
                log.warning("gitpython pull failed: %s", e)
        else:
            await _git_subprocess(["fetch", "origin"], workspace)
            rc, out, err = await _git_subprocess(["reset", "--mixed", f"origin/{branch}"], workspace)
            if rc == 0:
                await _git_subprocess(["checkout", "--", "."], workspace)
                return f"OK: pulled latest from origin/{branch}"

    # ── GitHub API fallback ──
    if api_meta.exists():
        meta = json.loads(api_meta.read_text(encoding="utf-8"))
        owner, repo_name = meta["owner"], meta["repo"]
        return await _github_pull_repo(owner, repo_name, workspace, branch)

    if not git_dir.exists() and not api_meta.exists():
        return "ERROR: No repo in workspace. Run git_init first."

    return "ERROR: git binary not available and no API metadata found. Run git_init to re-clone."


async def git_status(workspace: Path) -> str:
    git_dir = workspace / ".git"
    api_meta = workspace / ".git_api" / "meta.json"

    if git_dir.exists() and _git_binary_available():
        if HAS_GITPYTHON:
            def _do():
                repo = gitpython.Repo(str(workspace))
                changed = [item.a_path for item in repo.index.diff(None)]
                untracked = repo.untracked_files
                if not changed and not untracked:
                    return "(clean -- nothing to commit)"
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

    # ── API fallback: list local files vs remote ──
    if api_meta.exists():
        meta = json.loads(api_meta.read_text(encoding="utf-8"))
        return await _github_status(meta["owner"], meta["repo"], workspace, meta.get("branch", "main"))

    if not git_dir.exists():
        return "No git repo initialised yet."
    return "ERROR: git binary not available"


async def git_commit_push(
    workspace: Path,
    message: str,
    branch: str = "main",
    add_all: bool = True,
) -> str:
    """Stage all changes, commit, and push to GitHub."""
    git_dir = workspace / ".git"
    api_meta = workspace / ".git_api" / "meta.json"

    # ── Try gitpython ──
    if git_dir.exists() and _git_binary_available() and HAS_GITPYTHON:
        def _do():
            repo = gitpython.Repo(str(workspace))
            if add_all:
                repo.git.add(A=True)

            if not repo.index.diff("HEAD") and not repo.untracked_files:
                try:
                    repo.head.commit
                    return "Nothing to commit -- working tree clean."
                except ValueError:
                    pass

            commit = repo.index.commit(message)
            origin = repo.remote("origin")
            push_info = origin.push(refspec=f"HEAD:refs/heads/{branch}", set_upstream=True)

            for info in push_info:
                if info.flags & info.ERROR:
                    return f"Committed ({commit.hexsha[:7]}) but push failed: {info.summary}"

            return f'Committed & pushed: "{message}" ({commit.hexsha[:7]})'

        try:
            return await _run_in_thread(_do)
        except Exception as e:
            log.warning("gitpython commit/push failed: %s", e)

    # ── Try subprocess ──
    elif git_dir.exists() and _git_binary_available():
        if add_all:
            await _git_subprocess(["add", "-A"], workspace)
        rc, _, _ = await _git_subprocess(["diff", "--cached", "--quiet"], workspace)
        if rc == 0:
            return "Nothing to commit -- working tree clean."
        rc, out, err = await _git_subprocess(["commit", "-m", message], workspace)
        if rc != 0:
            return f"ERROR commit: {err}"
        rc, out, err = await _git_subprocess(["push", "-u", "origin", branch], workspace)
        if rc != 0:
            return f"Committed locally but push failed: {err}"
        return f'Committed & pushed: "{message}"'

    # ── GitHub API fallback ──
    if api_meta.exists():
        meta = json.loads(api_meta.read_text(encoding="utf-8"))
        owner, repo_name = meta["owner"], meta["repo"]
        log.info("Using GitHub API fallback for commit+push: %s/%s", owner, repo_name)
        return await _github_commit_push(owner, repo_name, workspace, message, branch)

    if not git_dir.exists() and not api_meta.exists():
        return "ERROR: No git repo in workspace. Run git_init first."
    return "ERROR: git binary not available and no API metadata found. Run git_init to re-clone."


async def git_log(workspace: Path, n: int = 5) -> str:
    git_dir = workspace / ".git"
    api_meta = workspace / ".git_api" / "meta.json"

    if git_dir.exists() and _git_binary_available():
        if HAS_GITPYTHON:
            def _do():
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

    # ── API fallback: show recent commits ──
    if api_meta.exists():
        meta = json.loads(api_meta.read_text(encoding="utf-8"))
        owner, repo_name = meta["owner"], meta["repo"]
        try:
            commits = _github_api("GET", f"/repos/{owner}/{repo_name}/commits?per_page={n}&sha={meta.get('branch', 'main')}")
            if isinstance(commits, list):
                lines = []
                for c in commits:
                    sha = c.get("sha", "")[:7]
                    msg = c.get("commit", {}).get("message", "").split("\n")[0]
                    lines.append(f"{sha} {msg}")
                return "\n".join(lines) or "(no commits yet)"
        except Exception as e:
            return f"ERROR fetching log via API: {e}"

    return "(no git repo)"


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
.git_api/
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


# ── Claude Code CLI helper ───────────────────────────────────────────────────
async def run_claude_code(task: str, working_dir: Path) -> str:
    """
    Run the Claude Code CLI. Gives a clear error if not installed.
    """
    import shutil
    claude_cmd = shutil.which("claude")
    if not claude_cmd:
        return (
            "Claude Code CLI not found.\n"
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
            "It downloads all existing files so agents can read and modify them. "
            "Works even without git binary installed (uses GitHub API fallback)."
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
