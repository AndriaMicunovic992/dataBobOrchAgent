"""
project_memory.py
=================
Manages all persistent project knowledge that survives restarts.

Stores in ./projects/<slug>/
    project.json      — identity, tech stack, structure, conventions
    decisions.md      — architecture decisions log
    tasks.db          — task history (shared with engine)
    workspace/        — all generated code (git repo lives here)
"""

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

PROJECTS_ROOT = Path("./projects")


# ── Helpers ───────────────────────────────────────────────────────────────────
def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def list_projects() -> list[dict]:
    if not PROJECTS_ROOT.exists():
        return []
    result = []
    for d in sorted(PROJECTS_ROOT.iterdir()):
        pf = d / "project.json"
        if pf.exists():
            data = json.loads(pf.read_text())
            result.append({"slug": d.name, **data})
    return result


# ── Project class ─────────────────────────────────────────────────────────────
class Project:
    def __init__(self, slug: str):
        self.slug    = slug
        self.root    = PROJECTS_ROOT / slug
        self.file    = self.root / "project.json"
        self.dec_log = self.root / "decisions.md"
        self.db_path = self.root / "tasks.db"
        self.workspace = self.root / "workspace"
        self.root.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._db: sqlite3.Connection | None = None

    # ── project.json ──────────────────────────────────────────────────────────
    def exists(self) -> bool:
        return self.file.exists()

    def load(self) -> dict:
        if not self.file.exists():
            return {}
        return json.loads(self.file.read_text())

    def save(self, data: dict):
        data["updated"] = datetime.utcnow().isoformat()
        self.file.write_text(json.dumps(data, indent=2))

    def update(self, **kwargs):
        data = self.load()
        data.update(kwargs)
        self.save(data)

    # ── decisions.md ──────────────────────────────────────────────────────────
    def log_decision(self, title: str, reasoning: str):
        ts   = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        line = f"\n## [{ts}] {title}\n{reasoning}\n"
        with open(self.dec_log, "a", encoding="utf-8") as f:
            f.write(line)

    def get_decisions(self) -> str:
        if not self.dec_log.exists():
            return "(no decisions logged yet)"
        return self.dec_log.read_text(encoding="utf-8")

    # ── SQLite task DB ────────────────────────────────────────────────────────
    def get_db(self) -> sqlite3.Connection:
        if self._db is None:
            self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    title    TEXT,
                    status   TEXT DEFAULT 'pending',
                    result   TEXT,
                    created  TEXT,
                    updated  TEXT
                )
            """)
            self._init_enhanced_tables(self._db)
            self._db.commit()
        return self._db

    def _init_enhanced_tables(self, db: sqlite3.Connection):
        db.execute("""
            CREATE TABLE IF NOT EXISTS task_groups (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                name     TEXT NOT NULL,
                feature  TEXT,
                status   TEXT DEFAULT 'pending',
                created  TEXT,
                updated  TEXT
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS enhanced_tasks (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id        INTEGER NOT NULL,
                title           TEXT NOT NULL,
                description     TEXT,
                agent_type      TEXT DEFAULT 'backend',
                status          TEXT DEFAULT 'pending',
                result          TEXT,
                output_summary  TEXT,
                files_written   TEXT,
                depends_on      TEXT DEFAULT '[]',
                created         TEXT,
                updated         TEXT,
                FOREIGN KEY (group_id) REFERENCES task_groups(id)
            )
        """)

    def add_task(self, title: str) -> int:
        now = datetime.utcnow().isoformat()
        cur = self.get_db().execute(
            "INSERT INTO tasks (title, status, created, updated) VALUES (?,?,?,?)",
            (title, "pending", now, now),
        )
        self.get_db().commit()
        return cur.lastrowid

    def update_task(self, task_id: int, status: str, result: str = ""):
        self.get_db().execute(
            "UPDATE tasks SET status=?, result=?, updated=? WHERE id=?",
            (status, result, datetime.utcnow().isoformat(), task_id),
        )
        self.get_db().commit()

    def list_tasks(self, limit: int = 20) -> list[dict]:
        rows = self.get_db().execute(
            "SELECT id, title, status, result, created FROM tasks ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"id": r[0], "title": r[1], "status": r[2], "result": r[3], "created": r[4]}
                for r in rows]

    # ── Task groups & dependency management ──────────────────────────────────
    def create_task_group(self, name: str, feature: str, tasks: list[dict]) -> dict:
        now = datetime.utcnow().isoformat()
        db = self.get_db()
        cur = db.execute(
            "INSERT INTO task_groups (name, feature, status, created, updated) VALUES (?,?,?,?,?)",
            (name, feature, "pending", now, now),
        )
        group_id = cur.lastrowid

        task_ids = []
        for t in tasks:
            cur = db.execute(
                "INSERT INTO enhanced_tasks "
                "(group_id, title, description, agent_type, status, depends_on, created, updated) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (group_id, t["title"], t.get("description", ""),
                 t.get("agent_type", "backend"), "pending", "[]", now, now),
            )
            task_ids.append(cur.lastrowid)

        # Resolve dependency indices → real IDs
        for i, t in enumerate(tasks):
            dep_indices = t.get("depends_on_indices", [])
            if dep_indices:
                dep_ids = [task_ids[idx] for idx in dep_indices if idx < len(task_ids)]
                db.execute(
                    "UPDATE enhanced_tasks SET depends_on=?, status='blocked' WHERE id=?",
                    (json.dumps(dep_ids), task_ids[i]),
                )
            else:
                db.execute(
                    "UPDATE enhanced_tasks SET status='ready' WHERE id=?",
                    (task_ids[i],),
                )
        db.commit()

        return {
            "group_id": group_id,
            "tasks": [
                {"id": task_ids[i], "title": tasks[i]["title"],
                 "status": "ready" if not tasks[i].get("depends_on_indices") else "blocked"}
                for i in range(len(tasks))
            ],
        }

    def get_ready_tasks(self, group_id: int) -> list[dict]:
        rows = self.get_db().execute(
            "SELECT id, title, description, agent_type FROM enhanced_tasks "
            "WHERE group_id=? AND status='ready'",
            (group_id,),
        ).fetchall()
        return [{"id": r[0], "title": r[1], "description": r[2], "agent_type": r[3]} for r in rows]

    def mark_task_running(self, task_id: int):
        self.get_db().execute(
            "UPDATE enhanced_tasks SET status='running', updated=? WHERE id=?",
            (datetime.utcnow().isoformat(), task_id),
        )
        self.get_db().commit()

    def mark_task_done(self, task_id: int, result: str, files_written: str = ""):
        now = datetime.utcnow().isoformat()
        self.get_db().execute(
            "UPDATE enhanced_tasks SET status='done', result=?, output_summary=?, "
            "files_written=?, updated=? WHERE id=?",
            (result[:2000], result[:500], files_written, now, task_id),
        )
        self.get_db().commit()

    def mark_task_failed(self, task_id: int, error: str):
        self.get_db().execute(
            "UPDATE enhanced_tasks SET status='failed', result=?, updated=? WHERE id=?",
            (error[:2000], datetime.utcnow().isoformat(), task_id),
        )
        self.get_db().commit()

    def resolve_dependencies(self, group_id: int) -> int:
        db = self.get_db()
        blocked = db.execute(
            "SELECT id, depends_on FROM enhanced_tasks WHERE group_id=? AND status='blocked'",
            (group_id,),
        ).fetchall()

        unblocked = 0
        for task_id, deps_json in blocked:
            dep_ids = json.loads(deps_json)
            if not dep_ids:
                db.execute("UPDATE enhanced_tasks SET status='ready' WHERE id=?", (task_id,))
                unblocked += 1
                continue
            placeholders = ",".join("?" * len(dep_ids))
            done_count = db.execute(
                f"SELECT COUNT(*) FROM enhanced_tasks WHERE id IN ({placeholders}) AND status='done'",
                dep_ids,
            ).fetchone()[0]
            if done_count == len(dep_ids):
                db.execute(
                    "UPDATE enhanced_tasks SET status='ready', updated=? WHERE id=?",
                    (datetime.utcnow().isoformat(), task_id),
                )
                unblocked += 1
        db.commit()
        return unblocked

    def get_group_status(self, group_id: int) -> dict:
        rows = self.get_db().execute(
            "SELECT id, title, agent_type, status, output_summary, files_written "
            "FROM enhanced_tasks WHERE group_id=?",
            (group_id,),
        ).fetchall()
        counts: dict[str, int] = {}
        tasks = []
        for r in rows:
            counts[r[3]] = counts.get(r[3], 0) + 1
            tasks.append({
                "id": r[0], "title": r[1], "agent_type": r[2],
                "status": r[3], "summary": r[4], "files": r[5],
            })
        return {
            "group_id": group_id, "total": len(rows), "counts": counts,
            "all_done": counts.get("done", 0) == len(rows),
            "has_failures": counts.get("failed", 0) > 0,
            "tasks": tasks,
        }

    def get_task_results_for_review(self, group_id: int) -> list[dict]:
        rows = self.get_db().execute(
            "SELECT id, title, agent_type, result, files_written FROM enhanced_tasks "
            "WHERE group_id=? AND status='done'",
            (group_id,),
        ).fetchall()
        return [{"id": r[0], "title": r[1], "agent_type": r[2],
                 "result": r[3], "files": r[4]} for r in rows]

    # ── Context string injected into every Opus prompt ───────────────────────
    def build_context_prompt(self) -> str:
        data = self.load()
        if not data:
            return ""

        recent_tasks = self.list_tasks(10)
        task_lines = "\n".join(
            f"  {'✅' if t['status']=='done' else '⏳' if t['status']=='running' else '❌' if t['status']=='failed' else '🕐'} "
            f"#{t['id']} {t['title']}"
            for t in recent_tasks
        ) or "  (none yet)"

        # Summarise workspace structure
        ws_tree = self._workspace_tree(depth=2)

        return f"""
=== ACTIVE PROJECT: {data.get('name', self.slug)} ===

DESCRIPTION:
{data.get('description', 'N/A')}

TECH STACK:
  Backend  : {data.get('backend', 'TBD')}
  Frontend : {data.get('frontend', 'TBD')}
  Data     : {data.get('data_layer', 'TBD')}
  Database : {data.get('database', 'TBD')}
  Auth     : {data.get('auth', 'TBD')}

CONVENTIONS:
{data.get('conventions', 'TBD')}

GITHUB REPO:
{data.get('github_repo', 'not set yet')}

WORKSPACE STRUCTURE:
{ws_tree}

RECENT TASKS:
{task_lines}

ARCHITECTURE DECISIONS:
{self.get_decisions()}

=== END PROJECT CONTEXT ===
"""

    def _workspace_tree(self, depth: int = 2) -> str:
        if not self.workspace.exists():
            return "(empty)"
        lines = []
        def _walk(path: Path, prefix: str, current_depth: int):
            if current_depth > depth:
                return
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
            for i, entry in enumerate(entries):
                connector = "└── " if i == len(entries) - 1 else "├── "
                lines.append(f"{prefix}{connector}{entry.name}")
                if entry.is_dir() and current_depth < depth:
                    extension = "    " if i == len(entries) - 1 else "│   "
                    _walk(entry, prefix + extension, current_depth + 1)
        try:
            _walk(self.workspace, "", 0)
        except Exception:
            pass
        return "\n".join(lines) or "(empty)"

    def build_enhanced_context_prompt(self) -> str:
        base = self.build_context_prompt()
        try:
            active_groups = self.get_db().execute(
                "SELECT id, name, status FROM task_groups "
                "WHERE status != 'done' ORDER BY id DESC LIMIT 3"
            ).fetchall()
            if active_groups:
                lines = ["\nACTIVE TASK GROUPS:"]
                for g in active_groups:
                    status = self.get_group_status(g[0])
                    lines.append(f"  Group #{g[0]} '{g[1]}': {status['counts']}")
                base += "\n".join(lines)
        except Exception:
            pass
        return base


# ── Persistent user → active project mapping ─────────────────────────────────
_ACTIVE_FILE = Path("./data/active_projects.json")
_ACTIVE_FILE.parent.mkdir(parents=True, exist_ok=True)

_active: dict[int, str] = {}   # user_id → project slug
_active_loaded = False


def _load_active():
    global _active, _active_loaded
    if _active_loaded:
        return
    _active_loaded = True
    if _ACTIVE_FILE.exists():
        try:
            data = json.loads(_ACTIVE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _active = {int(k): v for k, v in data.items()}
        except Exception:
            pass


def _save_active():
    try:
        _ACTIVE_FILE.write_text(
            json.dumps({str(k): v for k, v in _active.items()}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def get_active(user_id: int) -> Project | None:
    _load_active()
    slug = _active.get(user_id)
    if not slug:
        return None
    p = Project(slug)
    return p if p.exists() else None


def set_active(user_id: int, slug: str):
    _load_active()
    _active[user_id] = slug
    _save_active()


def clear_active(user_id: int):
    _load_active()
    _active.pop(user_id, None)
    _save_active()
