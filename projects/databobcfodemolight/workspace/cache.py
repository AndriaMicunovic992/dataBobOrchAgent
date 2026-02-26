"""
cache.py — SQLite row cache for budget data.

Persists fetched budget rows between server restarts so the agent can
continue a conversation without re-fetching from Power BI.

Row format: [{"account": 112, "date": "2026-01-01", "amount": 7837544.07, ...}, ...]
All keys are plain strings (no brackets), dates are YYYY-MM-DD strings.
"""

import json
import sqlite3

from config import CACHE_DB


def cache_save(rows: list[dict]):
    """Persist rows to the SQLite cache, replacing any previous data."""
    assert rows, "Refusing to save empty list"
    with sqlite3.connect(str(CACHE_DB), timeout=5) as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("CREATE TABLE IF NOT EXISTS rows (data TEXT NOT NULL)")
        con.execute("DELETE FROM rows")
        con.execute("INSERT INTO rows VALUES (?)", (json.dumps(rows),))
    print(f"[Cache] Saved {len(rows)} rows → {CACHE_DB}")


def cache_load() -> list[dict]:
    """Load rows from the SQLite cache. Returns [] if cache is empty or missing."""
    if not CACHE_DB.exists():
        return []
    try:
        with sqlite3.connect(str(CACHE_DB), timeout=5) as con:
            con.execute("PRAGMA journal_mode=WAL")
            row = con.execute("SELECT data FROM rows LIMIT 1").fetchone()
        if not row:
            return []
        rows = json.loads(row[0])
        print(f"[Cache] Loaded {len(rows)} rows from {CACHE_DB}")
        return rows
    except sqlite3.OperationalError as e:
        print(f"[Cache] DB error ({e}), deleting and starting fresh")
        try:
            CACHE_DB.unlink()
        except Exception:
            pass
        return []
