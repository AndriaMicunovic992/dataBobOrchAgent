#!/usr/bin/env python3
"""
FINANCE Scenario Agent — Web Server
====================================
Wraps scenario_agent.py in a Flask HTTP server so the HTML UI can talk to it.

Usage:
    pip install flask
    python server.py
Then open http://localhost:5000 in your browser.
"""

import asyncio, os, sys, threading, json, webbrowser
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

# Ensure the project directory is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUT_DIR, CACHE_DB
from pbi_client import PBIClient, list_pbi_instances
from scenario import build_scenario
from agent import Agent

app = Flask(__name__, static_folder=str(Path(__file__).parent))

# ── Single shared agent instance (thread-safe via asyncio lock) ───────────────
_loop   = asyncio.new_event_loop()
_lock   = threading.Lock()
_pbi    : PBIClient | None = None
_agent  : Agent | None = None
_status = {"connected": False, "message": "Not connected"}

def _run(coro):
    """Run a coroutine on the background event loop."""
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result(timeout=120)

def _start_loop():
    asyncio.set_event_loop(_loop)
    _loop.run_forever()

threading.Thread(target=_start_loop, daemon=True).start()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(Path(__file__).parent), "ui.html")


@app.route("/api/instances")
def get_instances():
    """
    Discover all open Power BI Desktop models.
    Returns a list of {display_name, connection_string, database, port}.
    This starts a temporary MCP session — no persistent connection is made.
    """
    try:
        instances = _run(list_pbi_instances())
        return jsonify({"ok": True, "instances": instances})
    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e),
                        "trace": traceback.format_exc(), "instances": []})


@app.route("/api/connect", methods=["POST"])
def connect():
    """
    Connect to a specific Power BI Desktop model.
    Body: { "connection_string": "...", "database": "..." }
    """
    global _pbi, _agent
    with _lock:
        data     = request.get_json() or {}
        conn_str = data.get("connection_string", "").strip()
        db_guid  = data.get("database", "").strip()

        if not conn_str or not db_guid:
            return jsonify({"ok": False,
                            "message": "connection_string and database are required"}), 400

        try:
            _pbi = PBIClient()
            _run(_pbi.connect(conn_str, db_guid))
            _agent = Agent(_pbi)
            _status["connected"] = True
            _status["message"]   = "Connected to Power BI Desktop"
            return jsonify({"ok": True, "message": _status["message"]})
        except Exception as e:
            _status["connected"] = False
            _status["message"]   = str(e)
            # Still create agent so UI stays functional
            _pbi   = PBIClient()
            _agent = Agent(_pbi)
            return jsonify({"ok": False, "message": f"Connection failed: {e}"})


@app.route("/api/status")
def status():
    return jsonify({
        "connected":   _status["connected"],
        "message":     _status["message"],
        "cache_db":    str(CACHE_DB),
        "output_dir":  str(OUTPUT_DIR),
        "agent_ready": _agent is not None,
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    global _agent, _pbi
    data = request.get_json()
    msg  = (data or {}).get("message", "").strip()
    if not msg:
        return jsonify({"ok": False, "error": "Empty message"}), 400

    with _lock:
        if _agent is None:
            _pbi   = PBIClient()
            _agent = Agent(_pbi)
        try:
            reply = _run(_agent.chat(msg))
            sql_files = sorted(OUTPUT_DIR.glob("scenario_*.sql"),
                               key=lambda f: f.stat().st_mtime, reverse=True)
            latest_sql = str(sql_files[0]) if sql_files else None
            return jsonify({"ok": True, "reply": reply, "latest_sql": latest_sql})
        except Exception as e:
            import traceback
            return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()})


@app.route("/api/reset", methods=["POST"])
def reset():
    with _lock:
        if _agent:
            _agent.reset()
    return jsonify({"ok": True})


@app.route("/api/scenario/staged")
def get_staged():
    """Return the current staged adjustments and next scenario_id."""
    with _lock:
        if _agent is None:
            return jsonify({"staged": [], "next_id": 3, "adjustment_count": 0})
        return jsonify(_agent.get_staged())


@app.route("/api/scenario/clear", methods=["POST"])
def clear_staged():
    """Discard all staged adjustments without generating SQL."""
    with _lock:
        if _agent:
            _agent.clear_staged()
    return jsonify({"ok": True})


@app.route("/api/scenario/staged/<int:index>", methods=["DELETE"])
def remove_staged_step(index):
    """Remove a single staged step by its list index."""
    with _lock:
        if _agent is None:
            return jsonify({"ok": False, "error": "Agent not initialised"}), 400
        removed = _agent.remove_staged(index)
        if not removed:
            return jsonify({"ok": False, "error": f"No staged step at index {index}"}), 400
    return jsonify({"ok": True})


@app.route("/api/scenario/preview")
def scenario_preview():
    """
    Preview the GL impact of all staged adjustments.

    Applies adjustments to the in-memory budget rows and returns a pivot:
      accounts × months → {original, scenario, delta}
    plus column totals and row totals so the UI can render an impact table.
    """
    with _lock:
        if _agent is None:
            return jsonify({"ok": False, "error": "Agent not initialised"})
        if not _agent.rows:
            return jsonify({"ok": False,
                            "error": "No budget data loaded — ask the agent to load the budget first."})
        if not _agent.staged:
            return jsonify({"ok": False, "error": "No adjustments staged yet."})

        # Flatten all staged groups into one adjustment list
        all_adjs = []
        for s in _agent.staged:
            all_adjs.extend(s["adjustments"])

        orig_rows = _agent.rows
        sc_rows   = build_scenario(orig_rows, all_adjs)

        # Aggregate amounts by (account, YYYY-MM)
        orig_pivot: dict[tuple, float] = {}
        for r in orig_rows:
            key = (r["account"], r["date"][:7])
            orig_pivot[key] = orig_pivot.get(key, 0.0) + r["amount"]

        sc_pivot: dict[tuple, float] = {}
        for r in sc_rows:
            key = (r["account"], r["date"][:7])
            sc_pivot[key] = sc_pivot.get(key, 0.0) + r["amount"]

        account_ids = sorted({r["account"] for r in orig_rows})
        months      = sorted({r["date"][:7] for r in orig_rows})

        # Collect account metadata from enriched rows (account_nr / account_name / account_grp
        # are added by fetch_budget → fetch_account_map).
        acc_meta: dict[int, dict] = {}
        for r in orig_rows:
            acc = r["account"]
            if acc not in acc_meta:
                acc_meta[acc] = {
                    "id":   acc,
                    "nr":   r.get("account_nr",   str(acc)),
                    "name": r.get("account_name", f"Account {acc}"),
                    "grp":  r.get("account_grp",  ""),
                }

        # Build per-account rows with per-month and total delta
        result_accounts = []
        for acc in account_ids:
            meta = acc_meta[acc]
            month_data = {}
            for m in months:
                key  = (acc, m)
                orig = round(orig_pivot.get(key, 0.0), 2)
                scen = round(sc_pivot.get(key, 0.0),   2)
                month_data[m] = {
                    "original": orig,
                    "scenario": scen,
                    "delta":    round(scen - orig, 2),
                }
            row_orig = sum(v["original"] for v in month_data.values())
            row_scen = sum(v["scenario"] for v in month_data.values())
            result_accounts.append({
                **meta,
                "months": month_data,
                "total":  {
                    "original": round(row_orig, 2),
                    "scenario": round(row_scen, 2),
                    "delta":    round(row_scen - row_orig, 2),
                },
            })

        # Column totals (sum across all accounts per month)
        col_totals: dict[str, dict] = {}
        for m in months:
            c_orig = sum(orig_pivot.get((acc, m), 0.0) for acc in account_ids)
            c_scen = sum(sc_pivot.get((acc, m),  0.0) for acc in account_ids)
            col_totals[m] = {
                "original": round(c_orig, 2),
                "scenario": round(c_scen, 2),
                "delta":    round(c_scen - c_orig, 2),
            }

        grand_orig = sum(v["original"] for v in col_totals.values())
        grand_scen = sum(v["scenario"] for v in col_totals.values())
        col_totals["total"] = {
            "original": round(grand_orig, 2),
            "scenario": round(grand_scen, 2),
            "delta":    round(grand_scen - grand_orig, 2),
        }

        return jsonify({
            "ok":       True,
            "months":   months,
            "accounts": result_accounts,
            "totals":   col_totals,
        })


@app.route("/api/files")
def list_files():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(OUTPUT_DIR.glob("scenario_*.sql"),
                   key=lambda f: f.stat().st_mtime, reverse=True)
    return jsonify([{
        "name":  f.name,
        "path":  str(f),
        "size":  f.stat().st_size,
        "mtime": f.stat().st_mtime,
    } for f in files[:20]])


@app.route("/api/file/<filename>")
def get_file(filename):
    f = OUTPUT_DIR / filename
    if not f.exists() or f.suffix != ".sql":
        return jsonify({"error": "Not found"}), 404
    return jsonify({"name": f.name, "content": f.read_text(encoding="utf-8")})


# ── Start ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY"); sys.exit(1)

    port = 5000
    print(f"Starting FINANCE Scenario Agent UI on http://localhost:{port}")
    print("Opening browser...")
    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
