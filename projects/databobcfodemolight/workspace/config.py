"""
config.py — Central configuration for the FINANCE Scenario Agent.

All constants are defined here. Import from this module everywhere else
so there is a single place to change paths, account lists, etc.
"""

import tempfile
from pathlib import Path

# ── Power BI MCP executable ───────────────────────────────────────────────────
POWERBI_EXE = (
    r"C:\Users\andri\.vscode\extensions"
    r"\analysis-services.powerbi-modeling-mcp-0.1.9-win32-x64"
    r"\server\powerbi-modeling-mcp.exe"
)

# ── Model constants ───────────────────────────────────────────────────────────
COMPANY_ID   = 4
REVENUE_ACCS = {112, 114, 118, 119, 120, 121, 122, 123, 124, 130}
COGS_ACCS    = {126, 127}

# ── Claude model ──────────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-6"

# ── File paths ────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent.resolve() / "output"
CACHE_DB   = Path(tempfile.gettempdir()) / "finance_scenario_cache.db"
