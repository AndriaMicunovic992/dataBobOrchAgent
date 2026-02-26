"""
agent.py — Claude AI agent for generating financial scenarios.

Provides:
  Agent   — wraps the Anthropic API, handles tool calls, stages adjustments,
            and generates SQL when the user explicitly applies a scenario.
"""

import os
import re
import json

import anthropic

from config import CLAUDE_MODEL, COMPANY_ID, REVENUE_ACCS, COGS_ACCS
from pbi_client import PBIClient
from cache import cache_save, cache_load
from dax import fetch_budget
from scenario import build_scenario, make_sql, save_sql


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a financial scenario specialist for the FINANCE Power BI model (Hans Kohler AG).

== DATA MODEL ==
Fakten Hauptbuch (GL ledger): The table where scenarios are written.
  Budget grain: account × date × cost_object_id. This is what run_dax_query fetches.
  Each row has: account (ID + GL-Nr. + name), date, amount, budget_amount, cost_object (ID + name).
Fakten Rechnungszeile (Invoice lines): Actual sales at customer × item level.
  Budget rows here have customer_id=NULL — aggregated, not per-customer.
Dim Kunde: Customer master. Columns: customer_id (int), Kundenname, Kundenname Alias.

== GL ACCOUNTS ==
Always call run_dax_query before staging adjustments — the result includes a full
account breakdown with account_nr (D365 GL number), account_name, and account_grp
(reporting group) pulled live from the semantic model. Use those names when talking
to the user and those IDs when writing adjustment blocks.

Revenue accounts: account_group="revenue" targets all revenue IDs.
COGS accounts:    account_group="cogs"    targets all COGS IDs.
Specific subset:  account_group="112,114" (comma-separated IDs).
Reporting group:  list the individual IDs that belong to it, e.g. "122,123,124".

== CUSTOMER SCENARIOS ==
Since budget invoice lines have no customer_id, customer scenarios are modelled via GL accounts:
1. Call query_customers → get customer's actual revenue share % from prior year actuals
2. Calculate net GL impact: customer_share% × requested_change% = GL_adjustment%
   Example: customer is 8.5% of revenue, reduce by 25% → GL adjustment = -(8.5% × 25%) = -2.125%
3. Call run_dax_query to load GL budget baseline
4. Apply calculated GL_adjustment% to revenue accounts

Always show the user the maths before staging.

== STAGING WORKFLOW ==
Adjustments are STAGED across multiple turns. SQL is only generated when the user
explicitly asks to "generate", "apply", or "create" the scenario (with a name).

Step 1 — STAGE adjustments (one or more turns):
- For customer-based requests  → call query_customers first, then run_dax_query
- For account/group requests   → call run_dax_query directly
- Once confirmed, output a ```stage block (see format below)
- After staging, summarise what was staged and list ALL staged adjustments so far
  (so the user can keep building before generating SQL)

Step 2 — GENERATE SQL (only when user provides a scenario name and asks to generate):
- Output a ```apply block (see format below)
- The server will collect ALL staged adjustments, apply them to the budget baseline,
  and write a SQL INSERT file.

== STAGE FORMAT ==
Output EXACTLY this block to accumulate one adjustment group (no extra text inside it):
```stage
{
  "description": "Brief human-readable description shown in the sidebar",
  "adjustments": [
    {"months": [2], "account_group": "revenue", "pct_change": 2.0}
  ]
}
```

== APPLY FORMAT ==
Output EXACTLY this block to trigger SQL generation for ALL staged adjustments:
```apply
{
  "label": "short_name_no_spaces",
  "description": "Human-readable one-liner describing the full scenario"
}
```

Rules:
- months: 1-12 list, or omit for full year.
- account_group: "revenue", "cogs", "all", or comma-separated account IDs e.g. "112,114".
- pct_change: always the BUSINESS direction — never flip the sign because a GL account
  happens to be stored as a negative number.

  COGS accounts are stored as NEGATIVE amounts (e.g., -80 000). The formula
  amount × (1 + pct/100) handles this correctly without any sign adjustment:

    User says "increase costs 5%"  → pct_change = +5  → -80 000 × 1.05 = -84 000  ✓ (more costly)
    User says "decrease costs 5%"  → pct_change = -5  → -80 000 × 0.95 = -76 000  ✓ (less costly)

  DO NOT stage a negative pct_change just because the GL value is negative.
  A positive pct_change always makes costs larger in absolute terms.

- Never output both a ```stage and ```apply block in the same message.
- Never mention caches, files, sessions, or backends."""


# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "run_dax_query",
        "description": "Fetch budget data from Fakten Hauptbuch (GL). Returns a summary; data stored internally.",
        "input_schema": {
            "type": "object",
            "required": ["year"],
            "properties": {
                "year":   {"type": "integer", "description": "Fiscal year e.g. 2026"},
                "months": {"type": "array", "items": {"type": "integer"},
                           "description": "Specific months 1-12. Omit for full year."},
            },
        },
    },
    {
        "name": "query_customers",
        "description": (
            "Look up customers and their actual revenue share from Fakten Rechnungszeile. "
            "Use this when the user mentions a customer name, 'biggest customer', 'top N customers', "
            "or any customer-specific scenario. Returns customer IDs, names, revenue totals, and "
            "their % share of total company revenue — which can then be used to calculate the "
            "proportional impact on GL budget rows."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "year":        {"type": "integer", "description": "Year for actuals lookup (default 2025)"},
                "top_n":       {"type": "integer", "description": "Return top N customers by revenue (default 10)"},
                "search_name": {"type": "string",  "description": "Optional: search by customer name substring"},
            },
        },
    },
]


# ── Helper functions ──────────────────────────────────────────────────────────

def data_summary(rows: list[dict]) -> str:
    """
    Return a rich text summary of loaded budget rows.

    Account names, GL numbers, and reporting groups come from the enriched row
    fields added by fetch_budget() (live from Dim Hauptkonto) — no static map
    needed here.
    """
    months:      set[str]         = set()
    acct_totals: dict[int, float] = {}
    acct_meta:   dict[int, dict]  = {}   # first-seen metadata per account

    for r in rows:
        months.add(r["date"][:7])
        acc = r["account"]
        acct_totals[acc] = acct_totals.get(acc, 0.0) + r["amount"]
        if acc not in acct_meta:
            acct_meta[acc] = {
                "nr":   r.get("account_nr",   str(acc)),
                "name": r.get("account_name", f"Account {acc}"),
                "grp":  r.get("account_grp",  "—"),
            }

    rev   = sum(v for acc, v in acct_totals.items() if acc in REVENUE_ACCS)
    cogs  = sum(v for acc, v in acct_totals.items() if acc in COGS_ACCS)
    total = sum(acct_totals.values())
    ms    = sorted(months)

    lines = [
        f"Budget loaded: {len(rows)} rows | {len(acct_totals)} accounts "
        f"| {ms[0]} to {ms[-1]}",
        f"  Revenue : CHF {rev:>15,.0f}",
        f"  COGS    : CHF {cogs:>15,.0f}",
        f"  Total   : CHF {total:>15,.0f}",
        "",
        "Account breakdown:",
        f"  {'ID':>4}  {'GL-Nr.':>8}  {'Group':<26}  {'Name':<36}  {'Amount CHF':>14}",
        "  " + "─" * 96,
    ]

    for acc in sorted(acct_totals.keys()):
        meta = acct_meta[acc]
        lines.append(
            f"  {acc:>4}  {meta['nr']:>8}  {meta['grp']:<26}  {meta['name']:<36}"
            f"  {acct_totals[acc]:>14,.0f}"
        )

    # Cost-object summary: collect names from enriched rows
    co_names: dict[int, str] = {}
    for r in rows:
        cid  = r.get("cost_object_id")
        name = r.get("cost_object_name")
        if cid is not None and name and cid not in co_names:
            co_names[cid] = name

    if co_names:
        lines += ["", f"  Cost objects ({len(co_names)} unique):"]
        for cid in sorted(co_names.keys())[:8]:
            lines.append(f"    {cid}: {co_names[cid]}")
        if len(co_names) > 8:
            lines.append(f"    … and {len(co_names) - 8} more")

    lines += ["", "Ready — tell me the adjustments to stage."]
    return "\n".join(lines)


def extract_stage_block(text: str) -> dict | None:
    """Parse a ```stage JSON block from Claude's response."""
    m = re.search(r"```stage\s*(\{[\s\S]*?\})\s*```", text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception as e:
        print(f"[Agent] Bad stage JSON: {e}\n{m.group(1)}")
        return None


def extract_apply_block(text: str) -> dict | None:
    """Parse an ```apply JSON block from Claude's response."""
    m = re.search(r"```apply\s*(\{[\s\S]*?\})\s*```", text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception as e:
        print(f"[Agent] Bad apply JSON: {e}\n{m.group(1)}")
        return None


# ── Agent ─────────────────────────────────────────────────────────────────────

class Agent:
    """
    Wraps the Anthropic Claude API with tool-use for financial scenario generation.

    Adjustments are staged across multiple turns via ```stage blocks.
    SQL is only generated when the user triggers a ```apply block.
    Each applied scenario gets a unique, incrementing scenario_id (value_type_id).
    """

    def __init__(self, pbi: PBIClient):
        self.pbi              = pbi
        self.ai               = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.conv : list[dict] = []
        self.rows : list[dict] = []   # in-memory budget data
        self.staged: list[dict] = []  # staged adjustment groups [{description, adjustments}]
        self.next_scenario_id  = 3    # increments with each applied scenario

    # ── Staged adjustments ────────────────────────────────────────────────────

    def get_staged(self) -> dict:
        """Return staged adjustments and metadata for the UI."""
        return {
            "staged":           self.staged,
            "next_id":          self.next_scenario_id,
            "adjustment_count": sum(len(s["adjustments"]) for s in self.staged),
        }

    def clear_staged(self):
        """Discard all staged adjustments."""
        self.staged = []
        print("[Agent] Staged adjustments cleared.")

    def remove_staged(self, index: int) -> bool:
        """Remove a single staged step by list index. Returns True if removed."""
        if 0 <= index < len(self.staged):
            removed = self.staged.pop(index)
            print(f"[Agent] Removed staged step {index}: {removed.get('description', '')!r}")
            return True
        return False

    # ── Tool execution ────────────────────────────────────────────────────────

    async def _handle_tool(self, name: str, inp: dict) -> str:
        if name == "run_dax_query":
            rows = await fetch_budget(self.pbi, inp.get("year", 2026), inp.get("months"))
            if not rows:
                return "DAX returned no rows. Check the connection and filters."
            self.rows = rows
            cache_save(rows)
            return data_summary(rows)

        if name == "query_customers":
            year   = inp.get("year", 2025)
            top_n  = inp.get("top_n", 10)
            search = inp.get("search_name", "").strip()

            # Top-N customers by actual revenue from invoice lines.
            # TOPN selects the right rows but does NOT sort its output —
            # ORDER BY is required to get rank 1 = highest revenue.
            dax_top = f"""EVALUATE
TOPN({top_n},
  SUMMARIZECOLUMNS(
    'Fakten Rechnungszeile'[customer_id],
    FILTER('Fakten Rechnungszeile',
      'Fakten Rechnungszeile'[company_id]    = {COMPANY_ID}
      && 'Fakten Rechnungszeile'[value_type_id] = 1
      && YEAR('Fakten Rechnungszeile'[invoice_date]) = {year}
      && NOT ISBLANK('Fakten Rechnungszeile'[customer_id])
    ),
    "revenue", SUM('Fakten Rechnungszeile'[actual_revenue])
  ),
  [revenue], DESC
)
ORDER BY [revenue] DESC"""
            resp1    = await self.pbi.dax(dax_top)
            top_rows = resp1.get("data", {}).get("rows", [])
            if not top_rows:
                return f"No customer revenue data found for {year}."

            # Total company revenue for share calculation
            dax_total = f"""EVALUATE
ROW("total",
  CALCULATE(
    SUM('Fakten Rechnungszeile'[actual_revenue]),
    'Fakten Rechnungszeile'[company_id]    = {COMPANY_ID},
    'Fakten Rechnungszeile'[value_type_id] = 1,
    YEAR('Fakten Rechnungszeile'[invoice_date]) = {year}
  )
)"""
            resp2     = await self.pbi.dax(dax_total)
            total_rev = float(
                (resp2.get("data", {}).get("rows", [{}])[0]).get("[total]", 1) or 1
            )

            # Customer names from Dim Kunde
            cust_ids  = [int(r.get("Fakten Rechnungszeile[customer_id]", 0)) for r in top_rows]
            ids_str   = ", ".join(str(i) for i in cust_ids)
            dax_names = f"""EVALUATE
SELECTCOLUMNS(
  FILTER('Dim Kunde', 'Dim Kunde'[customer_id] IN {{{ids_str}}}),
  "id",    'Dim Kunde'[customer_id],
  "name",  'Dim Kunde'[Kundenname],
  "alias", 'Dim Kunde'[Kundenname Alias]
)"""
            resp3    = await self.pbi.dax(dax_names)
            name_map = {
                int(r["[id]"]): (r["[name]"], r["[alias]"])
                for r in resp3.get("data", {}).get("rows", [])
            }

            # Optional name search
            search_result = []
            if search:
                dax_search = f"""EVALUATE
SELECTCOLUMNS(
  FILTER('Dim Kunde',
    SEARCH("{search}", 'Dim Kunde'[Kundenname], 1, 0) > 0
    || SEARCH("{search}", 'Dim Kunde'[Kundenname Alias], 1, 0) > 0
  ),
  "id",    'Dim Kunde'[customer_id],
  "name",  'Dim Kunde'[Kundenname],
  "alias", 'Dim Kunde'[Kundenname Alias]
)"""
                resp4 = await self.pbi.dax(dax_search)
                for r in resp4.get("data", {}).get("rows", []):
                    search_result.append(
                        f"  • customer_id={int(r['[id]'])} — {r['[name]']} ({r['[alias]']})"
                    )

            # Build result table
            lines = [f"Top {top_n} customers by actual revenue {year} "
                     f"(total company: CHF {total_rev:,.0f}):", ""]
            lines.append(f"  {'Rank':<5} {'customer_id':<13} {'Revenue CHF':>14} {'Share':>7}  Name")
            lines.append("  " + "─" * 70)
            for rank, r in enumerate(top_rows, 1):
                cid         = int(r.get("Fakten Rechnungszeile[customer_id]", 0))
                rev         = float(r.get("[revenue]", 0))
                share       = rev / total_rev * 100
                name, alias = name_map.get(cid, ("—", "—"))
                lines.append(
                    f"  {rank:<5} {cid:<13} {rev:>14,.0f} {share:>6.1f}%  {name} ({alias})"
                )

            lines += [
                "",
                "NOTE: Budget rows in Fakten Rechnungszeile have no customer_id "
                "(aggregated at cost_object level).",
                "To model a customer revenue change, use their revenue SHARE to scale "
                "the matching GL accounts",
                "in Fakten Hauptbuch. For example: if customer X = 8.5% of revenue, "
                "reducing them by 25%",
                "means a -2.1% adjustment on total revenue GL accounts "
                "(8.5% × 25% = 2.125%).",
            ]
            if search_result:
                lines += ["", f"Search results for '{search}':"] + search_result

            return "\n".join(lines)

        return f"Unknown tool: {name}"

    # ── Main chat loop ────────────────────────────────────────────────────────

    async def chat(self, msg: str) -> str:
        """
        Send a user message, handle any tool calls, and return the agent's reply.

        - ```stage blocks accumulate adjustments in self.staged (no SQL yet)
        - ```apply blocks consume ALL of self.staged to generate and save one SQL file
        """
        if not self.rows:
            self.rows = cache_load()

        self.conv.append({"role": "user", "content": msg})

        while True:
            resp = self.ai.messages.create(
                model=CLAUDE_MODEL, max_tokens=1024,
                system=SYSTEM_PROMPT, tools=TOOLS, messages=self.conv,
            )
            self.conv.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason == "tool_use":
                results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        print(f"[Tool] {block.name}({block.input})")
                        result = await self._handle_tool(block.name, block.input)
                        results.append({
                            "type":        "tool_result",
                            "tool_use_id": block.id,
                            "content":     result,
                        })
                self.conv.append({"role": "user", "content": results})
                continue

            # end_turn — extract text
            text = "".join(b.text for b in resp.content if hasattr(b, "text"))

            # ── Stage block: accumulate adjustments ──────────────────────────
            stage = extract_stage_block(text)
            if stage:
                self.staged.append(stage)
                adj_count = len(stage["adjustments"])
                total_adjs = sum(len(s["adjustments"]) for s in self.staged)
                print(f"[Stage] +{adj_count} adjustment(s). Total groups staged: {len(self.staged)}, total adjustments: {total_adjs}")
                clean = re.sub(r"```stage[\s\S]*?```", "", text).strip()
                return clean

            # ── Apply block: generate SQL from all staged adjustments ────────
            apply_spec = extract_apply_block(text)
            if apply_spec:
                if not self.staged:
                    return (re.sub(r"```apply[\s\S]*?```", "", text).strip() +
                            "\n\n⚠️  Nothing staged yet — describe your adjustments first.")
                if not self.rows:
                    return (re.sub(r"```apply[\s\S]*?```", "", text).strip() +
                            "\n\n⚠️  No budget data loaded — please fetch the budget first.")

                # Flatten all staged adjustment groups
                all_adjs = []
                for s in self.staged:
                    all_adjs.extend(s["adjustments"])

                scenario_id = self.next_scenario_id
                print(f"[SQL] Applying {len(all_adjs)} adjustment(s) from "
                      f"{len(self.staged)} group(s) as scenario_id={scenario_id} ...")

                sc   = build_scenario(self.rows, all_adjs)
                sql  = make_sql(sc, apply_spec["label"],
                                apply_spec.get("description", ""),
                                scenario_id=scenario_id)
                path = save_sql(sql, apply_spec["label"], scenario_id=scenario_id)
                dates = sorted({r["date"] for r in sc})

                # Advance counter and clear staging area for next scenario
                self.next_scenario_id += 1
                self.staged = []

                clean = re.sub(r"```apply[\s\S]*?```", "", text).strip()
                return (
                    f"{clean}\n\n"
                    f"✅ Scenario {scenario_id} SQL saved: {path}\n"
                    f"   {len(sc)} rows | {len(all_adjs)} adjustments | {', '.join(dates)}"
                )

            return text

    def reset(self):
        """Clear conversation history and staged adjustments (budget data is preserved)."""
        self.conv   = []
        self.staged = []
        print("[Agent] Conversation and staging area reset.")
