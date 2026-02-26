# FINANCE Scenario Agent

A specialist AI agent for generating Power BI financial scenarios from budget baselines.

## What it does

- **Reads** budget data (value_type_id=2) from `Fakten Hauptbuch` via DAX queries
- **Calculates** scenario adjustments (% changes on revenue, COGS, or any account group)
- **Generates** SQL INSERT scripts ready to load as scenario data (value_type_id=3)
- **Remembers** context across a conversation so you can refine scenarios iteratively

---

## Setup

### 1. Install dependencies

```bash
pip install anthropic mcp
```

### 2. Set your Anthropic API key

```bash
# Windows
set ANTHROPIC_API_KEY=

# macOS / Linux
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Make sure Power BI Desktop is running

The FINANCE model must be open in Power BI Desktop.
The default connection is `localhost:51583`.

If your port differs, edit this line in `scenario_agent.py`:
```python
POWERBI_CONNECTION_STRING = "localhost:51583"
```

### 4. Run the agent

```bash
python scenario_agent.py
```

---

## Usage examples

Once running, type natural language requests:

```
You: Create a 2026 scenario with +2% revenue in February and +5% COGS in March

You: Add a -3% adjustment to all operating expenses in Q2

You: Show me the summary of adjustments before generating the SQL

You: Generate the SQL file
```

The agent will:
1. Fetch the relevant months from Power BI via DAX
2. Show you a summary of the adjustments
3. Ask for confirmation
4. Generate and save a `.sql` file in the `./output/` folder

---

## Output files

SQL scripts are saved to `./output/` with timestamped names, e.g.:
```
output/scenario_Rev+2pct_Feb_COGS+5pct_Mar_20260224_143021.sql
```

Each script includes:
- A header comment with the adjustment description
- A commented-out DELETE statement for safe re-loading
- `INSERT INTO [Fakten Hauptbuch]` with all 93 accounts × N months
- A verification SELECT at the bottom

---

## Model constants (pre-configured)

| Setting | Value |
|---------|-------|
| Company | Hans Kohler AG (company_id = 4) |
| Budget value_type_id | 2 |
| Scenario value_type_id | 3 |
| Revenue accounts | 112, 114, 118, 119, 120, 121, 122, 123, 124, 130 |
| COGS accounts | 126, 127 |
| Currency | CHF |

---

## Commands

| Input | Action |
|-------|--------|
| Any natural language | Send to agent |
| `reset` | Clear conversation history (keeps loaded data) |
| `exit` | Quit |

---

## Troubleshooting

**"Could not connect to Power BI MCP"**
→ Make sure Power BI Desktop is open with the FINANCE model loaded.
→ Check the port in `POWERBI_CONNECTION_STRING`.

**"No budget data loaded yet"**
→ Ask the agent to fetch data first: *"Load the 2026 budget data"*

**DAX query returns no rows**
→ Verify `company_id=4` and `value_type_id=2` exist in your model.