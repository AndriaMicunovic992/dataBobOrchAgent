# AI Orchestrator v3 — Project-Aware Telegram Bot

Full-stack AI coding team accessible from Telegram on any device.

## Architecture

```
Telegram (mobile / desktop)
         │
         ▼
  telegram_bot.py          ← commands, wizard, message routing
         │
         ▼
  orchestrator_engine.py   ← Opus 4.6, tools, subagent dispatch
    ├── Coding agents       ← Sonnet 4.6, file read/write
    ├── Data agent          ← Sonnet 4.6, specialised DAX/SQL
    └── Claude Code CLI     ← fully autonomous shell agent
         │
  project_memory.py        ← persistent project state
    ├── projects/<slug>/project.json     (stack, conventions)
    ├── projects/<slug>/decisions.md     (architecture log)
    ├── projects/<slug>/tasks.db         (task history)
    └── projects/<slug>/workspace/       (all generated code)
         │
  git_tools.py             ← git init, commit, push to GitHub
```

---

## What the agent remembers across sessions

| Memory type | Where stored | What's in it |
|-------------|-------------|--------------|
| Project identity | `project.json` | Name, description, tech stack, GitHub URL |
| Decisions | `decisions.md` | Every architectural choice + reasoning |
| Task history | `tasks.db` | All tasks with status and results |
| Conventions | `project.json` | Coding style, patterns to follow |
| Workspace | `workspace/` | All generated files (also in GitHub) |

---

## Setup

### 1. Get your Telegram credentials
- **Bot token**: message `@BotFather` → `/newbot`
- **Your user ID**: message `@userinfobot` → `/start`

### 2. Get a GitHub Personal Access Token
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token → check **repo** scope
3. Copy the `ghp_...` token

### 3. Install and run locally

```bash
# Windows
pip install -r requirements.txt

# Create .env file (copy .env.example and fill in)
copy .env.example .env
# edit .env with your values

# Run
python telegram_bot.py
```

Add these two lines to the **top** of `telegram_bot.py`:
```python
from dotenv import load_dotenv
load_dotenv()
```

### 4. Deploy to Railway (always-on)

1. Push to GitHub
2. railway.app → New Project → Deploy from GitHub
3. Add environment variables:
   - `ANTHROPIC_API_KEY`
   - `TELEGRAM_TOKEN`
   - `ALLOWED_USER_IDS`
   - `GITHUB_TOKEN`
4. Start command: `python telegram_bot.py`

---

## Bot commands

| Command | What it does |
|---------|-------------|
| `/newproject` | 7-step wizard to set up a new project |
| `/project` | Show active project info & tech stack |
| `/switchproject` | Switch between projects |
| `/tasks` | Task history with statuses |
| `/files` | Workspace file tree |
| `/decisions` | Architecture decisions log |
| `/commit` | Manually trigger git commit + push |
| `/clear` | Reset conversation memory |
| `/help` | Show all commands |
| _(any message)_ | Sent to orchestrator |

---

## Example session

```
You: /newproject

Bot: 🚀 New Project Setup
     Step 1/7 — What's the project name?

You: Sales Dashboard

Bot: Step 2/7 — Describe the project…

You: A web dashboard showing sales KPIs from our Power BI 
     semantic model, with user auth and drill-down filters

Bot: Step 3/7 — Backend framework?
     [FastAPI] [Django] [Node/Express] [Type it]

You: [FastAPI]

Bot: Step 4/7 — Frontend?

You: React + Vite

     ... (stack, DB, GitHub repo)

Bot: ✅ Project created!
     📛 Sales Dashboard
     🖥️ FastAPI (Python)
     🌐 React + Vite
     📊 Power BI Semantic Models (DAX)
     🐙 https://github.com/you/sales-dashboard


You: Build the backend auth system with JWT tokens

Bot: ⚙️ spawn_parallel_agents…
     ⚡ Launching 3 agents in parallel…
     🔧 [auth-models] started…
     🔧 [auth-routes] started…
     🔧 [auth-tests] started…
     ✅ [auth-models] finished.
     ✅ [auth-routes] finished.
     ✅ [auth-tests] finished.
     ⚙️ git_commit_push…

     ✅ Done! Auth system built and pushed.
     Files: backend/models/user.py, backend/routes/auth.py, tests/test_auth.py
     Commit: "Add JWT authentication system"


You: Now write a DAX measure for monthly revenue growth vs last year

Bot: 📊 [data-agent] data agent started…
     📂 [data-agent] write_file
     ✅ [data-agent] data agent done.
     ⚙️ git_commit_push…

     ✅ DAX measure written to queries/revenue_growth_yoy.dax
     Committed: "Add YoY revenue growth DAX measure"


You: /decisions

Bot: ## [2025-01-15 14:30] Use JWT for authentication
     Chose JWT over session cookies for stateless auth,
     better suited for API-first architecture with React frontend.

     ## [2025-01-15 14:45] DAX measure for YoY comparison
     Used SAMEPERIODLASTYEAR for time intelligence instead of
     manual date offset, more readable and handles year boundaries.
```

---

## File structure

```
agent-telegram-v3/
├── telegram_bot.py          ← Telegram interface
├── orchestrator_engine.py   ← AI orchestrator + subagents
├── project_memory.py        ← Persistent project state
├── git_tools.py             ← Git operations
├── requirements.txt
├── .env.example
└── projects/                ← Created at runtime
    └── sales-dashboard/
        ├── project.json
        ├── decisions.md
        ├── tasks.db
        └── workspace/       ← Git repo lives here
            ├── .git/
            ├── backend/
            ├── frontend/
            └── queries/
```
