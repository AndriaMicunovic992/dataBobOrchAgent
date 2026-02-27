"""
telegram_bot.py  — thin Telegram interface, all intelligence is in the AI
==========================================================================
The bot does almost nothing itself. Every message goes straight to Opus.
The AI decides how to respond, what to ask, how to set up projects, etc.

Commands are just shortcuts that inject a hint — the AI handles everything.
"""

import logging
import os

from dotenv import load_dotenv
load_dotenv()

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode, ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

import orchestrator_engine as engine
from project_memory import Project, list_projects, get_active, set_active

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
ALLOWED_USER_IDS = set(
    int(x) for x in os.environ.get("ALLOWED_USER_IDS", "").split(",") if x.strip()
)

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)
MAX_MSG = 4000


def is_allowed(user_id: int) -> bool:
    return not ALLOWED_USER_IDS or user_id in ALLOWED_USER_IDS


async def send_long(update: Update, text: str):
    if not text.strip():
        return
    for chunk in [text[i:i + MAX_MSG] for i in range(0, len(text), MAX_MSG)]:
        try:
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(chunk)


async def route(update: Update, user_id: int, message: str):
    """Send any message to the AI and stream status updates back."""
    project = get_active(user_id)

    if project is None:
        # Check if this user has conversation history (session may have restarted)
        conv = engine.get_conversation(user_id)
        if conv:
            log.warning(
                "User %s has %d conversation messages but get_active() returned None — "
                "engine.chat() will attempt project recovery",
                user_id, len(conv),
            )

    await update.message.chat.send_action(ChatAction.TYPING)

    async def status_cb(text: str):
        try:
            await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            pass

    try:
        reply = await engine.chat(
            user_id=user_id,
            message=message,
            project=project,
            status_cb=status_cb,
        )
    except Exception as e:
        log.exception("Engine error")
        await update.message.reply_text(f"❌ {e}")
        return

    await send_long(update, reply)


# ── Commands — inject a hint to the AI, let it handle the rest ────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    await route(update, update.effective_user.id,
        "The user just started the bot. Introduce yourself briefly and ask what they want to build or work on.")

async def cmd_newproject(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    await route(update, update.effective_user.id,
        "The user wants to start a new project. Ask them what they want to build. "
        "When you have enough info, call create_new_project with the name and tech stack details. "
        "Then call git_init if they provide a GitHub repo URL.")

async def cmd_project(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    await route(update, update.effective_user.id,
        "Show the current project status and tech stack. Be concise.")

async def cmd_tasks(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    await route(update, update.effective_user.id,
        "Call list_tasks and show the results.")

async def cmd_files(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    await route(update, update.effective_user.id,
        "Call list_workspace and show the project file tree.")

async def cmd_decisions(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    project = get_active(update.effective_user.id)
    if project:
        decisions = project.get_decisions()
        await send_long(update, f"📋 *Architecture Decisions*\n\n{decisions or '_(none logged yet)_'}")
    else:
        await update.message.reply_text("No active project.")

async def cmd_commit(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    await route(update, update.effective_user.id,
        "The user wants to commit and push to GitHub. Call git_status then git_commit_push with a meaningful message.")

async def cmd_plan(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    await route(update, update.effective_user.id,
        "Show the current task group execution status. Call get_group_status for any active groups.")

async def cmd_review(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    await route(update, update.effective_user.id,
        "Run review_results on the most recently completed task group. Report findings.")

async def cmd_clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    engine.clear_conversation(update.effective_user.id)
    await update.message.reply_text("🧹 Conversation cleared.")

async def cmd_switchproject(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    projects = list_projects()
    if not projects:
        await update.message.reply_text("No projects yet. Just tell me what you want to build.")
        return
    keyboard = [
        [InlineKeyboardButton(p.get("name", p["slug"]), callback_data=f"switch:{p['slug']}")]
        for p in projects
    ]
    await update.message.reply_text("Switch to:", reply_markup=InlineKeyboardMarkup(keyboard))

async def callback_switch(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    slug = query.data.replace("switch:", "")
    set_active(query.from_user.id, slug)
    engine.clear_conversation(query.from_user.id)
    project = Project(slug)
    data = project.load()
    await query.edit_message_text(f"✅ Switched to *{data.get('name', slug)}*", parse_mode=ParseMode.MARKDOWN)

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        return
    await update.message.reply_text(
        "/newproject — start a new project\n"
        "/project — current project info\n"
        "/switchproject — switch projects\n"
        "/tasks — task history\n"
        "/files — workspace files\n"
        "/decisions — architecture log\n"
        "/plan — show execution plan status\n"
        "/review — review latest completed work\n"
        "/commit — push to GitHub\n"
        "/clear — reset conversation\n\n"
        "_Everything else — just talk to me._",
        parse_mode=ParseMode.MARKDOWN,
    )


# ── All normal messages go straight to the AI ─────────────────────────────────

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        await update.message.reply_text("⛔ Not authorised.")
        return
    await route(update, update.effective_user.id, update.message.text.strip())


# ── Start ──────────────────────────────────────────────────────────────────────

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",         cmd_start))
    app.add_handler(CommandHandler("help",          cmd_help))
    app.add_handler(CommandHandler("newproject",    cmd_newproject))
    app.add_handler(CommandHandler("project",       cmd_project))
    app.add_handler(CommandHandler("tasks",         cmd_tasks))
    app.add_handler(CommandHandler("files",         cmd_files))
    app.add_handler(CommandHandler("decisions",     cmd_decisions))
    app.add_handler(CommandHandler("commit",        cmd_commit))
    app.add_handler(CommandHandler("plan",          cmd_plan))
    app.add_handler(CommandHandler("review",        cmd_review))
    app.add_handler(CommandHandler("clear",         cmd_clear))
    app.add_handler(CommandHandler("switchproject", cmd_switchproject))
    app.add_handler(CallbackQueryHandler(callback_switch, pattern="^switch:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("🤖 Bot starting…")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()