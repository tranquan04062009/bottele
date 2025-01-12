import os
import time
import secrets
import requests
import asyncio
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

# Load environment vaảiableS
load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')

if not BOT_TOKEN:
    raise ValueError("No BOT_TOKEN found in environment variables. Please add it to .env file")

# Store user spam sessions and blocked users
user_spam_sessions = {}  # Store spam lists by user
blocked_users = []  # Store blocked users
progress_bars = {}  # Store progress bars for each session

class ProgressTracker:
    def __init__(self, chat_id, session_id, message_id=None):
        self.chat_id = chat_id
        self.session_id = session_id
        self.message_id = message_id
        self.counter = 0
        self.start_time = datetime.now()
        self.last_update = datetime.now()

    def format_progress(self):
        current_time = datetime.now()
        elapsed = (current_time - self.start_time).total_seconds()
        rate = self.counter / elapsed if elapsed > 0 else 0
        
        progress = (
            f"Session {self.session_id} Progress:\n"
            f"Messages Sent: {self.counter}\n"
            f"Rate: {rate:.2f} msgs/sec\n"
            f"Running Time: {int(elapsed)}s"
        )
        return progress

async def update_progress(context, progress_tracker):
    """Update progress message periodically"""
    try:
        current_time = datetime.now()
        # Update every 5 seconds to avoid API rate limits
        if (current_time - progress_tracker.last_update).total_seconds() >= 5:
            progress_text = progress_tracker.format_progress()
            
            if progress_tracker.message_id:
                await context.bot.edit_message_text(
                    chat_id=progress_tracker.chat_id,
                    message_id=progress_tracker.message_id,
                    text=progress_text
                )
            else:
                message = await context.bot.send_message(
                    chat_id=progress_tracker.chat_id,
                    text=progress_text
                )
                progress_tracker.message_id = message.message_id
            
            progress_tracker.last_update = current_time
    except Exception as e:
        print(f"Error updating progress: {str(e)}")

async def send_message(username: str, message: str, chat_id: int, session_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Function to send spam messages"""
    progress_tracker = ProgressTracker(chat_id, session_id)
    progress_bars[f"{chat_id}_{session_id}"] = progress_tracker
    
    while user_spam_sessions.get(chat_id, {}).get(session_id - 1, {}).get('is_active', False):
        try:
            device_id = secrets.token_hex(21)
            url = "https://ngl.link/api/submit"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/109.0",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
            }
            data = f"username={username}&question={message}&deviceId={device_id}&gameSlug=&referrer="

            response = requests.post(url, headers=headers, data=data)

            if response.status_code != 200:
                print("[Error] Rate limited, waiting 5 seconds...")
                await asyncio.sleep(5)
            else:
                progress_tracker.counter += 1
                await update_progress(context, progress_tracker)

            await asyncio.sleep(2)
        except Exception as e:
            print(f"[Error] {str(e)}")
            await asyncio.sleep(2)

    # Final progress update when session ends
    await update_progress(context, progress_tracker)
    await context.bot.send_message(
        chat_id=chat_id,
        text=f"Session {session_id} has ended.\nTotal messages sent: {progress_tracker.counter}"
    )
    
    # Cleanup
    if f"{chat_id}_{session_id}" in progress_bars:
        del progress_bars[f"{chat_id}_{session_id}"]

def is_blocked(chat_id: int) -> bool:
    """Check if user is blocked"""
    return chat_id in blocked_users

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    chat_id = update.message.chat_id
    username = update.message.from_user.username or "No username"
    first_name = update.message.from_user.first_name or "No name"
    user_id = update.message.from_user.id

    if is_blocked(chat_id):
        await update.message.reply_text("You are blocked from using this bot.")
        return

    await update.message.reply_text(f"Welcome! Your Telegram ID is: {user_id}")

    if chat_id not in user_spam_sessions:
        user_spam_sessions[chat_id] = []

    keyboard = [
        ["Start Spam", "Spam List"]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text(
        "Choose the feature you want to use:",
        reply_markup=reply_markup
    )

async def handle_start_spam(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle Start Spam button"""
    chat_id = update.message.chat_id

    if is_blocked(chat_id):
        await update.message.reply_text("You are blocked from using this bot.")
        return

    await update.message.reply_text("Enter the username you want to spam:")
    context.user_data['waiting_for'] = 'username'

async def handle_spam_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle Spam List button"""
    chat_id = update.message.chat_id

    if is_blocked(chat_id):
        await update.message.reply_text("You are blocked from using this bot.")
        return

    sessions = user_spam_sessions.get(chat_id, [])
    if sessions:
        list_message = "Current spam sessions list:\n"
        buttons = []
        for session in sessions:
            progress_key = f"{chat_id}_{session['id']}"
            counter = progress_bars.get(progress_key, ProgressTracker(chat_id, session['id'])).counter
            list_message += (
                f"{session['id']}: {session['username']} - {session['message']}\n"
                f"Messages sent: {counter} [Active: {session['is_active']}]\n"
            )
            buttons.append([InlineKeyboardButton(
                f"Stop session {session['id']}", 
                callback_data=f"stop_{session['id']}"
            )])

        reply_markup = InlineKeyboardMarkup(buttons)
        await update.message.reply_text(list_message, reply_markup=reply_markup)
    else:
        await update.message.reply_text("No active spam sessions.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular messages"""
    if 'waiting_for' not in context.user_data:
        return

    chat_id = update.message.chat_id
    message_text = update.message.text

    if context.user_data['waiting_for'] == 'username':
        context.user_data['spam_username'] = message_text
        context.user_data['waiting_for'] = 'message'
        await update.message.reply_text("Enter the message you want to send:")
    
    elif context.user_data['waiting_for'] == 'message':
        username = context.user_data['spam_username']
        if chat_id not in user_spam_sessions:
            user_spam_sessions[chat_id] = []
        
        current_session_id = len(user_spam_sessions[chat_id]) + 1
        user_spam_sessions[chat_id].append({
            'id': current_session_id,
            'username': username,
            'message': message_text,
            'is_active': True
        })
        
        # Start spam session with progress tracking
        await update.message.reply_text(f"Spam session {current_session_id} is starting...")
        asyncio.create_task(send_message(username, message_text, chat_id, current_session_id, context))
        context.user_data.clear()

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle callback queries"""
    query = update.callback_query
    chat_id = query.message.chat_id
    session_id = int(query.data.split('_')[1])

    sessions = user_spam_sessions.get(chat_id, [])
    session = next((s for s in sessions if s['id'] == session_id), None)

    if session:
        session['is_active'] = False
        await query.message.reply_text(f"Stopping spam session {session_id}...")
    else:
        await query.message.reply_text(f"Couldn't find spam session with ID {session_id}.")

def main():
    """Main function to run the bot"""
    application = Application.builder().token(BOT_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.regex('^Start Spam$'), handle_start_spam))
    application.add_handler(MessageHandler(filters.regex('^Spam List$'), handle_spam_list))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(handle_callback))

    # Start the bot
    print("Bot is running...")
    application.run_polling()

if __name__ == '__main__':
    main()