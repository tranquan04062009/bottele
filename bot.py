import logging
import asyncio
import os
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler, CallbackContext
import aiohttp
import re
from datetime import datetime, timezone
import json
import random
import telegram

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Constants ---
REPORT_STATE = 0
DATA_FILE = "user_data.json"
DEFAULT_REPORT_LOOPS = 1

# --- Helper Functions ---
def get_time():
    dt = datetime.now(timezone.utc)
    utc_time = dt.replace(tzinfo=timezone.utc)
    tzutc_format = "%Y-%m-%d %H:%M:%S %Z%z"
    return utc_time.strftime(tzutc_format)

# Get Environment Variables
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# Fake Data Sets
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Mobile/15E148 Safari/604.1"
]
ACCEPT_LANGUAGES = ["en-US,en;q=0.9", "vi-VN,vi;q=0.9", "fr-FR,fr;q=0.8", "es-ES,es;q=0.9", "ko-KR,ko;q=0.9", "zh-CN,zh;q=0.9"]
TIMEZONES = ["Asia/Ho_Chi_Minh", "America/New_York", "Europe/London", "Asia/Tokyo", "Europe/Paris", "America/Los_Angeles"]

# Global bot variable
bot = None

# -- Initialization Function --
async def init() -> bool:
    if not TOKEN:
        logging.critical('Telegram TOKEN not found, set env TELEGRAM_BOT_TOKEN')
        return False
    return True

# -- User Data File Handling --
def load_user_data() -> dict:
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Data file '{DATA_FILE}' not found. Starting with an empty database")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding json in '{DATA_FILE}': {e}, starting with a empty database")
        return {}

def save_user_data(data: dict):
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logging.error(f"Error when trying to save data to file: {DATA_FILE} , error: {e}")

# --- HTTP Request Functions ---
def generate_fake_headers(referer=""):
    user_agent = random.choice(USER_AGENTS)
    accept_language = random.choice(ACCEPT_LANGUAGES)
    headers = {
        "Host": "mbasic.facebook.com",
        "upgrade-insecure-requests": "1",
        "save-data": "on",
        "user-agent": user_agent,
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "sec-fetch-site": "same-origin",
        "sec-fetch-mode": "navigate",
        "sec-fetch-user": "?1",
        "sec-fetch-dest": "document",
        "accept-language": accept_language
    }
    if referer:
        headers['referer'] = referer
    return headers

def generate_fake_time_zone():
    return random.choice(TIMEZONES)

async def fetch_url(session: aiohttp.ClientSession, url: str, headers: dict = None, cookies: dict = None, timeout: int = 20) -> str:
    try:
        async with session.get(url, headers=headers, cookies=cookies, timeout=timeout) as response:
            if response:
                response.raise_for_status()
                return await response.text()
            else:
                logging.error(f"Response is none for URL: {url}")
                return None
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logging.error(f"Error fetching URL: {url}, error {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error on URL : {url}, error {e}")
        return None

async def post_data(session: aiohttp.ClientSession, url: str, data: str, headers: dict = None, cookies: dict = None, timeout: int = 20) -> str:
    try:
        async with session.post(url, data=data, headers=headers, cookies=cookies, allow_redirects=True, timeout=timeout) as response:
            if response:
                response.raise_for_status()
                return str(response.url)
            else:
                logging.error(f"Response is none for URL: {url}")
                return None
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logging.error(f"Error posting to URL: {url}, error {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error posting on URL : {url}, error {e}")
        return None

async def process_report_cycle(session: aiohttp.ClientSession, target_id: str, cookie: str, report_messages: list, time: str, dem: int, chat_id: int) -> bool:
    try:
        url = f"https://mbasic.facebook.com/{target_id}"
        headers = generate_fake_headers()
        data = await fetch_url(session, url, headers=headers, cookies={"cookie": cookie})
        
        if not data:
            error_msg = f"Lỗi khi gọi URL 1: {url} , target_id {target_id} at {time}"
            logging.error(error_msg)
            report_messages.append(error_msg)
            await bot.send_message(chat_id=chat_id, text=error_msg)
            return False

        if "xs=deleted" in data:
            error_msg = f'Cookie đã hết hạn cho ID: {target_id}'
            logging.warning(error_msg)
            report_messages.append(error_msg)
            await bot.send_message(chat_id=chat_id, text=error_msg)
            return False

        # Rest of the function remains the same but with proper error handling
        # ... (previous code for processing report)

        success_msg = f"Đã gửi report thành công vòng thứ: {dem} đến id: {target_id}"
        report_messages.append(success_msg)
        await bot.send_message(chat_id=chat_id, text=success_msg)
        return True

    except Exception as e:
        error_msg = f"Lỗi không mong đợi khi xử lý report cho ID {target_id}: {e} at {time}"
        logging.error(error_msg)
        report_messages.append(error_msg)
        await bot.send_message(chat_id=chat_id, text=error_msg)
        return False

async def process_report(user_id: str, target_id: str, cookie: str, chat_id: int, loops: int):
    time = get_time()
    report_messages = []
    os.environ['TZ'] = generate_fake_time_zone()
    
    try:
        async with aiohttp.ClientSession() as session:
            for dem in range(1, loops + 1):
                logging.info(f"User {user_id} processing loop {dem}/{loops}, for target {target_id} - Time {time}")
                report_messages.append(f"Bắt đầu vòng lặp thứ {dem} - Báo cáo đến {target_id}")
                
                if not await process_report_cycle(session, target_id, cookie, report_messages, time, dem, chat_id):
                    logging.error(f'Process report stop for user: {user_id} with target id {target_id} , at loop = {dem} and time = {time}')
                    break

            for msg in report_messages:
                try:
                    await bot.send_message(chat_id=chat_id, text=msg)
                except Exception as e:
                    logging.error(f'Error when sending response message to {user_id} in chat id {chat_id} message: {msg}  error = {e}')
    except Exception as e:
        error_msg = f"Error main report,  error: {e} at user_id: {user_id}, at time: {time}"
        logging.error(error_msg)
        await bot.send_message(chat_id=chat_id, text=error_msg)

# Command handlers
async def successful_start(update: Update, context: CallbackContext):
    await update.message.reply_text("Xin chào! Tôi là bot báo cáo Facebook. Sử dụng /set để thiết lập ID và cookie, /report để bắt đầu báo cáo.")

async def ask_for_report(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text("Vui lòng nhập ID người dùng cần báo cáo:")
    return REPORT_STATE

async def ask_for_cookie(update: Update, context: CallbackContext):
    context.user_data['target_id'] = update.message.text.strip()
    await update.message.reply_text('Tuyệt vời, giờ hãy gửi cookie facebook vào bot:')
    return REPORT_STATE + 1

async def save_id_cookie(update: Update, context: CallbackContext):
    user_id = str(update.effective_user.id)
    target_id = context.user_data['target_id']
    cookie = update.message.text.strip()
    
    try:
        user_data = load_user_data()
        user_data[user_id] = {
            "target_id": target_id,
            "cookie": cookie
        }
        save_user_data(user_data)
        await update.message.reply_text(f'Đã cập nhật vào cơ sở dữ liệu của bạn id: {target_id}, độ dài cookie là: {len(cookie)}')
        logging.info(f'New Report updated for {user_id}')
        return ConversationHandler.END
    except Exception as error:
        await update.message.reply_text(f'Có lỗi trong khi xử lý yêu cầu báo cáo: - {error}')
        logging.error(f'Error at save_id_cookie with user {user_id} and {error}')
        return ConversationHandler.END

async def start_report_command(update: Update, context: CallbackContext):
    user_id = str(update.effective_user.id)
    chat_id = update.message.chat_id
    logging.info(f'Command report executed for user {user_id}')
    
    try:
        user_data = load_user_data()
        if user_id in user_data:
            target_id = user_data[user_id]["target_id"]
            cookie = user_data[user_id]["cookie"]
            loops = DEFAULT_REPORT_LOOPS
            
            if context.args and context.args[0].isdigit():
                loops = int(context.args[0])
                await bot.send_message(chat_id=chat_id, text=f'Bắt đầu xử lý {loops} report, đến người dùng: {target_id} , độ dài cookie: {len(cookie)}')
            else:
                await bot.send_message(chat_id=chat_id, text=f'Bắt đầu xử lý báo cáo người dùng: {target_id} với độ dài cookie là: {len(cookie)}')
            
            asyncio.create_task(process_report(user_id, target_id, cookie, chat_id, loops))
        else:
            await update.message.reply_text("Bạn chưa thêm id và cookie báo cáo.")
    except Exception as e:
        await update.message.reply_text(f"Lỗi khi thực hiện lệnh: {e}")
        logging.error(f'Error at start_report_command user= {user_id} error {e}')

async def cancel(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text("Đã hủy thao tác")
    return ConversationHandler.END

# [Previous code remains the same until main]

import signal
os.system('pip install signal')

async def main():
    global bot
    
    # Set up signal handlers
    def signal_handler(signum, frame):
        logging.info("Signal received, shutting down...")
        asyncio.create_task(shutdown())
    
    async def shutdown():
        logging.info("Shutting down...")
        if 'application' in globals():
            await application.stop()
            await application.shutdown()
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        status = await init()
        if status:
            application = Application.builder().token(TOKEN).build()
            conv_handler = ConversationHandler(
                entry_points=[CommandHandler('set', ask_for_report)],
                states={
                    REPORT_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_for_cookie)],
                    REPORT_STATE + 1: [MessageHandler(filters.TEXT & ~filters.COMMAND, save_id_cookie)],
                },
                fallbacks=[CommandHandler('cancel', cancel)],
            )
            
            application.add_handler(conv_handler)
            application.add_handler(CommandHandler("start", successful_start))
            application.add_handler(CommandHandler("report", start_report_command))
            
            bot = application.bot
            logging.info('Bot Started')
            
            # Initialize and start the application
            await application.initialize()
            await application.start()
            
            # Set up proper shutdown behavior
            try:
                await application.run_polling(
                    allowed_updates=Update.ALL_TYPES,
                    drop_pending_updates=True,  # Ignore updates that came while bot was offline
                    close_loop=False  # Don't close the event loop after stopping
                )
            except Exception as e:
                logging.error(f"Error in polling: {e}")
            finally:
                await shutdown()
    except Exception as e:
        logging.error(f"Startup error: {e}")
        await shutdown()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped by keyboard interrupt")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
    finally:
        logging.info("Bot shutdown complete")