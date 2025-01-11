import logging
import asyncio
import os
from typing import List, Dict, Optional
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler, CallbackContext
import aiohttp
import re
from datetime import datetime, timezone
import json
import random
from http import HTTPStatus


# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Constants ---
REPORT_STATE = 0
DATA_FILE = "user_data.json"
DEFAULT_REPORT_LOOPS = 1
USER_AGENT_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Mobile/15E148 Safari/604.1"
]
ACCEPT_LANGUAGE_LIST = ["en-US,en;q=0.9", "vi-VN,vi;q=0.9", "fr-FR,fr;q=0.8","es-ES,es;q=0.9", "ko-KR,ko;q=0.9","zh-CN,zh;q=0.9"]
TIMEZONE_LIST = ["Asia/Ho_Chi_Minh", "America/New_York", "Europe/London", "Asia/Tokyo", "Europe/Paris","America/Los_Angeles"]

# --- Helper Functions ---
def get_time() -> str:
    """Gets the current UTC time in a formatted string."""
    dt = datetime.now(timezone.utc)
    tzutc_format = "%Y-%m-%d %H:%M:%S %Z%z"
    return dt.strftime(tzutc_format)


# Get Environment Variables
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# --- Data File Handling ---
def load_user_data() -> Dict[str, Dict[str, str]]:
    """Loads user data from the JSON file."""
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Data file '{DATA_FILE}' not found. Starting with an empty database")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON in '{DATA_FILE}': {e}, starting with an empty database")
        return {}


def save_user_data(data: Dict[str, Dict[str, str]]) -> None:
    """Saves user data to the JSON file."""
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.error(f"Error saving data to file '{DATA_FILE}': {e}")

# --- HTTP Request Functions ---
def generate_fake_headers(referer: str = "") -> Dict[str, str]:
    """Generates fake HTTP headers for requests."""
    user_agent = random.choice(USER_AGENT_LIST)
    accept_language = random.choice(ACCEPT_LANGUAGE_LIST)
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


def generate_fake_time_zone() -> str:
    """Generates a random timezone."""
    return random.choice(TIMEZONE_LIST)


async def fetch_url(
    session: aiohttp.ClientSession,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    cookies: Optional[Dict[str, str]] = None,
    timeout: int = 20,
) -> Optional[str]:
    """Fetches the content of a URL."""
    try:
        async with session.get(url, headers=headers, cookies=cookies, timeout=timeout) as response:
             if response and response.status == HTTPStatus.OK:
                 return await response.text()
             else:
                 log_message = f"Failed to fetch URL: {url}, status code: {response.status if response else 'N/A'}"
                 logging.error(log_message)
                 return None

    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logging.error(f"Error fetching URL: {url}, error {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error fetching URL: {url}, error {e}")
        return None

async def post_data(
    session: aiohttp.ClientSession,
    url: str,
    data: str,
    headers: Optional[Dict[str, str]] = None,
    cookies: Optional[Dict[str, str]] = None,
    timeout: int = 20,
) -> Optional[str]:
    """Posts data to a URL and returns the final URL after redirects."""
    try:
        async with session.post(url, data=data, headers=headers, cookies=cookies, allow_redirects=True, timeout=timeout) as response:
            if response and response.status == HTTPStatus.OK:
                 return str(response.url)
            else:
                 log_message = f"Failed to post data to URL: {url}, status code: {response.status if response else 'N/A'}"
                 logging.error(log_message)
                 return None

    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logging.error(f"Error posting to URL: {url}, error {e}")
        return None
    except Exception as e:
         logging.error(f"Unexpected error posting to URL: {url}, error {e}")
         return None

async def process_report_cycle(
    session: aiohttp.ClientSession,
    target_id: str,
    cookie: str,
    report_messages: List[str],
    time: str,
    loop_number: int,
) -> bool:
    """Processes a single reporting cycle."""
    try:
        url = f"https://mbasic.facebook.com/{target_id}"
        headers = generate_fake_headers()
        data = await fetch_url(session, url, headers=headers, cookies={"cookie": cookie})
        if not data:
            error_msg = f"Failed to fetch initial URL: {url} for target_id {target_id} at {time}"
            logging.error(error_msg)
            report_messages.append(error_msg)
            return False

        if "xs=deleted" in data:
            error_msg = f'Cookie expired for ID: {target_id}'
            logging.warning(error_msg)
            report_messages.append(error_msg)
            return False

        redirect_match = re.search(r'location: (.*?)\r\n', data)
        if redirect_match:
            redirect_url = redirect_match.group(1).strip()
            headers = generate_fake_headers(referer=url)
            redirect_data = await fetch_url(session, redirect_url, headers=headers)
            if not redirect_data:
               error_msg = f'Failed to fetch redirect URL: {redirect_url} for target_id: {target_id} at {time}'
               logging.error(error_msg)
               report_messages.append(error_msg)
               return False

            action_match = re.search(r'/nfx/basic/direct_actions/(.*?)"', redirect_data)
            if action_match:
                action_data = action_match.group(1).strip()
                l1, l2, l3 = action_data.split("amp;")
                link1 = f"https://mbasic.facebook.com/nfx/basic/direct_actions/{l1}{l2}{l3}"
                headers = generate_fake_headers(referer=redirect_url)
                link1_data = await fetch_url(session, link1, headers=headers, cookies={"cookie": cookie})
                if not link1_data:
                    error_msg = f'Failed at link1: {link1} for id: {target_id} at time: {time}'
                    logging.error(error_msg)
                    report_messages.append(error_msg)
                    return False
                handle_action_match = re.search(r'/nfx/basic/handle_action/(.*?)"', link1_data)
                if handle_action_match:
                    handle_action_data = handle_action_match.group(1).strip()
                    z1, z2, z3, z4, z5, z6 = handle_action_data.split('amp;')
                    fb_dtsg_match = re.search(r'name="fb_dtsg" value="(.*?)"', link1_data)
                    jazoest_match = re.search(r'name="jazoest" value="(.*?)"', link1_data)

                    if fb_dtsg_match and jazoest_match:
                        fb_dtsg = fb_dtsg_match.group(1).strip()
                        jazoest = jazoest_match.group(1).strip()
                        link2 = f"https://mbasic.facebook.com/nfx/basic/handle_action/{z1}{z2}{z3}{z4}{z5}{z6}"
                        data = f"fb_dtsg={fb_dtsg}&jazoest={jazoest}&action_key=RESOLVE_PROBLEM&submit=Gửi"
                        headers = generate_fake_headers(referer=link1)
                        headers['content-length'] = str(len(data))
                        headers['content-type'] = "application/x-www-form-urlencoded"
                        link3 = await post_data(session, link2, data, headers=headers, cookies={"cookie": cookie})
                        if not link3:
                            error_msg = f'Failed to post data at route {link2} for target_id = {target_id} at time: {time}'
                            logging.error(error_msg)
                            report_messages.append(error_msg)
                            return False
                        headers = generate_fake_headers(referer=link2)
                        link3_data = await fetch_url(session, link3, headers=headers, cookies={"cookie": cookie})
                        if not link3_data:
                            error_msg = f'Failed at final route of first report for target_id: {target_id} at time: {time}'
                            logging.error(error_msg)
                            report_messages.append(error_msg)
                            return False

                        tag_match = re.search(r'/ixt/screen/frxtagselectionscreencustom/post/msite/(.*?)"', link3_data)
                        if tag_match:
                           tag_data = tag_match.group(1).strip()
                           x1, x2 = tag_data.split('amp;')
                           fb_dtsg_match = re.search(r'name="fb_dtsg" value="(.*?)"', link3_data)
                           jazoest_match = re.search(r'name="jazoest" value="(.*?)"', link3_data)

                           if fb_dtsg_match and jazoest_match:
                                fb_dtsg = fb_dtsg_match.group(1).strip()
                                jazoest = jazoest_match.group(1).strip()
                                link4 = f"https://mbasic.facebook.com/ixt/screen/frxtagselectionscreencustom/post/msite/{x1}{x2}"
                                data = f"fb_dtsg={fb_dtsg}&jazoest={jazoest}&tag=spam&action=Gửi"
                                headers = generate_fake_headers(referer=link3)
                                headers['content-length'] = str(len(data))
                                headers['content-type'] = "application/x-www-form-urlencoded"
                                link5 = await post_data(session, link4, data, headers=headers, cookies={"cookie": cookie})
                                if not link5:
                                   error_msg = f'Failed at tag spam step 2 for ID:{target_id} at time: {time}'
                                   logging.error(error_msg)
                                   report_messages.append(error_msg)
                                   return False

                                headers = generate_fake_headers(referer=link4)
                                link5_data = await fetch_url(session, link5, headers=headers, cookies={"cookie": cookie})
                                if not link5_data:
                                     error_msg = f'Failed at route type 2, id: {target_id}, time = {time}'
                                     logging.error(error_msg)
                                     report_messages.append(error_msg)
                                     return False

                                rapid_report_match = re.search(r'/rapid_report/basic/actions/post/(.*?)"', link5_data)
                                if rapid_report_match:
                                    rapid_report_data = rapid_report_match.group(1).strip()
                                    x1, x2, x3, x4 = rapid_report_data.split('amp;')
                                    fb_dtsg_match = re.search(r'name="fb_dtsg" value="(.*?)"', link5_data)
                                    jazoest_match = re.search(r'name="jazoest" value="(.*?)"', link5_data)

                                    if fb_dtsg_match and jazoest_match:
                                        fb_dtsg = fb_dtsg_match.group(1).strip()
                                        jazoest = jazoest_match.group(1).strip()
                                        link6 = f"https://mbasic.facebook.com/rapid_report/basic/actions/post/{x1}{x2}{x3}{x4}"
                                        data = f"fb_dtsg={fb_dtsg}&jazoest={jazoest}&action=Gửi"
                                        headers = generate_fake_headers(referer=link5)
                                        headers['content-length'] = str(len(data))
                                        headers['content-type'] = "application/x-www-form-urlencoded"
                                        result = await post_data(session, link6, data, headers=headers, cookies={"cookie": cookie})
                                        if not result:
                                          error_msg = f'Failed at final report post, target_id = {target_id} at {time}'
                                          logging.error(error_msg)
                                          report_messages.append(error_msg)
                                          return False
                                        success_msg = f"Report sent successfully for loop: {loop_number}, id: {target_id}"
                                        report_messages.append(success_msg)
                                        return True
    except Exception as e:
        error_msg = f"Unexpected error processing report for target id {target_id}: {e} at time: {time}"
        logging.exception(error_msg)  # Use logging.exception to include stack trace
        report_messages.append(error_msg)
        return False
    return False


async def process_report(user_id: str, target_id: str, cookie: str, chat_id: int, loops: int):
    """Manages the report processing flow."""
    time = get_time()
    report_messages = []
    os.environ['TZ'] = generate_fake_time_zone()
    try:
        async with aiohttp.ClientSession() as session:
            for loop_number in range(1, loops + 1):
                logging.info(f"User {user_id} processing loop {loop_number}/{loops} for target {target_id} at {time}")
                report_messages.append(f"Starting loop {loop_number} - Reporting {target_id}")
                if not await process_report_cycle(session, target_id, cookie, report_messages, time, loop_number):
                    logging.error(f'Report process stopped for user: {user_id} with target id {target_id} at loop = {loop_number} at time = {time}')
                    break  # Stop on the first failure
            for msg in report_messages:
                try:
                   await bot.send_message(chat_id=chat_id, text=msg)
                except Exception as e:
                    logging.error(f'Error sending message to user {user_id} in chat id {chat_id}: {msg}, error = {e}')

    except Exception as e:
        error_msg = f"Error in main report process for user_id: {user_id}: {e} at time: {time}"
        logging.exception(error_msg)
        await bot.send_message(chat_id=chat_id, text=error_msg)


# Command Handlers
async def ask_for_report(update: Update, context: CallbackContext) -> int:
    """Asks for the target ID."""
    await update.message.reply_text("Please enter the ID of the user to report:")
    return REPORT_STATE


async def ask_for_cookie(update: Update, context: CallbackContext) -> Optional[int]:
    """Asks for the cookie after receiving the target ID."""
    if 'target_id' in context.user_data:
        context.user_data['target_id'] = update.message.text.strip()
        await update.message.reply_text('Great, now please send the Facebook cookie:')
        return REPORT_STATE + 1
    else:
        await update.message.reply_text('Error: You did not enter a user ID. Please use /set first.')
        return ConversationHandler.END


async def save_id_cookie(update: Update, context: CallbackContext) -> int:
    """Saves the target ID and cookie to user data."""
    user_id = str(update.effective_user.id)
    target_id = context.user_data['target_id']
    cookie = update.message.text.strip()
    user_data = load_user_data()
    try:
        user_data[user_id] = {"target_id": target_id, "cookie": cookie}
        save_user_data(user_data)
        await update.message.reply_text(f'Updated user data with target ID: {target_id}, cookie length: {len(cookie)}')
        logging.info(f'New report updated for user {user_id}')
        return ConversationHandler.END
    except Exception as error:
        await update.message.reply_text(f'Error processing report request: {error}')
        logging.error(f'Error in save_id_cookie for user {user_id}: {error}')
        return ConversationHandler.END


async def start_report_command(update: Update, context: CallbackContext) -> None:
    """Starts the report process."""
    global bot
    user_id = str(update.effective_user.id)
    chat_id = update.message.chat_id
    logging.info(f'Command /report executed for user {user_id}')
    user_data = load_user_data()
    try:
        if user_id in user_data:
            target_id, cookie = user_data[user_id]["target_id"], user_data[user_id]["cookie"]
            loops = DEFAULT_REPORT_LOOPS
            if context.args and context.args[0].isdigit():
                loops = int(context.args[0])
                await bot.send_message(chat_id=chat_id, text=f'Starting {loops} reports for user: {target_id}, cookie length: {len(cookie)}')
            else:
                await bot.send_message(chat_id=chat_id, text=f'Starting report for user: {target_id}, with cookie length: {len(cookie)}')
            asyncio.create_task(process_report(user_id, target_id, cookie, chat_id, loops))
        else:
            await update.message.reply_text("You have not added an ID and cookie for reporting.")
    except Exception as e:
        await update.message.reply_text(f"Error executing the command: {e}")
        logging.error(f'Error in start_report_command for user {user_id}: {e}')


async def successful_start(update: Update, context: CallbackContext) -> None:
    """Simple handler to check the bot is running"""
    await update.message.reply_text("Bot is running")

async def cancel(update: Update, context: CallbackContext) -> int:
    """Cancels and ends the conversation."""
    await update.message.reply_text("Operation cancelled")
    return ConversationHandler.END

async def init_bot() -> None:
    """Initialize bot and run it"""
    if not TOKEN:
        logging.critical('Telegram TOKEN not found, set env TELEGRAM_BOT_TOKEN')
        return
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
    global bot
    bot = application.bot #declare global var
    logging.info('Bot Started')
    await application.run_polling(allowed_updates=Application.ALL_ALLOWED_UPDATES)



if __name__ == '__main__':
    asyncio.run(init_bot())