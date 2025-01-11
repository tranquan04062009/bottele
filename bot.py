import logging
import asyncio
import os
from typing import List, Dict, Optional, Union
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler, CallbackContext
import aiohttp
import re
from datetime import datetime, timezone
import random
from http import HTTPStatus

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Constants ---
REPORT_STATE = 0
DEFAULT_REPORT_LOOPS = 1
MAX_REPORT_LOOPS = 5 # Avoid overload and getting ban
USER_AGENT_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Mobile/15E148 Safari/604.1"
]
ACCEPT_LANGUAGE_LIST = ["en-US,en;q=0.9", "vi-VN,vi;q=0.9", "fr-FR,fr;q=0.8","es-ES,es;q=0.9", "ko-KR,ko;q=0.9","zh-CN,zh;q=0.9"]
TIMEZONE_LIST = ["Asia/Ho_Chi_Minh", "America/New_York", "Europe/London", "Asia/Tokyo", "Europe/Paris","America/Los_Angeles"]

# --- Global Variables ---
user_data: Dict[str, Dict[str, str]] = {}  # In-memory user data
bot = None

# --- Helper Functions ---
def get_time() -> str:
    """Gets the current UTC time in a formatted string."""
    dt = datetime.now(timezone.utc)
    tzutc_format = "%Y-%m-%d %H:%M:%S %Z%z"
    return dt.strftime(tzutc_format)


# Get Environment Variables
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")


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
                 log_message = f"Lỗi khi tải URL: {url}, trạng thái: {response.status if response else 'N/A'}"
                 logging.error(log_message)
                 return None

    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logging.error(f"Lỗi khi tải URL: {url}, lỗi {e}")
        return None
    except Exception as e:
        logging.error(f"Lỗi không xác định khi tải URL: {url}, lỗi {e}")
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
                 log_message = f"Lỗi khi gửi dữ liệu đến URL: {url}, trạng thái: {response.status if response else 'N/A'}"
                 logging.error(log_message)
                 return None

    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logging.error(f"Lỗi khi gửi dữ liệu đến URL: {url}, lỗi {e}")
        return None
    except Exception as e:
         logging.error(f"Lỗi không xác định khi gửi dữ liệu đến URL: {url}, lỗi {e}")
         return None


def is_valid_facebook_id(fb_id: str) -> bool:
    """Checks if a given string is a valid Facebook ID format (numbers only)."""
    return re.match(r'^\d+$', fb_id) is not None
def is_valid_facebook_token(fb_token: str) -> bool:
    """Simple check if a token contains only alphanumerical characters with a max length of 100"""
    return re.match(r'^[a-zA-Z0-9]{1,100}$', fb_token) is not None
def generate_url(target_id: str) -> str:
   """Generate the URL if the user id or a page id"""
   if is_valid_facebook_id(target_id):
      return  f"https://mbasic.facebook.com/{target_id}"
   else:
       return f"https://mbasic.facebook.com/{target_id}" # If the id is a name, lets handle the request like it is a user

async def process_report_cycle(
    session: aiohttp.ClientSession,
    target_id: str,
    cookie: str,
    report_messages: List[str],
    time: str,
    loop_number: int,
) -> bool:
    """Processes a single reporting cycle with different types of reports."""
    report_types = [
        ('spam', 'spam'), ('fake_account', 'fake account'),
        ('violates_terms', 'violates terms'),
         ('bullying_harassment','bullying or harassment'),
        ('violence','Violence'),
        ('nudity','Nudity'),
         ('hate_speech','Hate Speech') ,
         ('terrorism', 'Terrorism'),
        ('not_qualified', 'This content is not qualified for community standars'),
          ('other', 'Other Reason')
         ]  # Added several more reasons

    for tag, reason in report_types: # Report a variety of reasons
        try:
          report_url = generate_url(target_id)
          headers = generate_fake_headers()
          data = await fetch_url(session, report_url, headers=headers, cookies={"cookie": cookie})

          if not data:
               error_msg = f"Lỗi khi tải URL đầu tiên (loại: {reason}): {report_url} cho target_id {target_id} lúc {time}"
               logging.error(error_msg)
               report_messages.append(error_msg)
               return False
          if "xs=deleted" in data:
                error_msg = f'Cookie đã hết hạn cho ID: {target_id}'
                logging.warning(error_msg)
                report_messages.append(error_msg)
                return False
          redirect_match = re.search(r'location: (.*?)\r\n', data)
          if redirect_match:
                 redirect_url = redirect_match.group(1).strip()
                 headers = generate_fake_headers(referer=report_url)
                 redirect_data = await fetch_url(session, redirect_url, headers=headers)

                 if not redirect_data:
                      error_msg = f'Lỗi khi tải URL chuyển hướng (loại: {reason}): {redirect_url} cho target_id: {target_id} lúc {time}'
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
                            error_msg = f'Lỗi tại link1 (loại: {reason}): {link1} cho id: {target_id} lúc {time}'
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
                                error_msg = f'Lỗi khi gửi dữ liệu tại route {link2} (loại: {reason}) cho target_id = {target_id} lúc {time}'
                                logging.error(error_msg)
                                report_messages.append(error_msg)
                                return False

                            headers = generate_fake_headers(referer=link2)
                            link3_data = await fetch_url(session, link3, headers=headers, cookies={"cookie": cookie})
                            if not link3_data:
                                    error_msg = f'Lỗi tại route cuối của report thứ nhất (loại: {reason}) cho target_id: {target_id} lúc {time}'
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
                                   data = f"fb_dtsg={fb_dtsg}&jazoest={jazoest}&tag={tag}&action=Gửi"
                                   headers = generate_fake_headers(referer=link3)
                                   headers['content-length'] = str(len(data))
                                   headers['content-type'] = "application/x-www-form-urlencoded"
                                   link5 = await post_data(session, link4, data, headers=headers, cookies={"cookie": cookie})
                                   if not link5:
                                     error_msg = f'Lỗi tại tag {reason} bước 2 cho ID:{target_id} lúc {time}'
                                     logging.error(error_msg)
                                     report_messages.append(error_msg)
                                     return False

                                   headers = generate_fake_headers(referer=link4)
                                   link5_data = await fetch_url(session, link5, headers=headers, cookies={"cookie": cookie})
                                   if not link5_data:
                                        error_msg = f'Lỗi tại route loại 2 (loại: {reason}), id: {target_id}, time = {time}'
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
                                               error_msg = f'Lỗi tại bước gửi report cuối (loại: {reason}), target_id = {target_id} lúc {time}'
                                               logging.error(error_msg)
                                               report_messages.append(error_msg)
                                               return False
                                             success_msg = f"Đã gửi report (loại: {reason}) thành công vòng thứ: {loop_number}, id: {target_id}"
                                             report_messages.append(success_msg)
                                             break
        except Exception as e:
            error_msg = f"Lỗi không xác định khi xử lý report (loại: {reason}) cho target id {target_id}: {e} lúc {time}"
            logging.exception(error_msg)
            report_messages.append(error_msg)
    return True

async def process_report(user_id: str, target_id: str, cookie: Optional[str], chat_id: int, loops: int):
    """Manages the report processing flow."""
    time = get_time()
    report_messages = []
    os.environ['TZ'] = generate_fake_time_zone()
    try:
        async with aiohttp.ClientSession() as session:
            for loop_number in range(1, loops + 1):
                logging.info(f"User {user_id} đang xử lý vòng {loop_number}/{loops} cho target {target_id} lúc {time}")
                report_messages.append(f"Bắt đầu vòng lặp {loop_number} - Báo cáo {target_id}")
                if not await process_report_cycle(session, target_id, cookie, report_messages, time, loop_number):
                    logging.error(f'Quá trình report dừng cho user: {user_id} với target id {target_id} tại vòng = {loop_number} lúc {time}')
                    break  # Stop on the first failure
            for msg in report_messages:
                 try:
                    await bot.send_message(chat_id=chat_id, text=msg)
                 except Exception as e:
                     logging.error(f'Lỗi khi gửi tin nhắn đến user {user_id} trong chat id {chat_id}: {msg}, lỗi = {e}')
    except Exception as e:
        error_msg = f"Lỗi trong quá trình report chính cho user_id: {user_id}: {e} lúc {time}"
        logging.exception(error_msg)
        await bot.send_message(chat_id=chat_id, text=error_msg)

    #Ask if the user wants to report again
    if user_id in user_data:
        await bot.send_message(chat_id=chat_id, text=f"Đã hoàn thành báo cáo cho {target_id}. Bạn có muốn báo cáo lại ID này không? (có/không)")
        return

# Command Handlers
async def ask_for_report(update: Update, context: CallbackContext) -> int:
    """Asks for the target ID."""
    await update.message.reply_text("Vui lòng nhập ID người dùng hoặc trang cần báo cáo:")
    return REPORT_STATE


async def ask_for_cookie_or_token(update: Update, context: CallbackContext) -> Optional[int]:
    """Asks for the cookie or token after receiving the target ID."""
    if 'target_id' in context.user_data:
        target_id = context.user_data['target_id']
        if is_valid_facebook_id(target_id):
           await update.message.reply_text('Tuyệt vời, bây giờ hãy gửi cookie Facebook:')
        else:
          await update.message.reply_text('Tuyệt vời, bây giờ hãy gửi token Facebook:')
        return REPORT_STATE + 1
    else:
      await update.message.reply_text('Lỗi: Bạn chưa nhập ID người dùng. Vui lòng dùng /set trước.')
      return ConversationHandler.END



async def save_id_cookie(update: Update, context: CallbackContext) -> int:
    """Saves the target ID and cookie or token to user data."""
    user_id = str(update.effective_user.id)
    target_id = context.user_data['target_id']
    cookie_or_token = update.message.text.strip()
    global user_data
    try:
        if is_valid_facebook_id(target_id):
            if is_valid_facebook_token(cookie_or_token) == False :
              user_data[user_id] = {"target_id": target_id, "cookie": cookie_or_token}
              await update.message.reply_text(f'Đã cập nhật dữ liệu người dùng với ID mục tiêu: {target_id}, độ dài cookie: {len(cookie_or_token)}')
              logging.info(f'Đã cập nhật report mới cho user {user_id}')
              return ConversationHandler.END
            else:
                await update.message.reply_text("Lỗi : bạn nhập cookie không đúng định dạng")
                return ConversationHandler.END
        elif is_valid_facebook_token(cookie_or_token) :
             user_data[user_id] = {"target_id": target_id, "token": cookie_or_token}
             await update.message.reply_text(f"Đã cập nhật dữ liệu với ID : {target_id} và Token dài: {len(cookie_or_token)}")
             logging.info(f'Đã cập nhật report mới cho user {user_id}')
             return ConversationHandler.END
        else :
            await update.message.reply_text("Lỗi, bạn đã nhập cookie hoặc token không đúng định dạng.")
            return ConversationHandler.END
    except Exception as error:
       await update.message.reply_text(f'Lỗi khi xử lý yêu cầu report: {error}')
       logging.error(f'Lỗi trong save_id_cookie cho user {user_id}: {error}')
       return ConversationHandler.END



async def start_report_command(update: Update, context: CallbackContext) -> None:
    """Starts the report process."""
    global bot, user_data
    user_id = str(update.effective_user.id)
    chat_id = update.message.chat_id
    logging.info(f'Command /report được thực thi cho user {user_id}')

    try:
         if user_id in user_data:
              user = user_data[user_id]
              target_id = user["target_id"]
              cookie = user.get("cookie")
              token = user.get("token")

              if cookie:
                   loops = DEFAULT_REPORT_LOOPS # set loop
                   if context.args and context.args[0].isdigit():
                         loops = int(context.args[0])
                         if loops > MAX_REPORT_LOOPS:
                             loops = MAX_REPORT_LOOPS
                   await bot.send_message(chat_id=chat_id, text=f'Bắt đầu {loops} report cho user hoặc id : {target_id}, với cookie có độ dài: {len(cookie) if cookie else "None"}')
                   asyncio.create_task(process_report(user_id, target_id, cookie, chat_id,loops))

              elif token:
                   loops = DEFAULT_REPORT_LOOPS
                   if context.args and context.args[0].isdigit():
                      loops = int(context.args[0])
                      if loops > MAX_REPORT_LOOPS:
                         loops = MAX_REPORT_LOOPS
                   await bot.send_message(chat_id=chat_id, text=f"Bắt đầu {loops} báo cáo cho id trang hoặc user: {target_id}, với token dài : {len(token)}")
                   asyncio.create_task(process_report(user_id, target_id, token, chat_id, loops))
         else:
             await update.message.reply_text("Bạn chưa thêm ID và cookie hoặc token để báo cáo.")

    except Exception as e:
       await update.message.reply_text(f"Lỗi khi thực thi lệnh: {e}")
       logging.error(f'Lỗi trong start_report_command cho user {user_id}: {e}')


async def handle_report_again(update: Update, context: CallbackContext) -> None:
    """Handle the case where the user answers if want to report again"""
    global bot, user_data
    user_id = str(update.effective_user.id)
    chat_id = update.message.chat_id
    text = update.message.text.strip().lower()

    if text == "có":
       if user_id in user_data:
          user = user_data[user_id]
          target_id = user["target_id"]
          cookie = user.get("cookie")
          token = user.get("token")

          if cookie:
             loops = DEFAULT_REPORT_LOOPS
             await bot.send_message(chat_id=chat_id, text=f'Bắt đầu report lại cho id : {target_id}, với cookie có độ dài: {len(cookie) if cookie else "None"}')
             asyncio.create_task(process_report(user_id, target_id, cookie, chat_id,loops))
          elif token:
             loops = DEFAULT_REPORT_LOOPS
             await bot.send_message(chat_id=chat_id, text=f'Bắt đầu report lại cho id : {target_id}, với token có độ dài: {len(token) if token else "None"}')
             asyncio.create_task(process_report(user_id, target_id, token, chat_id,loops))


       else:
            await update.message.reply_text("Không có dữ liệu để báo cáo lại.")
    elif text == "không":
         if user_id in user_data:
             del user_data[user_id]
             await update.message.reply_text("Đã xóa thông tin ID và cookie của bạn.")
    else:
      await update.message.reply_text("Vui lòng trả lời 'có' hoặc 'không'.")

async def successful_start(update: Update, context: CallbackContext) -> None:
    """Simple handler to check the bot is running"""
    await update.message.reply_text("Bot đang chạy")

async def cancel(update: Update, context: CallbackContext) -> int:
    """Cancels and ends the conversation."""
    await update.message.reply_text("Thao tác đã bị hủy.")
    return ConversationHandler.END


def init_bot():
    """Initialize bot and run it"""
    if not TOKEN:
        logging.critical('Telegram TOKEN không tìm thấy, hãy set env TELEGRAM_BOT_TOKEN')
        return
    application = Application.builder().token(TOKEN).build()
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('set', ask_for_report)],
        states={
           REPORT_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_for_cookie_or_token)],
           REPORT_STATE + 1: [MessageHandler(filters.TEXT & ~filters.COMMAND, save_id_cookie)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("start", successful_start))
    application.add_handler(CommandHandler("report", start_report_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_report_again))
    global bot
    bot = application.bot  # declare global var
    logging.info('Bot đã khởi động')
    application.run_polling(allowed_updates=["message", "callback_query", "inline_query"])


if __name__ == '__main__':
    init_bot()