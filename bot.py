import logging
import asyncio
import os
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler
import aiohttp
import re
from datetime import datetime, timezone
from telegram import Update
from telegram.ext import CallbackContext
import json
import random

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
ACCEPT_LANGUAGES = ["en-US,en;q=0.9", "vi-VN,vi;q=0.9", "fr-FR,fr;q=0.8","es-ES,es;q=0.9", "ko-KR,ko;q=0.9","zh-CN,zh;q=0.9"]
TIMEZONES = ["Asia/Ho_Chi_Minh", "America/New_York", "Europe/London", "Asia/Tokyo", "Europe/Paris","America/Los_Angeles"]

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

# Under 18 and 13 Reason
under_18_burmese_report = """
မင်္ဂလာပါ Facebook၊
ကျွန်တော်/ကျွန်မ ဒီအကောင့်ကို ၁၃ နှစ်မပြည့်သေးသော အသုံးပြုသူကြောင့် Facebook ၏ အသက်အရွယ် ပေါ်မှာ အားနည်းမှုဖြစ်ပြန်လည်ကြောင်း တင်ပြလိုပါသည်။ ကွဲပြားသောစည်းကမ်းများအရ ကွဲပြားစွာစစ်ဆေးပြီး အခြားသတ်မှတ်ချက်များအတိုင်း အကောင့်အားစစ်ဆေးပြင်ဆင်ပေးရန် မေတ္တာရပ်ခံပါသည်။
ကျေးဇူးတင်ပါတယ်။
လေးစားစွာဖြင့်
"""
under_13_burmese_report = """
မင်္ဂလာပါ Facebook၊
ကျွန်တော်/ကျွန်မ ဒီအကောင့်ကို ၁၃ နှစ်အောက် အသုံးပြုသူတစ်ဦး ဖြစ်လို့ Facebook ရဲ့အသိုင်းအဝိုင်းစည်းမျဉ်းနဲ့မကိုက်ညီလို့ ဒီအကောင့်ကိုဖယ်ရှားပေးဖို့ တင်ပြအပ်ပါတယ်။
ကျေးဇူးတင်ပါတယ်ခင်ဗျာ။
လေးစားစွာဖြင့်။
"""
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
             response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
             return await response.text()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logging.error(f"Error fetching URL: {url}, error {e}")
        return None
    except Exception as e:
       logging.error(f"Unexpected error on URL : {url}, error {e}")
       return None
async def post_data(session: aiohttp.ClientSession, url: str, data: str, headers: dict = None, cookies: dict = None, timeout: int = 20) -> str:
     try:
         async with session.post(url, data=data, headers=headers, cookies=cookies, allow_redirects=True, timeout=timeout) as response:
              response.raise_for_status()
              return str(response.url)
     except (aiohttp.ClientError, asyncio.TimeoutError) as e:
          logging.error(f"Error posting to URL: {url}, error {e}")
          return None
     except Exception as e:
          logging.error(f"Unexpected error posting on URL : {url}, error {e}")
          return None

async def process_report_cycle(session: aiohttp.ClientSession, target_id: str, cookie: str, report_messages: list, time: str) -> bool:
    try:
        url = f"https://mbasic.facebook.com/{target_id}"
        headers = generate_fake_headers()
        data = await fetch_url(session, url, headers=headers, cookies={"cookie": cookie})
        if not data:
            error_msg = f"Lỗi khi gọi URL 1: {url} , target_id {target_id} at {time}"
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
            urldata = redirect_match.group(1).strip()
            headers = generate_fake_headers(referer=url)
            a = await fetch_url(session, urldata, headers=headers)
            if not a:
                error_msg = f'Lỗi khi gọi url redirect: {urldata}, for id: {target_id}, at {time}'
                logging.error(error_msg)
                report_messages.append(error_msg)
                return False
            data_match = re.search(r'/nfx/basic/direct_actions/(.*?)"',a)
            if data_match:
                data = data_match.group(1).strip()
                l1 = data.split("amp;")[0]
                l2 = data.split("amp;")[1]
                l3 = data.split("amp;")[2]
                link1 = f"https://mbasic.facebook.com/nfx/basic/direct_actions/{l1}{l2}{l3}"
                headers = generate_fake_headers(referer=urldata)
                a = await fetch_url(session, link1, headers=headers, cookies={"cookie": cookie})
                if not a:
                    error_msg = f'Lỗi tại link1 , id: {target_id}, link : {link1} at time: {time}'
                    logging.error(error_msg)
                    report_messages.append(error_msg)
                    return False
                data_match = re.search(r'/nfx/basic/handle_action/(.*?)"', a)
                if data_match:
                    data = data_match.group(1).strip()
                    z1 = data.split('amp;')[0]
                    z2 = data.split('amp;')[1]
                    z3 = data.split('amp;')[2]
                    z4 = data.split('amp;')[3]
                    z5 = data.split('amp;')[4]
                    z6 = data.split('amp;')[5]
                    fb_dtsg_match = re.search(r'name="fb_dtsg" value="(.*?)"', a)
                    jazoest_match = re.search(r'name="jazoest" value="(.*?)"', a)
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
                              error_msg = f'Lỗi tại resolve code at route {link2} and target_id = {target_id}, at time:{time}'
                              logging.error(error_msg)
                              report_messages.append(error_msg)
                              return False
                        headers = generate_fake_headers(referer=link2)
                        a = await fetch_url(session, link3, headers=headers, cookies={"cookie": cookie})
                        if not a:
                            error_msg = f'Lỗi route cuối cùng ở report thứ nhất,  target_id: {target_id}, at time:{time}'
                            logging.error(error_msg)
                            report_messages.append(error_msg)
                            return False
                        data_match = re.search(r'/ixt/screen/frxtagselectionscreencustom/post/msite/(.*?)"', a)
                        if data_match:
                           data = data_match.group(1).strip()
                           x1 = data.split('amp;')[0]
                           x2 = data.split('amp;')[1]
                           fb_dtsg_match = re.search(r'name="fb_dtsg" value="(.*?)"', a)
                           jazoest_match = re.search(r'name="jazoest" value="(.*?)"', a)
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
                                   error_msg = f'Lỗi tag spam bước 2 khi report đến ID:{target_id}, at time: {time}'
                                   logging.error(error_msg)
                                   report_messages.append(error_msg)
                                   return False
                                headers = generate_fake_headers(referer=link4)
                                a = await fetch_url(session, link5, headers=headers, cookies={"cookie": cookie})
                                if not a:
                                    error_msg = f'Lỗi tại route type 2 tại , id: {target_id}, time = {time}'
                                    logging.error(error_msg)
                                    report_messages.append(error_msg)
                                    return False
                                data_match = re.search(r'/rapid_report/basic/actions/post/(.*?)"', a)
                                if data_match:
                                    data = data_match.group(1).strip()
                                    x1 = data.split('amp;')[0]
                                    x2 = data.split('amp;')[1]
                                    x3 = data.split('amp;')[2]
                                    x4 = data.split('amp;')[3]
                                    fb_dtsg_match = re.search(r'name="fb_dtsg" value="(.*?)"', a)
                                    jazoest_match = re.search(r'name="jazoest" value="(.*?)"', a)
                                    if fb_dtsg_match and jazoest_match:
                                          fb_dtsg = fb_dtsg_match.group(1).strip()
                                          jazoest = jazoest_match.group(1).strip()
                                          link6 = f"https://mbasic.facebook.com/rapid_report/basic/actions/post/{x1}{x2}{x3}{x4}"
                                          data = f"fb_dtsg={fb_dtsg}&jazoest={jazoest}&action=Gửi"
                                          headers = generate_fake_headers(referer=link5)
                                          headers['content-length'] = str(len(data))
                                          headers['content-type'] = "application/x-www-form-urlencoded"
                                          result = await post_data(session,link6,data,headers=headers, cookies={"cookie": cookie})
                                          if not result:
                                              error_msg = f'Lỗi khi report final , target_id = {target_id} at {time}'
                                              logging.error(error_msg)
                                              report_messages.append(error_msg)
                                              return False
                                          success_msg = f"Đã gửi report thành công vòng thứ: {dem} đến id: {target_id}"
                                          report_messages.append(success_msg)
                                          return True
    except Exception as e:
         error_msg = f"Lỗi inesperado al process report para target id {target_id} causa: {e}  time:{time}"
         logging.error(error_msg)
         report_messages.append(error_msg)
         return False
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
                if not await process_report_cycle(session, target_id, cookie,report_messages, time):
                    logging.error(f'Process report stop for user: {user_id} with target id {target_id} , at loop = {dem} and time = {time}')
                    break
            for msg in report_messages:
                await bot.send_message(chat_id=chat_id, text=msg)
    except Exception as e:
        error_msg = f"Error main report,  error: {e} at user_id: {user_id}, at time: {time}"
        logging.error(error_msg)
        await bot.send_message(chat_id=chat_id, text=error_msg)

# Command Handler for add report target to db
async def ask_for_report(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text("Vui lòng nhập ID người dùng cần báo cáo:")
    return REPORT_STATE

async def ask_for_cookie(update: Update, context: CallbackContext):
    context.user_data['target_id'] = update.message.text.strip()
    await update.message.reply_text('Tuyệt vời, giờ hãy gửi cookie facebook vào bot:')
    return  REPORT_STATE + 1

#save with json
async def save_id_cookie(update: Update, context: CallbackContext):
    user_id = str(update.effective_user.id)
    target_id = context.user_data['target_id']
    cookie = update.message.text.strip()
    user_data = load_user_data()
    try:
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
# start report command handler
async def start_report_command(update: Update, context: CallbackContext):
    user_id = str(update.effective_user.id)
    chat_id = update.message.chat_id
    logging.info(f'Command report executed for user {user_id}')
    user_data = load_user_data()
    try:
        if user_id in user_data:
           target_id, cookie = user_data[user_id]["target_id"], user_data[user_id]["cookie"]
           loops = DEFAULT_REPORT_LOOPS
           if context.args and context.args[0].isdigit():
              loops = int(context.args[0])
              await bot.send_message(chat_id=chat_id, text=f'Bắt đầu xử lý {loops} report, đến người dùng: {target_id} , độ dài cookie: {len(cookie)}')
           else:
              await bot.send_message(chat_id=chat_id, text=f'Bắt đầu xử lý báo cáo người dùng: {target_id} với độ dài cookie là: {len(cookie)}')
           asyncio.create_task(process_report(user_id, target_id, cookie, chat_id,loops))
        else:
           await update.message.reply_text("Bạn chưa thêm id và cookie báo cáo.")
    except Exception as e:
        await update.message.reply_text(f"Lỗi khi thực hiện lệnh: {e}")
        logging.error(f'Error at  start_report_command user= {user_id} error {e}')
async def cancel(update: Update, context: CallbackContext) -> int:
    """Cancels and ends the conversation."""
    await update.message.reply_text("Đã hủy thao tác")
    return ConversationHandler.END

if __name__ == '__main__':
    status = asyncio.run(init())
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
       application.run_polling(allowed_updates=telegram.ext.Application.ALL_ALLOWED_UPDATES)