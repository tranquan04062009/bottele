import logging
import asyncio
import os
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler, ConversationHandler
import aiohttp
import re
from datetime import datetime, timezone
from telegram import Update
from telegram.ext import CallbackContext
import json
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---
REPORT_STATE = 0
DATA_FILE = "user_data.json"
DEFAULT_REPORT_LOOPS = 1  # Set here the default loop cycle when no arg was given at command
def get_time():
    dt = datetime.now(timezone.utc)
    utc_time = dt.replace(tzinfo=timezone.utc)
    tzutc_format = "%Y-%m-%d %H:%M:%S %Z%z"
    time = utc_time.strftime(tzutc_format)
    return time
# Get Environment Variables directly
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# Fake Data Set
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Mobile/15E148 Safari/604.1"
]
ACCEPT_LANGUAGES = ["en-US,en;q=0.9", "vi-VN,vi;q=0.9", "fr-FR,fr;q=0.8","es-ES,es;q=0.9", "ko-KR,ko;q=0.9","zh-CN,zh;q=0.9"]
TIMEZONES = ["Asia/Ho_Chi_Minh", "America/New_York", "Europe/London", "Asia/Tokyo", "Europe/Paris","America/Los_Angeles"]

async def init():
    # Check if TOKEN available
    if not TOKEN:
        logging.critical('Telegram TOKEN not found, set env TELEGRAM_BOT_TOKEN')
        return False
    return  True
# Load user data from file
def load_user_data():
     try:
          with open(DATA_FILE, "r") as f:
             return json.load(f)
     except FileNotFoundError:
         return {}
# Save user data to json
def save_user_data(data):
     with open(DATA_FILE,"w") as f:
        json.dump(data,f)
# Define an async function for a successful initialization message
async def successful_start(update: Update, context: CallbackContext):
  await update.message.reply_text("Xin chào, công cụ NVT đã kích hoạt", parse_mode='MarkdownV2')

# Under 18 Reason (Burmese)
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
async def process_report(user_id, target_id, cookie, chat_id, loops):
    dem = 0
    if not target_id or not cookie:
       await bot.send_message(chat_id=chat_id,text=f'Vui lòng thêm cookie và id vào database, chưa thêm! {cookie} hoặc {target_id}')
       return
    time = get_time()
    os.environ['TZ'] = generate_fake_time_zone()
    report_messages = []  # Store response here before send bulk to the client.
    while dem < loops:
        dem += 1
        try:
            logging.info(f"User {user_id} started loop target {target_id} counter {dem} - Time {time}")
            report_messages.append(f"Bắt đầu vòng lặp thứ {dem} - Báo cáo đến {target_id}")
            url = f"https://mbasic.facebook.com/{target_id}"
            headers = generate_fake_headers()
            try:
                 async with aiohttp.ClientSession() as session:
                    try:
                        async with session.get(url, headers=headers, cookies={"cookie": cookie}, timeout=20) as response:
                            if response.status != 200:
                                error_msg = f'Lỗi khi báo cáo bước 1 - Status not 200, target {target_id}, mã phản hồi là {response.status} tại thời điểm {time}'
                                logging.error(error_msg)
                                report_messages.append(error_msg)
                                await bot.send_message(chat_id=chat_id, text=error_msg)
                                continue
                            data = await response.text()
                            if "xs=deleted" in data:
                                error_msg = f'Cookie đã hết hạn cho ID: {target_id}'
                                logging.warning(error_msg)
                                report_messages.append(error_msg)
                                await bot.send_message(chat_id=chat_id, text=error_msg)
                                return
                            # get redirect
                            redirect_match = re.search(r'location: (.*?)\r\n', data)
                            if redirect_match:
                                urldata = redirect_match.group(1).strip()
                                headers = generate_fake_headers(referer=url)
                                try:
                                    async with session.get(urldata, headers=headers, timeout=20) as response_2:
                                        if response_2.status != 200:
                                             error_msg = f'Lỗi trên redirect khi report cho ID {target_id}, mã status: {response_2.status} time:{time}'
                                             logging.error(error_msg)
                                             report_messages.append(error_msg)
                                             await bot.send_message(chat_id=chat_id, text=error_msg)
                                             continue
                                        a = await response_2.text()
                                        data_match = re.search(r'/nfx/basic/direct_actions/(.*?)"',a)
                                        if data_match:
                                            data = data_match.group(1).strip()
                                            l1 = data.split("amp;")[0]
                                            l2 = data.split("amp;")[1]
                                            l3 = data.split("amp;")[2]
                                            link1 = f"https://mbasic.facebook.com/nfx/basic/direct_actions/{l1}{l2}{l3}"
                                            headers = generate_fake_headers(referer=urldata)
                                            try:
                                                async with session.get(link1, headers=headers, cookies={"cookie": cookie}, timeout=20) as response_3:
                                                     if response_3.status != 200:
                                                        error_msg =  f'Lỗi trên link1 , id: {target_id}, link : {link1} , status code = {response_3.status} at time {time}'
                                                        logging.error(error_msg)
                                                        report_messages.append(error_msg)
                                                        await bot.send_message(chat_id=chat_id, text=error_msg)
                                                        continue
                                                     a = await response_3.text()
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
                                                            try:
                                                                async with session.post(link2, data=data, headers=headers, cookies={"cookie": cookie}, allow_redirects=True, timeout=20) as response_4:
                                                                    if response_4.status != 200:
                                                                          error_msg = f'Lỗi tại resolve code at route {link2} status = {response_4.status} and target_id= {target_id}, at time:{time}'
                                                                          logging.error(error_msg)
                                                                          report_messages.append(error_msg)
                                                                          await bot.send_message(chat_id=chat_id, text=error_msg)
                                                                          continue
                                                                    link3 = str(response_4.url)
                                                                    headers = generate_fake_headers(referer=link2)
                                                                    try:
                                                                        async with session.get(link3, headers=headers, cookies={"cookie": cookie}, timeout=20) as response_5:
                                                                             if response_5.status != 200:
                                                                                   error_msg = f'Lỗi route cuối cùng ở report thứ nhất, status code= {response_5.status} target_id: {target_id}, time:{time}'
                                                                                   logging.error(error_msg)
                                                                                   report_messages.append(error_msg)
                                                                                   await bot.send_message(chat_id=chat_id, text=error_msg)
                                                                                   continue
                                                                             a = await response_5.text()
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
                                                                                   try:
                                                                                        async with session.post(link4, data=data, headers=headers, cookies={"cookie": cookie}, allow_redirects=True, timeout=20) as response_6:
                                                                                              if response_6.status != 200:
                                                                                                  error_msg = f'Lỗi tag spam bước 2 khi report đến ID:{target_id}, Status = {response_6.status} tại thời điểm: {time}'
                                                                                                  logging.error(error_msg)
                                                                                                  report_messages.append(error_msg)
                                                                                                  await bot.send_message(chat_id=chat_id, text=error_msg)
                                                                                                  continue
                                                                                              link5 = str(response_6.url)
                                                                                              headers = generate_fake_headers(referer=link4)
                                                                                              try:
                                                                                                    async with session.get(link5, headers=headers, cookies={"cookie": cookie}, timeout=20) as response_7:
                                                                                                        if response_7.status != 200:
                                                                                                           error_msg = f'Lỗi tại route type 2 tại {link5}, mã = {response_7.status}, time = {time}, id: {target_id}'
                                                                                                           logging.error(error_msg)
                                                                                                           report_messages.append(error_msg)
                                                                                                           await bot.send_message(chat_id=chat_id, text=error_msg)
                                                                                                           continue
                                                                                                        a = await response_7.text()
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
                                                                                                                try:
                                                                                                                      async with session.post(link6, data=data, headers=headers, cookies={"cookie": cookie}, allow_redirects=True, timeout=20) as response_8:
                                                                                                                             if response_8.status != 200:
                                                                                                                                 error_msg = f'Lỗi khi report final , target_id = {target_id}, Status code = {response_8.status} at {link6} at time {time}'
                                                                                                                                 logging.error(error_msg)
                                                                                                                                 report_messages.append(error_msg)
                                                                                                                                 await bot.send_message(chat_id=chat_id, text=error_msg)
                                                                                                                                 continue
                                                                                                                             success_msg = f"Đã gửi report thành công vòng thứ: {dem} đến id: {target_id}"
                                                                                                                             report_messages.append(success_msg)
                                                                                                                             await bot.send_message(chat_id=chat_id, text=success_msg)
                                                                                                                except Exception as e:
                                                                                                                    error_msg = f'Lỗi report bước cuối , mã = {e}, time : {time}, and id = {target_id} , try again'
                                                                                                                    logging.error(error_msg)
                                                                                                                    report_messages.append(error_msg)
                                                                                                                    await bot.send_message(chat_id=chat_id, text=error_msg)
                                                                                                                    continue
                                                                                              except Exception as e:
                                                                                                 error_msg = f'Lỗi khi route type 2 logic {e} time: {time}, target_id: {target_id}, try again '
                                                                                                 logging.error(error_msg)
                                                                                                 report_messages.append(error_msg)
                                                                                                 await bot.send_message(chat_id=chat_id, text=error_msg)
                                                                                                 continue
                                                                                    except Exception as e:
                                                                                         error_msg= f'Lỗi ở bước logic type 1 {e} and target: {target_id}, time: {time} and will try again'
                                                                                         logging.error(error_msg)
                                                                                         report_messages.append(error_msg)
                                                                                         await bot.send_message(chat_id=chat_id, text=error_msg)
                                                                                         continue
                                                                     except Exception as e:
                                                                        error_msg =  f'Lỗi bất thường , handle action route = {e}   target = {target_id}, time:{time} and try again'
                                                                        logging.error(error_msg)
                                                                        report_messages.append(error_msg)
                                                                        await bot.send_message(chat_id=chat_id, text=error_msg)
                                                                        continue
                                                            except Exception as e:
                                                                 error_msg = f'Lỗi Redirect khi báo cáo ID {target_id} and error: {e}, time: {time} try again'
                                                                 logging.error(error_msg)
                                                                 report_messages.append(error_msg)
                                                                 await bot.send_message(chat_id=chat_id,text=error_msg)
                                                                 continue
                                                except Exception as e:
                                                     error_msg = f'Lỗi yêu cầu đầu , ID = {target_id} , Error: {e} , time {time} try again '
                                                     logging.error(error_msg)
                                                     report_messages.append(error_msg)
                                                     await bot.send_message(chat_id=chat_id, text=error_msg)
                                                     continue

                            except asyncio.TimeoutError:
                                error_msg = f"Lỗi timeout trên ID:{target_id}, time = {time} - Hãy thử lại "
                                logging.error(error_msg)
                                report_messages.append(error_msg)
                                await bot.send_message(chat_id=chat_id, text=error_msg)
                                continue

                    except Exception as e:
                        error_msg =  f'Lỗi hệ thống, code main loop lý do= {e}, target_id = {target_id} and time= {time} please try again later'
                        logging.error(error_msg)
                        report_messages.append(error_msg)
                        await bot.send_message(chat_id=chat_id, text=error_msg)
                        return
            except Exception as error:
              error_msg =  f'Lỗi không xác định nguyên nhân = {error} - ID: {target_id}, time = {time}'
              logging.error(error_msg)
              report_messages.append(error_msg)
              await bot.send_message(chat_id=chat_id, text=error_msg)
              return
    except Exception as e:
      error_msg =  f'Lỗi hệ thống outer loop= {e}, target_id = {target_id} and time= {time}'
      logging.error(error_msg)
      report_messages.append(error_msg)
      await bot.send_message(chat_id=chat_id,text=error_msg)
      return


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