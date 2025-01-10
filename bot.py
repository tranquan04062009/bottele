import logging
import re
import asyncio
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
import aiohttp
from bs4 import BeautifulSoup

# Thay thế bằng token bot của bạn
BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"

# Cấu hình logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# Lưu trữ thông tin theo dõi
tracked_apps = {}
tracked_tasks = {}

async def check_testflight_status(session, url):
    try:
        async with session.get(url, timeout=10) as response:
            response.raise_for_status()  # Báo lỗi nếu request thất bại
            html = await response.text()

        soup = BeautifulSoup(html, 'html.parser')

        # Tìm đoạn text chứa thông tin đầy hay chưa
        text_elements = soup.find_all(string=re.compile(r'This beta is full|This beta is no longer accepting new testers', re.IGNORECASE))

        return not text_elements
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error(f"Error checking URL {url}: {e}")
        return None


async def send_update(chat_id, app_name, available, url):
    if available:
        message = f"🔥 Ứng dụng/game '{app_name}' trên TestFlight đã có chỗ trống! Nhanh tay tải về: {url}"
    else:
        message = f"⚠️ Ứng dụng/game '{app_name}' trên TestFlight đã đầy!"
    await bot.send_message(chat_id=chat_id, text=message)


async def check_and_notify(chat_id, url, session):
    while chat_id in tracked_apps and url in tracked_apps[chat_id]:
        logger.info(f"Checking URL: {url} for chat_id: {chat_id}")
        available = await check_testflight_status(session, url)
        if available is None:
            logger.error(f"Failed to check status for URL: {url}, Skipped.")
            await asyncio.sleep(30)
            continue
        
        app_name = tracked_apps[chat_id][url][0]
        previous_available = tracked_apps[chat_id][url][2] if len(tracked_apps[chat_id][url]) > 2 else None

        if previous_available is None or available != previous_available:
            await send_update(chat_id, app_name, available, url)
            tracked_apps[chat_id][url] = (app_name, url, available)
        else:
            logger.info(f"No change in status for {url}, skipping notification")
        
        await asyncio.sleep(30)


async def autocheck(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    if len(context.args) != 1:
        await update.message.reply_text("Sử dụng: /autocheck <url_testflight>")
        return

    url = context.args[0]
    if not url.startswith("https://testflight.apple.com/join/"):
         await update.message.reply_text("Url phải bắt đầu bằng https://testflight.apple.com/join/")
         return
    
    app_name = url.split("join/")[-1]
    if chat_id not in tracked_apps:
        tracked_apps[chat_id] = {}
    tracked_apps[chat_id][url] = (app_name, url)
    await update.message.reply_text(f"Đã thêm theo dõi cho ứng dụng/game '{app_name}'.")
    
    # Khởi chạy task cho người dùng và url
    if (chat_id, url) not in tracked_tasks:
        session = aiohttp.ClientSession()
        task = asyncio.create_task(check_and_notify(chat_id, url, session))
        tracked_tasks[(chat_id, url)] = (task,session)


async def stop_tracking(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    if len(context.args) != 1:
        await update.message.reply_text("Sử dụng: /stop <url_testflight>")
        return
    
    url = context.args[0]
    if chat_id in tracked_apps and url in tracked_apps[chat_id]:
        
        if (chat_id, url) in tracked_tasks:
            task,session = tracked_tasks[(chat_id, url)]
            task.cancel()  # Hủy bỏ task
            await session.close()
            del tracked_tasks[(chat_id,url)]
        del tracked_apps[chat_id][url]
        if not tracked_apps[chat_id]:
            del tracked_apps[chat_id]  # Nếu không còn url nào, xóa thông tin user
        await update.message.reply_text(f"Đã dừng theo dõi ứng dụng/game '{url}'.")
    else:
        await update.message.reply_text("Không tìm thấy ứng dụng này trong danh sách theo dõi của bạn.")


async def error_handler(update: Update, context: CallbackContext):
    logger.error(f"Update {update} caused error {context.error}")


async def main():
    global bot
    updater = Updater(BOT_TOKEN)
    bot = updater.bot
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("autocheck", autocheck))
    dispatcher.add_handler(CommandHandler("stop", stop_tracking))
    dispatcher.add_error_handler(error_handler)

    await updater.start_polling(allowed_updates=Update.ALL_TYPES)
    
if __name__ == '__main__':
    asyncio.run(main())