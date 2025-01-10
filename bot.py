import logging
import re
import asyncio
import os
from telegram import Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, CallbackContext
import aiohttp
from bs4 import BeautifulSoup
from typing import Dict, Tuple, Optional

# Lấy token bot từ biến môi trường
BOT_TOKEN: Optional[str] = os.getenv("BOT_TOKEN")

# Kiểm tra xem BOT_TOKEN có tồn tại không
if not BOT_TOKEN:
    logging.error("Error: BOT_TOKEN environment variable is not set.")
    exit(1)

# Cấu hình logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# Lưu trữ thông tin theo dõi
tracked_apps: Dict[int, Dict[str, Tuple[str, str]]] = {}
tracked_tasks: Dict[Tuple[int, str], asyncio.Task] = {}

async def check_testflight_status(session: aiohttp.ClientSession, url: str) -> Optional[bool]:
    try:
        logger.info(f"Checking URL: {url}")
        async with session.get(url, timeout=10) as response:
            response.raise_for_status()  # Báo lỗi nếu request thất bại
            html = await response.text()

        soup = BeautifulSoup(html, 'html.parser')

        # Tìm đoạn text chứa thông tin đầy hay chưa
        text_elements = soup.find_all(string=re.compile(r'This beta is full|This beta is no longer accepting new testers', re.IGNORECASE))
        
        logger.info(f"Checked URL: {url}, status: {'available' if not text_elements else 'full'}")
        return not text_elements
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error(f"Error checking URL {url}: {e}")
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred while checking URL {url}: {e}")
        return None

async def send_update(bot: Application, chat_id: int, app_name: str, available: bool, url: str):
    if available:
        message = f"🔥 Ứng dụng/game '{app_name}' trên TestFlight đã có chỗ trống! Nhanh tay tải về: {url}"
        try:
            await bot.bot.send_message(chat_id=chat_id, text=message)
        except Exception as e:
            logger.exception(f"An error occurred while sending message: {e}")

async def check_and_notify(chat_id: int, url: str):
    while True:
        if chat_id not in tracked_apps or url not in tracked_apps[chat_id]:
             logger.info(f"Stop tracking, chat_id {chat_id} or url {url} not found")
             if (chat_id, url) in tracked_tasks:
                task = tracked_tasks[(chat_id, url)]
                task.cancel()
                del tracked_tasks[(chat_id,url)]
             return
        try:
          async with aiohttp.ClientSession() as session:
            available = await check_testflight_status(session, url)
            if available is None:
                logger.error(f"Failed to check status for URL: {url}, Skipped.")
                await asyncio.sleep(0)
                continue
            
            app_name = tracked_apps[chat_id][url][0]
            if available:
               await send_update(Application.bot, chat_id, app_name, available, url)
            
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            logger.info(f"Task check_and_notify for URL {url} has been cancelled")
            return
        except Exception as e:
            logger.exception(f"An unexpected error occurred in check_and_notify for URL {url}: {e}")
            await asyncio.sleep(0)
        


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
        task = asyncio.create_task(check_and_notify(chat_id, url))
        tracked_tasks[(chat_id, url)] = task


async def stop_tracking(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    if len(context.args) != 1:
        await update.message.reply_text("Sử dụng: /stop <url_testflight>")
        return
    
    url = context.args[0]
    if chat_id in tracked_apps and url in tracked_apps[chat_id]:
        if (chat_id, url) in tracked_tasks:
            task = tracked_tasks[(chat_id, url)]
            task.cancel()  # Hủy bỏ task
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
    try:
        application = ApplicationBuilder().token(BOT_TOKEN).build()

        application.add_handler(CommandHandler("autocheck", autocheck))
        application.add_handler(CommandHandler("stop", stop_tracking))
        application.add_error_handler(error_handler)
        
        await application.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.exception(f"An unexpected error occurred in main: {e}")

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())