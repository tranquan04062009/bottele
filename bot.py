import os
import random
import string
import time
import logging
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler, ConversationHandler
import requests

# Cấu hình logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Lấy Token từ Biến Môi Trường
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("Token chưa được thiết lập trong biến môi trường TELEGRAM_BOT_TOKEN.")

bot = Bot(TOKEN)

# Biến lưu danh sách spam sessions và blocked users
user_spam_sessions = {}
blocked_users = set()

# Hàm tạo deviceId ngẫu nhiên
def generate_device_id():
    return ''.join(random.choices(string.hexdigits.lower(), k=42))

# Hàm tạo User-Agent ngẫu nhiên
def generate_user_agent():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/109.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:92.0) Gecko/20100101 Firefox/92.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0"
    ]
    return random.choice(user_agents)

# Hàm gửi tin nhắn spam nhanh và nhiều hơn
def send_spam(username, message, chat_id, session_id):
    counter = 0
    while user_spam_sessions.get(chat_id, {}).get(session_id, {}).get('is_active', False):
        try:
            device_id = generate_device_id()
            user_agent = generate_user_agent()
            url = "https://ngl.link/api/submit"
            headers = {
                "User-Agent": user_agent,
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
            }
            body = {'username': username, 'question': message, 'deviceId': device_id, 'gameSlug': '', 'referrer': ''}
            response = requests.post(url, headers=headers, data=body)

            if response.status_code != 200:
                logger.warning("[Lỗi] Bị giới hạn, đang chờ 3 giây...")
                time.sleep(3)  # Giảm thời gian chờ để spam nhanh hơn
            else:
                counter += 1
                logger.info(f"[Tin nhắn] Phiên {session_id}: Đã gửi {counter} tin nhắn.")
                bot.send_message(chat_id, f"Phiên {session_id}: Đã gửi {counter} tin nhắn.")
            
            time.sleep(1)  # Giảm thời gian chờ để spam nhanh hơn

        except Exception as e:
            logger.error(f"[Lỗi] {e}")
            time.sleep(1)

# Kiểm tra xem người dùng có bị chặn không
def is_blocked(chat_id):
    return chat_id in blocked_users

# Bắt đầu với lệnh /start
def start(update: Update, context):
    chat_id = update.message.chat_id
    user_id = update.message.from_user.id

    if is_blocked(chat_id):
        bot.send_message(chat_id, "Bạn đã bị chặn khỏi việc sử dụng bot này.")
        return

    bot.send_message(chat_id, f"Chào mừng! ID Telegram của bạn là: {user_id}")

    if chat_id not in user_spam_sessions:
        user_spam_sessions[chat_id] = []

    keyboard = [
        [InlineKeyboardButton("Bắt đầu Spam", callback_data='start_spam'),
         InlineKeyboardButton("Danh sách Spam", callback_data='list_spam')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    bot.send_message(chat_id, "Chọn tính năng bạn muốn sử dụng:", reply_markup=reply_markup)

# Xử lý bắt đầu spam
def start_spam(update: Update, context):
    chat_id = update.callback_query.message.chat_id

    if is_blocked(chat_id):
        bot.send_message(chat_id, "Bạn đã bị chặn khỏi việc sử dụng bot này.")
        return

    bot.send_message(chat_id, "Nhập tên người dùng muốn spam:")
    return "WAITING_USERNAME"

# Xử lý nhận tên người dùng và tin nhắn
def waiting_username(update: Update, context):
    chat_id = update.message.chat_id
    username = update.message.text

    bot.send_message(chat_id, "Nhập tin nhắn bạn muốn gửi:")
    context.user_data['username'] = username
    return "WAITING_MESSAGE"

def waiting_message(update: Update, context):
    chat_id = update.message.chat_id
    message = update.message.text
    username = context.user_data['username']

    session_id = len(user_spam_sessions[chat_id]) + 1
    user_spam_sessions[chat_id].append({'id': session_id, 'username': username, 'message': message, 'is_active': True})

    send_spam(username, message, chat_id, session_id)

    bot.send_message(chat_id, f"Phiên spam {session_id} đã bắt đầu!")
    return ConversationHandler.END

# Xử lý danh sách spam
def list_spam(update: Update, context):
    chat_id = update.callback_query.message.chat_id

    if is_blocked(chat_id):
        bot.send_message(chat_id, "Bạn đã bị chặn khỏi việc sử dụng bot này.")
        return

    sessions = user_spam_sessions.get(chat_id, [])
    if sessions:
        list_message = "Danh sách các phiên spam hiện tại:\n"
        keyboard = []
        for session in sessions:
            list_message += f"{session['id']}: {session['username']} - {session['message']} [Hoạt động: {session['is_active']}]\n"
            keyboard.append([InlineKeyboardButton(f"Dừng phiên {session['id']}", callback_data=f"stop_{session['id']}")])

        reply_markup = InlineKeyboardMarkup(keyboard)
        bot.send_message(chat_id, list_message, reply_markup=reply_markup)
    else:
        bot.send_message(chat_id, "Không có phiên spam nào đang hoạt động.")

# Xử lý dừng phiên
def stop_spam(update: Update, context):
    chat_id = update.callback_query.message.chat_id
    session_id = int(update.callback_query.data.split("_")[1])

    session = next((s for s in user_spam_sessions.get(chat_id, []) if s['id'] == session_id), None)
    if session:
        session['is_active'] = False
        bot.send_message(chat_id, f"Phiên spam {session_id} đã bị dừng.")
    else:
        bot.send_message(chat_id, f"Không tìm thấy phiên spam với ID {session_id}.")

# Hàm chính để chạy bot
def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start), CallbackQueryHandler(start_spam, pattern='^start_spam$')],
        states={
            "WAITING_USERNAME": [MessageHandler(Filters.text, waiting_username)],
            "WAITING_MESSAGE": [MessageHandler(Filters.text, waiting_message)]
        },
        fallbacks=[CallbackQueryHandler(list_spam, pattern='^list_spam$'), CallbackQueryHandler(stop_spam, pattern='^stop_')]
    )

    dp.add_handler(conv_handler)

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()