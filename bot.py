import os
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
import requests

# Lấy API Key từ biến môi trường
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
YEUMONEY_TOKEN = os.getenv("YEUMONEY_TOKEN")

# URL API Yeumoney
YEUMONEY_API_URL = "https://yeumoney.com/api/v1/"

def start(update: Update, context: CallbackContext):
    """Gửi tin nhắn chào mừng"""
    welcome_text = (
        "🎉 **Chào mừng bạn đến với bot thống kê Yeumoney!**\n\n"
        "🔹 **/thongke**: Xem thống kê tài khoản Yeumoney\n"
        "🔹 **/rutgon [URL]**: Rút gọn link để kiếm tiền\n"
        "🔹 **/trogiup**: Hướng dẫn sử dụng bot\n"
    )
    update.message.reply_text(welcome_text, parse_mode="Markdown")

def thongke(update: Update, context: CallbackContext):
    """Lấy thông tin thống kê Yeumoney"""
    try:
        response = requests.get(f"{YEUMONEY_API_URL}user", headers={"Authorization": YEUMONEY_API_KEY})
        data = response.json()

        if response.status_code == 200:
            stats = (
                f"📊 **Thống kê tài khoản Yeumoney**\n\n"
                f"👤 **Tên tài khoản**: {data['data']['name']}\n"
                f"💰 **Số dư**: {data['data']['balance']} VNĐ\n"
                f"🔗 **Số link rút gọn**: {data['data']['links_count']}\n"
                f"📈 **Thu nhập hôm nay**: {data['data']['today_earning']} VNĐ\n"
                f"📊 **Thu nhập tháng này**: {data['data']['month_earning']} VNĐ\n"
            )
            update.message.reply_text(stats, parse_mode="Markdown")
        else:
            update.message.reply_text("❌ Không thể lấy thống kê. Vui lòng kiểm tra API Key!")
    except Exception as e:
        update.message.reply_text(f"❌ Lỗi: {e}")

def rutgon(update: Update, context: CallbackContext):
    """Rút gọn link bằng Yeumoney"""
    if len(context.args) == 0:
        update.message.reply_text("❌ Vui lòng nhập link cần rút gọn. Ví dụ: `/rutgon https://example.com`", parse_mode="Markdown")
        return

    long_url = context.args[0]
    try:
        response = requests.post(
            f"{YEUMONEY_API_URL}shorten",
            headers={"Authorization": YEUMONEY_API_KEY},
            data={"url": long_url}
        )
        data = response.json()

        if response.status_code == 200:
            short_url = data['data']['shortenedUrl']
            update.message.reply_text(f"✅ Link rút gọn: {short_url}")
        else:
            update.message.reply_text("❌ Không thể rút gọn link. Vui lòng kiểm tra link hoặc API Key!")
    except Exception as e:
        update.message.reply_text(f"❌ Lỗi: {e}")

def trogiup(update: Update, context: CallbackContext):
    """Gửi hướng dẫn sử dụng"""
    help_text = (
        "🛠 **Hướng dẫn sử dụng bot**:\n"
        "🔹 **/thongke**: Hiển thị thống kê tài khoản Yeumoney của bạn\n"
        "🔹 **/rutgon [URL]**: Rút gọn link để kiếm tiền từ Yeumoney\n"
    )
    update.message.reply_text(help_text, parse_mode="Markdown")

def main():
    # Khởi tạo bot
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    # Thêm các lệnh vào bot
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("thongke", thongke))
    dispatcher.add_handler(CommandHandler("rutgon", rutgon))
    dispatcher.add_handler(CommandHandler("trogiup", trogiup))

    # Bắt đầu chạy bot
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()