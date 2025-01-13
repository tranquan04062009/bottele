import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import requests

# Lấy API Key từ biến môi trường
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
YEUMONEY_TOKEN = os.getenv("YEUMONEY_TOKEN")

# URL API Yeumoney
YEUMONEY_API_URL = "https://yeumoney.com/api/v1/"

# Bộ nhớ cục bộ để lưu link rút gọn
shortened_links = []

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gửi tin nhắn chào mừng"""
    welcome_text = (
        "🎉 Chào mừng bạn đến với bot thống kê Yeumoney!\n\n"
        "💡 Các tính năng của bot:\n"
        "1️⃣ /thongke - Xem thống kê tài khoản Yeumoney\n"
        "2️⃣ /rutgon [URL] - Rút gọn link để kiếm tiền\n"
        "3️⃣ /listlinks - Xem danh sách link đã rút gọn\n"
        "4️⃣ /trogiup - Hướng dẫn sử dụng bot\n\n"
        "👉 Được phát triển bởi Yeumoney."
    )

    buttons = [
        [InlineKeyboardButton("🔗 Truy cập Yeumoney", url="https://yeumoney.com")],
        [
            InlineKeyboardButton("📊 Thống kê", callback_data="thongke"),
            InlineKeyboardButton("✂️ Rút gọn link", callback_data="rutgon")
        ]
    ]
    keyboard = InlineKeyboardMarkup(buttons)

    await update.message.reply_text(welcome_text, reply_markup=keyboard)

async def thongke(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lấy thông tin thống kê Yeumoney"""
    try:
        response = requests.get(f"{YEUMONEY_API_URL}user", headers={"Authorization": YEUMONEY_TOKEN})
        data = response.json()

        if response.status_code == 200:
            stats = (
                "📊 Thống kê tài khoản Yeumoney\n\n"
                f"👤 Tên tài khoản: {data['data']['name']}\n"
                f"💰 Số dư: {data['data']['balance']} VNĐ\n"
                f"🔗 Số link rút gọn: {data['data']['links_count']}\n"
                f"📈 Thu nhập hôm nay: {data['data']['today_earning']} VNĐ\n"
                f"📊 Thu nhập tháng này: {data['data']['month_earning']} VNĐ\n"
                f"💸 Tổng thu nhập: {data['data']['total_earning']} VNĐ"
            )
            await update.callback_query.edit_message_text(stats)
        else:
            error_message = data.get("message", "Không thể lấy thống kê. Vui lòng kiểm tra API Key!")
            await update.callback_query.edit_message_text(f"❌ {error_message}")
    except Exception as e:
        await update.callback_query.edit_message_text(f"❌ Lỗi: {e}")

async def rutgon(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Rút gọn link bằng Yeumoney"""
    if len(context.args) == 0:
        await update.message.reply_text(
            "❌ Vui lòng nhập link cần rút gọn.\n\nVí dụ: /rutgon https://example.com"
        )
        return

    long_url = context.args[0]
    try:
        response = requests.post(
            f"{YEUMONEY_API_URL}shorten",
            headers={"Authorization": YEUMONEY_TOKEN},
            data={"url": long_url}
        )
        data = response.json()

        if response.status_code == 200:
            short_url = data['data']['shortenedUrl']
            shortened_links.append(short_url)  # Lưu link vào bộ nhớ

            # Nút bấm sao chép
            buttons = [
                [InlineKeyboardButton("🔗 Sao chép link", url=short_url)]
            ]
            keyboard = InlineKeyboardMarkup(buttons)

            await update.message.reply_text(
                f"✅ Link rút gọn thành công:\n{short_url}",
                reply_markup=keyboard
            )
        else:
            error_message = data.get("message", "Không thể rút gọn link. Vui lòng kiểm tra link hoặc API Key!")
            await update.message.reply_text(f"❌ {error_message}")
    except Exception as e:
        await update.message.reply_text(f"❌ Lỗi: {e}")

async def listlinks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Hiển thị danh sách link đã rút gọn"""
    if not shortened_links:
        await update.message.reply_text("📭 Chưa có link nào được rút gọn!")
        return

    links_list = "\n".join([f"{idx + 1}. {link}" for idx, link in enumerate(shortened_links[-10:])])
    await update.message.reply_text(f"📋 Danh sách link đã rút gọn gần đây:\n\n{links_list}")

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý các nút bấm"""
    query = update.callback_query
    await query.answer()

    if query.data == "thongke":
        await thongke(update, context)
    elif query.data == "rutgon":
        await query.edit_message_text(
            "✂️ Vui lòng nhập link để rút gọn:\n\nVí dụ: /rutgon https://example.com"
        )

async def trogiup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gửi hướng dẫn sử dụng"""
    help_text = (
        "🛠 Hướng dẫn sử dụng bot Yeumoney:\n\n"
        "1️⃣ /thongke - Hiển thị thống kê tài khoản Yeumoney.\n"
        "2️⃣ /rutgon [URL] - Rút gọn link để kiếm tiền từ Yeumoney.\n"
        "3️⃣ /listlinks - Xem danh sách link đã rút gọn gần đây.\n\n"
        "💡 Lưu ý:\n"
        "- Đảm bảo API Key của bạn hợp lệ.\n"
        "- Link nhập vào phải đầy đủ, bao gồm https:// hoặc http://.\n\n"
        "📞 Hỗ trợ: Truy cập Yeumoney để được hỗ trợ thêm."
    )
    await update.message.reply_text(help_text)

def main():
    # Khởi tạo ứng dụng
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Thêm các lệnh vào bot
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("thongke", thongke))
    application.add_handler(CommandHandler("rutgon", rutgon))
    application.add_handler(CommandHandler("listlinks", listlinks))
    application.add_handler(CommandHandler("trogiup", trogiup))
    application.add_handler(CallbackQueryHandler(callback_handler))

    # Chạy bot
    application.run_polling()

if __name__ == "__main__":
    main()