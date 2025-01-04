import os
import random
import numpy as np
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackContext

# Lấy token từ biến môi trường
TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise ValueError("Vui lòng đặt biến môi trường TELEGRAM_TOKEN chứa token bot!")

# Hàm dự đoán tài xỉu
def predict_tai_xiu(history):
    """Dự đoán kết quả tài xỉu dựa trên dãy số lịch sử."""
    
    # Kiểm tra nếu dãy lịch sử không đủ dữ liệu
    if len(history) < 5:
        return "Lịch sử quá ngắn, không đủ dữ liệu để dự đoán!"
    
    # Chuyển dãy lịch sử thành mảng numpy
    history_array = np.array(history)

    # Tính toán các chỉ số thống kê
    count_tai = np.sum(history_array == 1)  # Tính số lần "Tài"
    count_xiu = np.sum(history_array == 0)  # Tính số lần "Xỉu"

    # Dự đoán dựa trên tần suất
    if count_tai > count_xiu:
        prediction = "Tài"
    elif count_xiu > count_tai:
        prediction = "Xỉu"
    else:
        prediction = random.choice(["Tài", "Xỉu"])

    return prediction

# Lệnh /start
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "Chào mừng bạn đến với bot dự đoán tài xỉu! Dùng /taixiu để nhận dự đoán."
    )

# Lệnh /taixiu
async def taixiu(update: Update, context: CallbackContext):
    try:
        # Lấy dãy số từ người dùng
        user_input = ' '.join(context.args)
        
        if not user_input:
            await update.message.reply_text("Vui lòng nhập dãy số lịch sử tài xỉu!")
            return

        # Chuyển dãy số thành danh sách (1 = Tài, 0 = Xỉu)
        history = list(map(int, user_input.split()))

        # Dự đoán tài xỉu từ dãy số lịch sử
        result = predict_tai_xiu(history)
        await update.message.reply_text(f"Dự đoán của tôi: {result}")
    
    except ValueError:
        await update.message.reply_text("Lỗi: Dãy số chỉ bao gồm 1 (Tài) và 0 (Xỉu).")

# Lệnh /help
async def help_command(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "Dùng /taixiu theo cú pháp sau: /taixiu 1 0 1 1 0\n" 
        "Trong đó 1 = Tài, 0 = Xỉu. Bot sẽ dự đoán kết quả tiếp theo dựa trên dãy số lịch sử."
    )

# Khởi chạy bot
if __name__ == "__main__":
    print("Bot đang chạy...")

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("taixiu", taixiu))
    app.add_handler(CommandHandler("help", help_command))

    app.run_polling()
