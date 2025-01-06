import os
import random
import numpy as np
from flask import Flask
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from collections import Counter, deque
import threading

# Lấy token từ biến môi trường
TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise ValueError("Vui lòng đặt biến môi trường TELEGRAM_TOKEN chứa token bot!")

# Bộ nhớ lịch sử thực tế (cập nhật mỗi khi có trận đấu mới)
history_data = deque(maxlen=100)  # Lưu trữ tối đa 100 kết quả

def analyze_real_data(history):
    """
    Phân tích dữ liệu thực tế để phát hiện xu hướng phổ biến (cầu bệt, cầu 1-1).
    """
    if len(history) < 3:
        return None

    # Phân tích cầu bệt
    if all(item == history[0] for item in history):
        return history[0]

    # Phân tích cầu 1-1
    if all(history[i] != history[i + 1] for i in range(len(history) - 1)):
        return "t" if history[-1] == "x" else "x"

    return None

def weighted_prediction(history):
    """
    Dự đoán dựa trên phân phối tần suất thực tế.
    """
    if not history:
        return random.choice(["t", "x"])

    # Tính tần suất xuất hiện của mỗi kết quả
    counter = Counter(history)
    total = len(history)

    prob_tai = counter["t"] / total
    prob_xiu = counter["x"] / total

    # Dự đoán dựa trên trọng số
    return "t" if random.random() < prob_tai else "x"

# Tích hợp phân tích dữ liệu thực tế
def real_data_prediction(history):
    """
    Kết hợp phân tích thực tế và dự đoán trọng số.
    """
    # Phân tích quy luật thực tế
    rule_prediction = analyze_real_data(history)
    if rule_prediction:
        return rule_prediction

    # Dự đoán theo trọng số
    return weighted_prediction(history)

# Lệnh /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Chào mừng bạn đến với bot dự đoán Tài Xỉu thực tế!\n"
        "Sử dụng lệnh /tx để nhận dự đoán.\n"
        "Nhập /help để biết thêm thông tin chi tiết."
    )

# Lệnh /tx
async def tx(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Lấy dãy số từ người dùng
        user_input = ' '.join(context.args)

        if not user_input:
            await update.message.reply_text("Vui lòng nhập dãy lịch sử (t: Tài, x: Xỉu)!")
            return

        # Chuyển đổi lịch sử thành danh sách
        history = user_input.split()

        # Kiểm tra định dạng hợp lệ (chỉ chấp nhận "t" hoặc "x")
        if not all(item in ["t", "x"] for item in history):
            await update.message.reply_text("Dãy lịch sử chỉ được chứa 't' (Tài) và 'x' (Xỉu).")
            return

        # Cập nhật lịch sử thực tế vào bộ nhớ
        history_data.extend(history)

        # Dự đoán kết quả
        result = real_data_prediction(list(history_data))
        await update.message.reply_text(f"Kết quả dự đoán của tôi: {'Tài' if result == 't' else 'Xỉu'}")

    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")

# Lệnh /add (cập nhật dữ liệu thực tế)
async def add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = ' '.join(context.args)

        if not user_input:
            await update.message.reply_text("Vui lòng nhập kết quả thực tế (t: Tài, x: Xỉu)!")
            return

        # Chuyển đổi lịch sử thành danh sách
        new_data = user_input.split()

        # Kiểm tra định dạng hợp lệ
        if not all(item in ["t", "x"] for item in new_data):
            await update.message.reply_text("Kết quả chỉ được chứa 't' (Tài) và 'x' (Xỉu).")
            return

        # Cập nhật dữ liệu mới
        history_data.extend(new_data)
        await update.message.reply_text(f"Dữ liệu thực tế đã được cập nhật: {new_data}")

    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")

# Lệnh /history (xem lịch sử)
async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not history_data:
        await update.message.reply_text("Hiện tại chưa có dữ liệu lịch sử.")
    else:
        await update.message.reply_text(f"Lịch sử gần nhất: {' '.join(history_data)}")

# Lệnh /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hướng dẫn sử dụng bot:\n"
        "/tx [dãy lịch sử]: Dự đoán kết quả Tài/Xỉu.\n"
        "/add [kết quả]: Cập nhật kết quả thực tế.\n"
        "/history: Xem lịch sử gần đây.\n"
        "Ví dụ:\n"
        "- /tx t t x t x\n"
        "- /add t x x t t"
    )

# Khởi chạy bot
def run_bot():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("tx", tx))
    app.add_handler(CommandHandler("add", add))
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CommandHandler("help", help_command))

    print("Bot đang chạy...")
    app.run_polling()

# Web Server để duy trì bot hoạt động
app_flask = Flask("")

@app_flask.route("/")
def home():
    return "Bot đang hoạt động!"

def run_server():
    app_flask.run(host="0.0.0.0", port=8080)

# Chạy cả bot và web server
if __name__ == "__main__":
    threading.Thread(target=run_server).start()
    run_bot()