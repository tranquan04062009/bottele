import os
os.system("pip install scikit-learn")
import random
import numpy as np
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from collections import Counter, deque

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

def total_dice_prediction(dice_values):
    """
    Dự đoán kết quả dựa trên tổng điểm của các viên súc sắc.
    """
    total_points = sum(dice_values)

    # Tính xác suất theo tổng điểm của các viên súc sắc
    if total_points % 2 == 0:
        return "x"  # Xỉu (Tổng điểm chẵn)
    else:
        return "t"  # Tài (Tổng điểm lẻ)

def combine_predictions(history, dice_values):
    """
    Kết hợp dự đoán từ lịch sử và tổng điểm súc sắc sử dụng trọng số toán học.
    """
    # Phân tích kết quả dựa trên lịch sử
    rule_prediction = analyze_real_data(history)
    if rule_prediction:
        return rule_prediction

    # Dự đoán dựa trên trọng số từ lịch sử
    history_prediction = weighted_prediction(history)

    # Dự đoán dựa trên tổng điểm súc sắc
    dice_prediction = total_dice_prediction(dice_values)

    # Kết hợp xác suất dự đoán từ cả hai nguồn dữ liệu
    history_prob = Counter(history)["t"] / len(history) if len(history) > 0 else 0.5
    dice_prob = 0.5  # Giả sử xác suất cho tổng điểm chẵn và lẻ là tương đương

    # Sử dụng trọng số để kết hợp các dự đoán
    if random.random() < (history_prob + dice_prob) / 2:
        return history_prediction
    else:
        return dice_prediction

# Lệnh /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Chào mừng bạn đến với bot dự đoán Tài/Xỉu!\n"
        "Sử dụng lệnh /tx để nhận dự đoán từ lịch sử Tài/Xỉu.\n"
        "Sử dụng lệnh /txs để nhận dự đoán kết hợp từ tổng điểm súc sắc và lịch sử."
    )

# Lệnh /tx
async def tx(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Lấy dãy số từ người dùng
        user_input = ''.join(context.args)

        if not user_input:
            await update.message.reply_text("Vui lòng nhập dãy lịch sử (t: Tài, x: Xỉu)!")
            return

        # Chuyển đổi lịch sử thành danh sách
        history = list(user_input)

        # Kiểm tra định dạng hợp lệ (chỉ chấp nhận "t" hoặc "x")
        if not all(item in ["t", "x"] for item in history):
            await update.message.reply_text("Dãy lịch sử chỉ được chứa 't' (Tài) và 'x' (Xỉu).")
            return

        # Cập nhật lịch sử thực tế vào bộ nhớ
        history_data.extend(history)

        # Dự đoán kết quả
        result = weighted_prediction(list(history_data))
        await update.message.reply_text(f"Kết quả dự đoán của tôi: {'Tài' if result == 't' else 'Xỉu'}")

    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")

# Lệnh /txs
async def txs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Lấy dãy số súc sắc từ người dùng
        user_input = ''.join(context.args)

        if not user_input:
            await update.message.reply_text("Vui lòng nhập dãy số súc sắc (ví dụ: 12 6 8 14)!")
            return

        # Chuyển đổi dãy số súc sắc thành danh sách
        dice_values = list(map(int, user_input.split()))

        # Kiểm tra định dạng hợp lệ (chỉ chấp nhận các số nguyên)
        if not all(isinstance(i, int) for i in dice_values):
            await update.message.reply_text("Dãy súc sắc chỉ được chứa các số nguyên.")
            return

        # Cập nhật lịch sử thực tế vào bộ nhớ
        history_data.extend(dice_values)

        # Dự đoán kết hợp từ lịch sử và tổng điểm súc sắc
        result = combine_predictions(list(history_data), dice_values)
        await update.message.reply_text(f"Kết quả dự đoán của tôi: {'Tài' if result == 't' else 'Xỉu'}")

    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")

# Lệnh /add (cập nhật dữ liệu thực tế)
async def add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = ''.join(context.args)

        if not user_input:
            await update.message.reply_text("Vui lòng nhập kết quả thực tế (t: Tài, x: Xỉu)!")
            return

        # Chuyển đổi lịch sử thành danh sách
        new_data = list(user_input)

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
        await update.message.reply_text(f"Lịch sử gần nhất: {' '.join(map(str, history_data))}")

# Lệnh /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hướng dẫn sử dụng bot:\n"
        "/tx <dãy lịch sử (t/x)>: Dự đoán kết quả dựa trên lịch sử Tài/Xỉu.\n"
        "/txs <dãy súc sắc (12 6 8 14)>: Dự đoán kết quả dựa trên tổng điểm súc sắc.\n"
        "/history: Xem lịch sử kết quả.\n"
        "/add <dãy kết quả thực tế>: Cập nhật dữ liệu thực tế mới."
    )

# Khởi chạy bot
if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()

    # Các lệnh trong bot
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("tx", tx))
    app.add_handler(CommandHandler("txs", txs))
    app.add_handler(CommandHandler("add", add))
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CommandHandler("help", help_command))

    app.run_polling()