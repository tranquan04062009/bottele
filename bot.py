import os
os.system("pip install scikit-learn")
import random
import numpy as np
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from collections import Counter, deque
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from telegram.ext import CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,  # Thêm dòng này
    ContextTypes,
)

# Lấy token từ biến môi trường
TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise ValueError("Vui lòng đặt biến môi trường TELEGRAM_TOKEN chứa token bot!")

# Bộ nhớ lịch sử thực tế (cập nhật mỗi khi có trận đấu mới)
history_data = deque(maxlen=400)  # Lưu trữ tối đa 100 kết quả
train_data = []  # Lịch sử để huấn luyện
train_labels = []
le = LabelEncoder()

last_prediction = None

model = RandomForestClassifier(n_estimators=100, random_state=42)

def train_model():
    """
    Huấn luyện mô hình Machine Learning dựa trên lịch sử thực tế.
    """
    if len(train_data) >= 10:  # Chỉ huấn luyện nếu có đủ dữ liệu
        X = np.array(train_data)
        y = le.fit_transform(train_labels)
        model.fit(X, y)

def ml_prediction(history):
    """
    Dự đoán bằng Machine Learning.
    """
    if len(train_data) < 10:
        return weighted_prediction(history)  # Quay về dự đoán trọng số nếu thiếu dữ liệu

    # Chuyển đổi lịch sử thành dạng vector (0: Tài, 1: Xỉu)
    encoded_history = le.transform(history[-5:])  # Sử dụng 5 phần tử gần nhất
    features = np.array([encoded_history])
    prediction = model.predict(features)
    return le.inverse_transform(prediction)[0]

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

def combined_prediction(history):
    """
    Kết hợp các phương pháp dự đoán.
    """
    # Phân tích chuỗi liên tiếp
    streak_prediction = analyze_real_data(history)
    if streak_prediction:
        return streak_prediction

    # Dự đoán bằng Machine Learning
    return ml_prediction(history)

# Lệnh /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Chào mừng bạn đến với bot dự đoán Tài Xỉu thực tế!\n"
        "Sử dụng lệnh /tx để nhận dự đoán.\n"
        "Nhập /help để biết thêm thông tin chi tiết."
    )

# Lệnh /tx
async def tx(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global last_prediction
    try:
        user_input = ''.join(context.args)

        if not user_input:
            await update.message.reply_text("Vui lòng nhập dãy lịch sử (t: Tài, x: Xỉu)!")
            return

        # Chuyển đổi lịch sử thành danh sách
        history = user_input.split()

        # Kiểm tra định dạng hợp lệ (chỉ chấp nhận "t" hoặc "x")
        if not all(item in ["t", "x"] for item in user_input):
            await update.message.reply_text("Dãy lịch sử chỉ được chứa 't' (Tài) và 'x' (Xỉu).")
            return

        # Cập nhật lịch sử thực tế vào bộ nhớ
        history_data.extend(history)

        # Tính toán xác suất
        count_t = history.count("t")
        count_x = history.count("x")
        total = len(history)

        probability_t = (count_t / total) * 100 if total > 0 else 0
        probability_x = (count_x / total) * 100 if total > 0 else 0

        # Dự đoán kết quả
        last_prediction = "t" if probability_t > probability_x else "x"
        result_text = (
            f"Kết quả dự đoán của tôi: {'Tài' if last_prediction == 't' else 'Xỉu'}\n"
            f"Tỉ lệ xác suất: Tài {probability_t:.2f}%, Xỉu {probability_x:.2f}%"
        )

        # Hiển thị nút đúng/sai
        buttons = [
            [
                InlineKeyboardButton("Đúng", callback_data="correct"),
                InlineKeyboardButton("Sai", callback_data="wrong"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(buttons)
        await update.message.reply_text(result_text, reply_markup=reply_markup)

    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")
        
# Xử lý callback từ nút đúng/sai
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global last_prediction
    query = update.callback_query
    await query.answer()  # Trả lời callback để tránh lỗi "callback query expired"

    if query.data == "correct":
        # Nếu đúng, thêm kết quả vào dữ liệu huấn luyện
        history_data.append(last_prediction)
        train_data.append(le.fit_transform(list(history_data)[-5:]))
        train_labels.append(last_prediction)
        train_model()
        await query.edit_message_text("Kết quả đã được ghi nhận là ĐÚNG!")
    elif query.data == "wrong":
        # Nếu sai, không thêm gì nhưng ghi nhận thông báo
        await query.edit_message_text("Kết quả đã được ghi nhận là SAI.")

# Lệnh /add (cập nhật dữ liệu thực tế)
async def add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = ''.join(context.args)

        if not user_input:
            await update.message.reply_text("Vui lòng nhập kết quả thực tế (t: Tài, x: Xỉu)!")
            return

        # Chuyển đổi lịch sử thành danh sách
        new_data = user_input.split()

        # Kiểm tra định dạng hợp lệ
        if not all(item in ["t", "x"] for item in user_input):
            await update.message.reply_text("Kết quả chỉ được chứa 't' (Tài) và 'x' (Xỉu).")
            return

        # Cập nhật dữ liệu mới
        history_data.extend(new_data)

        # Thêm vào dữ liệu huấn luyện
        for i in range(len(new_data) - 5 + 1):  # Huấn luyện với từng tập dữ liệu
            train_data.append(le.fit_transform(new_data[i:i + 5]))
            train_labels.append(new_data[i + 4])
            train_model()

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
if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("tx", tx))
    app.add_handler(CommandHandler("add", add))
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(button_handler))
    print("Bot đang chạy...")
    app.run_polling()