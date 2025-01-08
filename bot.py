import os
import random
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
from collections import Counter, deque
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Lấy token từ biến môi trường
TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise ValueError("Vui lòng đặt biến môi trường TELEGRAM_TOKEN chứa token bot!")

# Bộ nhớ lịch sử thực tế (cập nhật mỗi khi có trận đấu mới)
history_data = deque(maxlen=400)  # Lưu trữ tối đa 100 kết quả
train_data = []  # Lịch sử để huấn luyện
train_labels = []
le = LabelEncoder()
model = LogisticRegression()

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
        "Chào mừng bạn đến với bot dự đoán Tài Xỉu thực tế! 😎\n"
        "Sử dụng lệnh /tx để nhận dự đoán và thử xem khả năng dự đoán của tôi có đáng tin không. 🤔\n"
        "Nếu kết quả sai, đừng ngại bóp méo mặt tôi bằng câu 'Sai lầm lớn rồi!' 😂\n"
        "Nhập /help để biết thêm thông tin chi tiết."
    )

# Lệnh /tx
async def tx(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Lấy dãy số từ người dùng
        user_input = ' '.join(context.args)

        if not user_input:
            await update.message.reply_text("Vui lòng nhập dãy lịch sử (t: Tài, x: Xỉu) đi bạn ơi! 😅")
            return

        # Chuyển đổi lịch sử thành danh sách
        history = user_input.split()

        # Kiểm tra định dạng hợp lệ (chỉ chấp nhận "t" hoặc "x")
        if not all(item in ["t", "x"] for item in history):
            await update.message.reply_text("Dãy lịch sử chỉ được chứa 't' (Tài) và 'x' (Xỉu), không có lúa nhé! 😜")
            return

        # Cập nhật lịch sử thực tế vào bộ nhớ
        history_data.extend(history)

        # Thêm vào dữ liệu huấn luyện
        if len(history) >= 5:  # Chỉ thêm khi có đủ dữ liệu
            train_data.append(le.fit_transform(history[-5:]))
            train_labels.append(history[-1])
            train_model()

        # Dự đoán kết quả
        result = combined_prediction(list(history_data))

        # Chuyển đổi kết quả dự đoán thành biểu tượng
        if result == "t":
            result_text = "Tài 🟠\nBật đèn xanh cho bạn, vô luôn! 💸"
        else:
            result_text = "Xỉu ⚪\nChắc chắn rồi, lần này không thể sai được! 💀"

        # Hiển thị nút "Đúng" và "Sai"
        buttons = [
            [
                InlineKeyboardButton("Đúng", callback_data="correct"),
                InlineKeyboardButton("Sai", callback_data="incorrect"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(buttons)
        await update.message.reply_text(result_text, reply_markup=reply_markup)

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

# Xử lý nút "Đúng" và "Sai"
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "correct":
        # Nếu người dùng chọn "Đúng", lưu dữ liệu để huấn luyện mô hình
        result_text = "Chúc mừng! Bạn đã chọn đúng! 🎉\nĐã lưu dữ liệu vào hệ thống để tôi càng ngày càng thông minh hơn! 🤖"
        history_data.append("t" if query.message.text.startswith("Tài") else "x")
        train_data.append(le.fit_transform(history_data[-5:]))
        train_labels.append("t" if query.message.text.startswith("Tài") else "x")
        train_model()
    else:
        # Nếu người dùng chọn "Sai", thông báo
        result_text = "Hây, bạn đã chọn sai rồi! 😜\nĐể tôi cải thiện lần sau! 😎"

    await query.edit_message_text(text=result_text)

# Lệnh /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hướng dẫn sử dụng bot:\n"
        "/tx [dãy lịch sử]: Dự đoán kết quả Tài/Xỉu.\n"
        "/add [dữ liệu thực tế]: Cập nhật dữ liệu thực tế.\n"
        "/history: Xem lịch sử gần nhất.\n"
        "/help: Hướng dẫn sử dụng bot."
    )

def main():
    application = ApplicationBuilder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("tx", tx))
    application.add_handler(CommandHandler("add", add))
    application.add_handler(CommandHandler("history", history))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(button_handler))

    application.run_polling()

if __name__ == "__main__":
    main()