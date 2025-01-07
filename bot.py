import os
os.system("pip install scikit-learn")
import random
import numpy as np
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from collections import Counter, deque
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Lấy token từ biến môi trường
TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise ValueError("Vui lòng đặt biến môi trường TELEGRAM_TOKEN chứa token bot!")

# Bộ nhớ lịch sử thực tế (cập nhật mỗi khi có trận đấu mới)
history_data = deque(maxlen=100)  # Lưu trữ tối đa 100 kết quả
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

def ml_prediction(history, dice_sum=None):
    """
    Dự đoán bằng Machine Learning kết hợp lịch sử và dữ liệu súc sắc.
    """
    if len(train_data) < 10:
        return weighted_recent_analysis(history)  # Quay về dự đoán gần nhất nếu thiếu dữ liệu

    # Chuyển đổi lịch sử thành dạng vector (0: Tài, 1: Xỉu)
    encoded_history = le.transform(history[-5:])  # Sử dụng 5 phần tử gần nhất
    features = np.array([encoded_history])

    if dice_sum is not None:
        features = np.concatenate((features, np.array([[dice_sum]])), axis=1)

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

def weighted_recent_analysis(history):
    """
    Phân tích trọng số ưu tiên kết quả gần nhất.
    """
    weights = np.arange(1, len(history) + 1)
    counter = Counter(history)
    prob_tai = counter["t"] * weights[-1] / sum(weights)
    prob_xiu = counter["x"] * weights[-1] / sum(weights)
    return "t" if prob_tai > prob_xiu else "x"

def longest_sequence_analysis(history):
    """
    Tìm chuỗi dài nhất của một ký tự trong lịch sử.
    """
    current = max_streak = history[0]
    current_count = max_count = 1

    for i in range(1, len(history)):
        if history[i] == current:
            current_count += 1
        else:
            current = history[i]
            current_count = 1

        if current_count > max_count:
            max_streak = current
            max_count = current_count

    return max_streak

def combined_prediction(history, dice_sum=None):
    """
    Kết hợp các phương pháp dự đoán.
    """
    # Phân tích chuỗi liên tiếp
    streak_prediction = analyze_real_data(history)
    if streak_prediction:
        return streak_prediction

    # Dự đoán chuỗi dài nhất
    sequence_prediction = longest_sequence_analysis(history)

    # Dự đoán bằng Machine Learning kết hợp lịch sử và tổng điểm súc sắc
    ml_result = ml_prediction(history, dice_sum)

    # Kết hợp dự đoán
    return random.choice([sequence_prediction, ml_result])

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

        # Thêm vào dữ liệu huấn luyện
        if len(history) >= 5:  # Chỉ thêm khi có đủ dữ liệu
            train_data.append(le.fit_transform(history[-5:]))
            train_labels.append(history[-1])
            train_model()

        # Dự đoán kết quả
        result = combined_prediction(list(history_data))
        await update.message.reply_text(f"Kết quả dự đoán của tôi: {'Tài' if result == 't' else 'Xỉu'}")

    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")

# Lệnh /txs
async def txs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Lấy dãy số súc sắc từ người dùng
        dice_input = ''.join(context.args)

        if not dice_input:
            await update.message.reply_text("Vui lòng nhập dãy số súc sắc (ví dụ: 12 6 8 14)! ")
            return

        # Chuyển dãy súc sắc thành danh sách các số
        dice_numbers = list(map(int, dice_input.split()))

        # Tính tổng điểm của các viên súc sắc
        dice_sum = sum(dice_numbers)

        # Cập nhật lịch sử thực tế vào bộ nhớ
        result = combined_prediction(list(history_data), dice_sum)
        await update.message.reply_text(f"Kết quả dự đoán của tôi (kết hợp dữ liệu súc sắc): {'Tài' if result == 't' else 'Xỉu'}")

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
        await update.message.reply_text(f"Lịch sử gần nhất: {''.join(history_data)}")

# Lệnh /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hướng dẫn sử dụng bot:\n"
        "/tx <dãy lịch sử (t/x)>: Dự đoán kết quả dựa trên lịch sử Tài/Xỉu.\n"
        "/txs <dãy súc sắc (12 6 8 14)>: Dự đoán kết quả dựa trên tổng điểm súc sắc.\n"
        "/history: Xem lịch sử kết quả.\n"
        "/add <dãy kết quả thực tế>: Cập nhật dữ liệu thực tế mới."
    )

def main():
    application = ApplicationBuilder().token(TOKEN).build()

    # Các lệnh trong bot
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("tx", tx))
    application.add_handler(CommandHandler("txs", txs))
    application.add_handler(CommandHandler("add", add))
    application.add_handler(CommandHandler("history", history))
    application.add_handler(CommandHandler("help", help_command))

    # Chạy bot
    application.run_polling()

if __name__ == "__main__":
    main()