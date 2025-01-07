import os
os.system("pip install scikit-learn")
import random
from collections import Counter, deque
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
import numpy as np

# Token bot Telegram
TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise ValueError("Vui lòng đặt biến môi trường TELEGRAM_TOKEN chứa token bot!")

# Bộ nhớ lịch sử
history_data = deque(maxlen=100)  # Lưu tối đa 100 kết quả Tài/Xỉu
dice_data = deque(maxlen=100)     # Lưu tối đa 100 kết quả súc sắc

# ==============================
# Các mô hình học máy
# ==============================

# Mô hình Naive Bayes
nb_model = GaussianNB()

# Mô hình Logistic Regression
lr_model = LogisticRegression()

# Mô hình LSTM (Long Short Term Memory)
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1, activation='sigmoid'))  # 1 output: Tài (1) hoặc Xỉu (0)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# ==============================
# Các hàm hỗ trợ
# ==============================

def detect_pattern(history):
    if len(history) < 4:
        return "Không đủ dữ liệu để phát hiện cầu."
    
    is_one_one = all(history[i] != history[i+1] for i in range(len(history) - 1))
    if is_one_one:
        return "Cầu 1-1: Tài, Xỉu xen kẽ."

    is_bet = all(history[i] == history[i+1] for i in range(len(history) - 1))
    if is_bet:
        return f"Cầu bệt: {history[0]} lặp lại."

    return "Không phát hiện cầu rõ ràng."

def weighted_prediction(history):
    if not history:
        return random.choice(["t", "x"]), 50.0, 50.0  # Nếu không có lịch sử, dự đoán ngẫu nhiên với tỷ lệ 50/50

    counter = Counter(history)
    prob_tai = counter["t"] / len(history) * 100
    prob_xiu = counter["x"] / len(history) * 100

    prediction = "t" if prob_tai > prob_xiu else "x"
    return prediction, prob_tai, prob_xiu

def train_models():
    # Huấn luyện mô hình Naive Bayes
    history_labels = [1 if result == "t" else 0 for result in history_data]  # Tài = 1, Xỉu = 0
    nb_model.fit(np.array(history_labels).reshape(-1, 1), history_labels)

    # Huấn luyện mô hình Logistic Regression
    X_lr = np.array([[i] for i in range(len(history_data))])  # Dữ liệu huấn luyện cho Logistic Regression
    lr_model.fit(X_lr, history_labels)

    # Huấn luyện mô hình LSTM (nếu có đủ dữ liệu)
    if len(history_data) > 10:
        X_lstm = np.array([[i] for i in range(len(history_data))])
        y_lstm = np.array([1 if result == "t" else 0 for result in history_data])
        X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))
        lstm_model = build_lstm_model((X_lstm.shape[1], 1))
        lstm_model.fit(X_lstm, y_lstm, epochs=5, batch_size=1)

# ==============================
# Các lệnh cho bot Telegram
# ==============================

# Lệnh /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Chào mừng bạn đến với bot dự đoán Tài/Xỉu!\n"
        "Sử dụng các lệnh sau để bắt đầu:\n"
        "- /tx <chuỗi lịch sử>: Dự đoán dựa trên lịch sử.\n"
        "- /txs <dãy số>: Dự đoán kết hợp từ lịch sử và súc sắc.\n"
        "- /add <lịch sử | súc sắc>: Thêm dữ liệu mới.\n"
        "- /history: Xem lịch sử.\n"
        "- /help: Hướng dẫn sử dụng.\n"
    )

# Lệnh /tx: Dự đoán dựa trên lịch sử
async def tx(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = ''.join(context.args)
        if not user_input:
            await update.message.reply_text("Vui lòng nhập chuỗi lịch sử (t: Tài, x: Xỉu)!")
            return

        history = list(user_input)
        if not all(item in ["t", "x"] for item in history):
            await update.message.reply_text("Dữ liệu chỉ được chứa 't' (Tài) hoặc 'x' (Xỉu).")
            return

        history_data.extend(history)
        prediction, prob_tai, prob_xiu = weighted_prediction(history)
        pattern = detect_pattern(list(history_data))

        buttons = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Đúng", callback_data=f"correct|{prediction}"), 
             InlineKeyboardButton("❌ Sai", callback_data=f"wrong|{prediction}")]
        ])

        await update.message.reply_text(
            f"Dự đoán: {'Tài' if prediction == 't' else 'Xỉu'}\n"
            f"Tỷ lệ phần trăm Tài: {prob_tai:.2f}%\n"
            f"Tỷ lệ phần trăm Xỉu: {prob_xiu:.2f}%\n"
            f"Phát hiện cầu: {pattern}",
            reply_markup=buttons
        )
    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")

# Lệnh /txs: Dự đoán kết hợp lịch sử và súc sắc
async def txs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = ''.join(context.args)
        if not user_input:
            await update.message.reply_text("Vui lòng nhập dãy số súc sắc!")
            return

        dice_values = list(map(int, user_input.split()))
        if not dice_values:
            await update.message.reply_text("Dữ liệu súc sắc không hợp lệ.")
            return

        dice_data.extend(dice_values)
        prediction, prob_tai, prob_xiu = weighted_prediction(list(history_data))
        pattern = detect_pattern(list(history_data))

        buttons = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Đúng", callback_data=f"correct|{prediction}"), 
             InlineKeyboardButton("❌ Sai", callback_data=f"wrong|{prediction}")]
        ])

        await update.message.reply_text(
            f"Dự đoán: {'Tài' if prediction == 't' else 'Xỉu'}\n"
            f"Tỷ lệ phần trăm Tài: {prob_tai:.2f}%\n"
            f"Tỷ lệ phần trăm Xỉu: {prob_xiu:.2f}%\n"
            f"Phát hiện cầu: {pattern}",
            reply_markup=buttons
        )
    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")

# Lệnh /add: Thêm dữ liệu
async def add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if len(context.args) < 2:
            await update.message.reply_text("Vui lòng cung cấp lịch sử và dãy súc sắc.")
            return

        history = context.args[0]
        dice_values = context.args[1]

        if not all(item in ["t", "x"] for item in history):
            await update.message.reply_text("Lịch sử chỉ chứa 't' (Tài) và 'x' (Xỉu).")
            return

        if len(dice_values) != 3:
            await update.message.reply_text("Súc sắc phải gồm 3 số.")
            return

        history_data.extend(history)
        dice_data.extend(dice_values)
        await update.message.reply_text(f"Dữ liệu đã được thêm thành công: {history} | {dice_values}.")
    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")

# Lệnh /history: Xem lịch sử
async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    history_str = ', '.join(history_data)
    dice_str = ', '.join(map(str, dice_data))
    await update.message.reply_text(
        f"Lịch sử Tài/Xỉu: {history_str}\nLịch sử Súc sắc: {dice_str}"
    )

# ==============================
# Xử lý callback cho nút Đúng/Sai
# ==============================

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data.split("|")
    action = data[0]
    prediction = data[1]

    if action == "correct":
        history_data.append(prediction)
        await query.edit_message_text("Cảm ơn! Kết quả đã được xác nhận và lưu lại.")
    elif action == "wrong":
        await query.edit_message_text("Cảm ơn! Kết quả sẽ không được lưu lại.")

# ==============================
# Chạy bot
# ==============================

if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()

    # Thêm các lệnh vào bot
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help))  # Thêm lệnh /help
    app.add_handler(CommandHandler("tx", tx))
    app.add_handler(CommandHandler("txs", txs))
    app.add_handler(CommandHandler("add", add))
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CallbackQueryHandler(handle_callback))

    print("Bot đang chạy...")
    app.run_polling()