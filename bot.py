import os
os.system("pip install scikit-learn")
os.system("pip install tensorflow")
os.system("pip install tensorflow-cpu")
import random
import threading
import time
from collections import Counter, deque
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint
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

#khởi tạo mô hình
nb_model = GaussianNB()
lr_model = LogisticRegression()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
scaler = MinMaxScaler(feature_range=(0, 1))

# Mô hình LSTM (Long Short Term Memory)
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

lstm_checkpoint = ModelCheckpoint(
    "lstm_best_model.keras",  # Thay đổi đuôi thành .keras
    monitor="loss",
    save_best_only=True,
    verbose=1
)

# ==============================
# Các hàm hỗ trợ
# ==============================

def detect_pattern(history):
    if len(history) < 4:
        return "Không đủ dữ liệu để phát hiện cầu."

    # Phát hiện cầu 1-1 (Tài, Xỉu xen kẽ)
    is_one_one = all(history[i] != history[i + 1] for i in range(len(history) - 1))
    if is_one_one:
        return "Cầu 1-1: Tài, Xỉu xen kẽ."

    # Phát hiện cầu bệt (chuỗi lặp lại cùng loại)
    is_bet = all(history[i] == history[i + 1] for i in range(len(history) - 1))
    if is_bet:
        return f"Cầu bệt: {history[0]} lặp lại."

    return "Không phát hiện cầu rõ ràng."
    
def weighted_prediction(history):
    if not history:
        return random.choice(["t", "x"]), 50.0, 50.0

    weights = [0.8**i for i in range(len(history))]
    counter = {"t": 0, "x": 0}

    for i, result in enumerate(history):
        counter[result] += weights[i]

    total_weight = sum(weights)
    prob_tai = (counter["t"] / total_weight) * 100
    prob_xiu = (counter["x"] / total_weight) * 100

    prediction = "t" if prob_tai > prob_xiu else "x"
    return prediction, prob_tai, prob_xiu

def optimize_hyperparameters(history_data, dice_data, labels):
    # Kết hợp dữ liệu lịch sử và xúc xắc thành một tập hợp
    X_combined = np.array([history_data + dice_data])
    y_combined = np.array(labels)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # Tối ưu Random Forest
    rf_params = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]}
    grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='accuracy')
    grid_rf.fit(X_scaled, y_combined)
    print("Best Random Forest Params:", grid_rf.best_params_)

    # Tối ưu Logistic Regression
    lr_params = {'C': [0.01, 0.1, 1, 10]}
    grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), lr_params, cv=3, scoring='accuracy')
    grid_lr.fit(X_scaled, y_combined)
    print("Best Logistic Regression Params:", grid_lr.best_params_)

    return grid_rf.best_params_, grid_lr.best_params_
    
def train_models():
    try:
        # Chuyển dữ liệu Tài/Xỉu thành nhãn
        history_labels = [1 if result == "t" else 0 for result in history_data]

        # Tạo dữ liệu đặc trưng: chỉ số + tổng giá trị + chẵn/lẻ
        X_features = []
        for i in range(len(history_data)):
            total = sum(dice_data[max(0, i - 3):i + 1])  # Tổng của 4 giá trị gần nhất
            even_odd = total % 2  # Chẵn/lẻ
            X_features.append([i, total, even_odd])

        X_features = np.array(X_features)

        # Chuẩn hóa dữ liệu
        X_scaled = scaler.fit_transform(X_features)

        # Huấn luyện Naive Bayes
        nb_model.fit(X_scaled, history_labels)

        # Huấn luyện Logistic Regression
        lr_model.fit(X_scaled, history_labels)

        # Huấn luyện Random Forest
        rf_model.fit(X_scaled, history_labels)

        # Huấn luyện LSTM nếu đủ dữ liệu
        if len(history_data) > 10:
            X_lstm = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
            lstm_model = build_lstm_model((X_lstm.shape[1], 1))
            lstm_model.fit(X_lstm, history_labels, epochs=10, batch_size=1, verbose=0)

        print("Huấn luyện mô hình thành công!")
    except Exception as e:
        print(f"Lỗi khi huấn luyện mô hình: {e}")

# Hàm dự đoán từ nhiều mô hình
def predict_combined(dice_values, history):
    try:
        # Kiểm tra dữ liệu đầu vào
        validate_input_data(dice_values)

        # Tính tổng, chẵn/lẻ từ xúc xắc
        total = sum(dice_values)
        even_odd = total % 2  # 0: chẵn, 1: lẻ

        # Thống kê từ lịch sử
        count_tai = history.count("t")
        count_xiu = history.count("x")
        total_history = len(history)
        ratio_tai = count_tai / total_history if total_history > 0 else 0
        ratio_xiu = count_xiu / total_history if total_history > 0 else 0

        # Dữ liệu đầu vào cho mô hình
        input_features = np.array([[total, even_odd, ratio_tai, ratio_xiu]])
        input_scaled = scaler.transform(input_features)  # Chuẩn hóa dữ liệu

        # Dự đoán bằng voting model
        prob_voting = voting_model.predict_proba(input_scaled)[:, 1][0]
        prediction = "t" if prob_voting > 0.5 else "x"

        # Tính xác suất cho tài/xỉu
        prob_tai = prob_voting * 100  # Xác suất tài
        prob_xiu = (1 - prob_voting) * 100  # Xác suất xỉu

        return prediction, prob_tai, prob_xiu
    except Exception as e:
        log_error(e)  # Ghi lại lỗi để debug
        return None, 0, 0  # Trả về dự đoán mặc định
# ==============================
# Các lệnh cho bot Telegram
# ==============================

# Hàm huấn luyện mô hình dưới nền
def background_training():
    while True:
        try:
            # Tiến hành huấn luyện mô hình nếu có đủ dữ liệu
            if len(history_data) > 10:
                train_models()  # Huấn luyện mô hình với dữ liệu hiện tại
            time.sleep(60)  # Chạy lại mỗi 60 giây để huấn luyện dưới nền
        except Exception as e:
            print(f"Lỗi khi huấn luyện dưới nền: {e}")
            time.sleep(60)  # Đợi trước khi thử lại
            
def start_background_training():
    training_thread = threading.Thread(target=background_training, daemon=True)
    training_thread.start()

# Lệnh /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    start_background_training()  # Khởi động huấn luyện nền
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
        # Nhận chuỗi lịch sử từ người dùng
        user_history = ''.join(context.args).strip()
        if not user_history:
            await update.message.reply_text("Vui lòng nhập chuỗi lịch sử (t: Tài, x: Xỉu)!")
            return

        # Xác thực chuỗi lịch sử
        history = list(user_history)
        if not all(item in ["t", "x"] for item in history):
            await update.message.reply_text("Dữ liệu lịch sử chỉ được chứa 't' (Tài) hoặc 'x' (Xỉu).")
            return

        # Nhận dãy số xúc xắc từ người dùng
        dice_values = context.args[1:]  # Lấy các tham số sau chuỗi lịch sử
        dice_values = list(map(int, dice_values)) if dice_values else []

        # Xác thực dữ liệu xúc xắc
        if not all(1 <= value <= 6 for value in dice_values):
            await update.message.reply_text("Dữ liệu súc sắc phải là các số từ 1 đến 6.")
            return

        # Cập nhật dữ liệu toàn cục
        history_data.extend(history)
        dice_data.extend(dice_values)

        # Tính toán dự đoán dựa trên dữ liệu
        prediction, prob_tai, prob_xiu = weighted_prediction(list(history_data), dice_data)
        pattern = detect_pattern(list(history_data))

        # Gửi dự đoán kèm nút xác nhận
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

# Lệnh /txs: Lệnh riêng xử lý súc sắc (nếu cần)
async def txs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Nhận dãy số xúc xắc
        user_input = ''.join(context.args).strip()
        if not user_input:
            await update.message.reply_text("Vui lòng nhập dãy số súc sắc!")
            return

        dice_values = list(map(int, user_input.split()))
        if not all(1 <= value <= 6 for value in dice_values):
            await update.message.reply_text("Dữ liệu súc sắc phải là các số từ 1 đến 6.")
            return

        # Cập nhật dữ liệu toàn cục
        dice_data.extend(dice_values)

        # Dự đoán dựa trên dữ liệu hiện có
        prediction, prob_tai, prob_xiu = weighted_prediction(list(history_data), dice_data)
        pattern = detect_pattern(list(history_data))

        # Gửi dự đoán kèm nút xác nhận
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
        
async def add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = ' '.join(context.args)
        if not user_input:
            await update.message.reply_text("Vui lòng nhập dữ liệu dạng: 't x t | 15 10 9'.")
            return

        # Tách lịch sử và súc sắc
        parts = user_input.split("|")
        if len(parts) != 2:
            await update.message.reply_text("Dữ liệu không hợp lệ! Nhập dạng 't x t | 15 10 9'.")
            return

        # Xử lý lịch sử
        history = parts[0].strip().split()
        if not all(item in ["t", "x"] for item in history):
            await update.message.reply_text("Lịch sử chỉ được chứa 't' (Tài) hoặc 'x' (Xỉu).")
            return

        # Xử lý dữ liệu súc sắc
        try:
            dice_values = list(map(int, parts[1].strip().split()))
        except ValueError:
            await update.message.reply_text("Dữ liệu súc sắc phải là số nguyên, cách nhau bởi dấu cách.")
            return

        # Thêm vào bộ nhớ
        history_data.extend(history)
        dice_data.extend(dice_values)

        await update.message.reply_text("Dữ liệu đã được thêm thành công!")
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

async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hướng dẫn sử dụng bot:\n"
        "1. **/tx <chuỗi lịch sử>**: Dự đoán kết quả dựa trên lịch sử Tài/Xỉu.\n"
        "   Ví dụ: /tx t x t x\n"
        "2. **/txs <dãy số súc sắc>**: Dự đoán kết hợp lịch sử và dãy số súc sắc.\n"
        "3. **/add <lịch sử hoặc súc sắc>**: Thêm dữ liệu vào lịch sử.\n"
        "4. **/history**: Xem lịch sử Tài/Xỉu và súc sắc.\n"
    )

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
