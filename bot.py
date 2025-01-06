import os
import os
os.system("pip install scipy scikit-learn pandas")
import random
import numpy as np
import hashlib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from collections import deque
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Lấy token từ biến môi trường
TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise ValueError("Vui lòng đặt biến môi trường TELEGRAM_TOKEN chứa token bot!")

# Bộ nhớ lịch sử
history_data = deque(maxlen=100)
real_data = []

# Các mô hình học máy
label_encoder = LabelEncoder()
label_encoder.fit(["Tài", "Xỉu"])

logistic_model = LogisticRegression()
decision_tree_model = DecisionTreeClassifier()
random_forest_model = RandomForestClassifier(n_estimators=100)
svm_model = SVC(probability=True)
ann_model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500)

# Từ điển mẫu MD5 (dành cho giải mã nhanh)
tx_dictionary = {
    hashlib.md5("Tài".encode()).hexdigest(): "Tài",
    hashlib.md5("Xỉu".encode()).hexdigest(): "Xỉu",
}

# Cập nhật mô hình khi có dữ liệu mới
def update_models():
    if len(real_data) > 1:
        X = np.array([item[0] for item in real_data]).reshape(-1, 1)
        y = np.array([item[1] for item in real_data])

        logistic_model.fit(X, y)
        decision_tree_model.fit(X, y)
        random_forest_model.fit(X, y)
        svm_model.fit(X, y)
        ann_model.fit(X, y)

# Tính toán trọng số cho lịch sử gần đây
def weighted_prediction(history):
    if len(real_data) > 1:
        X = np.array([i for i in range(len(history))]).reshape(-1, 1)

        logistic_pred = logistic_model.predict_proba(X)[-1]
        decision_tree_pred = decision_tree_model.predict_proba(X)[-1]
        random_forest_pred = random_forest_model.predict_proba(X)[-1]
        svm_pred = svm_model.predict_proba(X)[-1]
        ann_pred = ann_model.predict_proba(X)[-1]

        # Kết hợp dự đoán từ các mô hình
        weighted_avg = (logistic_pred + decision_tree_pred +
                        random_forest_pred + svm_pred + ann_pred) / 5

        return label_encoder.inverse_transform([np.argmax(weighted_avg)])[0]
    return random.choice(["Tài", "Xỉu"])

# Dự đoán bằng logic Markov
def markov_chain_prediction(history):
    if len(history) < 2:
        return random.choice(["Tài", "Xỉu"])

    transitions = {"Tài": {"Tài": 0, "Xỉu": 0}, "Xỉu": {"Tài": 0, "Xỉu": 0}}
    for i in range(len(history) - 1):
        transitions[history[i]][history[i + 1]] += 1

    last_state = history[-1]
    probabilities = transitions[last_state]
    return max(probabilities, key=probabilities.get)

# Giải mã MD5
def crack_md5(md5_hash):
    if md5_hash in tx_dictionary:
        return tx_dictionary[md5_hash]

    for guess in ["Tài", "Xỉu"]:
        if hashlib.md5(guess.encode()).hexdigest() == md5_hash:
            return guess

    return "Không thể xác định kết quả từ MD5."

# Lệnh /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Chào mừng bạn đến với bot Tài Xỉu AI!")

# Lệnh /tx
async def tx(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = context.args[0] if context.args else ""
        history = user_input.split()

        if not all(item in ["t", "x"] for item in history):
            await update.message.reply_text("Lịch sử chỉ bao gồm 't' (Tài) hoặc 'x' (Xỉu).")
            return

        history = [label_encoder.inverse_transform([item])[0] for item in history]
        prediction = weighted_prediction(history)
        await update.message.reply_text(f"Dự đoán của bot: {prediction}")
    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")

# Lệnh /txmd
async def txmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        md5_hash = context.args[0]
        result = crack_md5(md5_hash)
        await update.message.reply_text(f"Kết quả từ MD5: {result}")
    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")

# Lệnh /add
async def add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = context.args[0] if context.args else ""
        new_data = user_input.split()

        if not all(item in ["t", "x"] for item in new_data):
            await update.message.reply_text("Kết quả chỉ bao gồm 't' (Tài) hoặc 'x' (Xỉu).")
            return

        for item in new_data:
            real_data.append([len(real_data), label_encoder.transform([item])[0]])

        update_models()
        await update.message.reply_text(f"Dữ liệu đã được thêm: {new_data}")
    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")

# Lệnh /history
async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    history = " ".join(history_data)
    await update.message.reply_text(f"Lịch sử gần nhất: {history}")

# Lệnh /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hướng dẫn sử dụng bot:\n"
        "/tx [lịch sử]: Dự đoán Tài/Xỉu.\n"
        "/txmd [MD5]: Giải mã MD5.\n"
        "/add [kết quả]: Thêm kết quả thực tế.\n"
        "/history: Xem lịch sử gần đây.\n"
        "/help: Xem hướng dẫn."
    )

# Chạy bot
if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("tx", tx))
    app.add_handler(CommandHandler("txmd", txmd))
    app.add_handler(CommandHandler("add", add))
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CommandHandler("help", help_command))

    print("Bot đang chạy...")
    app.run_polling()