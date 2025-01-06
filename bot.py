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
import nest_asyncio  # Thêm vào để sửa lỗi event loop

# Sử dụng nest_asyncio để cho phép chạy lại event loop trong môi trường đã có event loop
nest_asyncio.apply()

# Lấy token từ biến môi trường
TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise ValueError("Vui lòng đặt biến môi trường TELEGRAM_TOKEN chứa token bot!")

# Bộ nhớ lịch sử thực tế
history_data = deque(maxlen=100)
real_data = []

# Sử dụng LabelEncoder để mã hóa 'Tài' và 'Xỉu'
label_encoder = LabelEncoder()
label_encoder.fit(["t", "x"])  # Đảm bảo rằng chúng ta chỉ mã hóa t và x

# Các mô hình học máy
logistic_model = LogisticRegression(max_iter=200)
decision_tree_model = DecisionTreeClassifier()
random_forest_model = RandomForestClassifier(n_estimators=100)
svm_model = SVC(probability=True)
ann_model = MLPClassifier(hidden_layer_sizes=(10,))

def update_models():
    """Huấn luyện lại các mô hình với dữ liệu mới từ lịch sử."""
    if len(real_data) > 1:
        X = np.array([item[0] for item in real_data]).reshape(-1, 1)
        y = np.array([item[1] for item in real_data])

        logistic_model.fit(X, y)
        decision_tree_model.fit(X, y)
        random_forest_model.fit(X, y)
        svm_model.fit(X, y)
        ann_model.fit(X, y)

def weighted_prediction(history):
    """
    Dự đoán dựa trên trọng số các kết quả gần đây.
    Tăng trọng số cho các kết quả gần nhất trong lịch sử.
    """
    if len(real_data) > 1:
        X = np.array([i for i in range(len(history))]).reshape(-1, 1)

        # Dự đoán từ các mô hình
        logistic_pred = logistic_model.predict(X)
        decision_tree_pred = decision_tree_model.predict(X)
        random_forest_pred = random_forest_model.predict(X)
        svm_pred = svm_model.predict(X)
        ann_pred = ann_model.predict(X)

        # Tính toán kết quả chung từ các mô hình
        predictions = [logistic_pred[-1], decision_tree_pred[-1], random_forest_pred[-1], svm_pred[-1], ann_pred[-1]]
        majority_vote = np.bincount(predictions).argmax()

        return label_encoder.inverse_transform([majority_vote])[0]

    return random.choice(['t', 'x'])

def crack_md5(md5_hash):
    """
    Tìm kiếm giá trị phù hợp với băm MD5 thông qua từ điển hoặc brute force.
    """
    # Kiểm tra trong từ điển
    if md5_hash in tx_dictionary:
        return tx_dictionary[md5_hash]

    # Tấn công brute force với các giá trị phổ biến
    for guess in ["t", "x"]:
        if hashlib.md5(guess.encode()).hexdigest() == md5_hash:
            return guess

    # Không tìm thấy
    return "Không thể xác định kết quả từ MD5."

# Lệnh /txmd
async def txmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Lấy giá trị MD5 từ người dùng
        if not context.args:
            await update.message.reply_text("Vui lòng cung cấp giá trị MD5 để kiểm tra!")
            return

        md5_hash = context.args[0]

        # Kiểm tra định dạng MD5 hợp lệ
        if len(md5_hash) != 32 or not all(c in "0123456789abcdef" for c in md5_hash.lower()):
            await update.message.reply_text("Giá trị MD5 không hợp lệ. Vui lòng kiểm tra lại.")
            return

        # Giải mã MD5
        result = crack_md5(md5_hash)
        await update.message.reply_text(f"Kết quả từ MD5: {result}")

    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")

# Lệnh /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Chào mừng bạn đến với bot dự đoán Tài Xỉu!\n"
        "Dùng lệnh /tx hoặc /txmd để kiểm tra kết quả."
    )

# Lệnh /tx
async def tx(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = ' '.join(context.args)

        if not user_input:
            await update.message.reply_text("Vui lòng nhập dãy lịch sử (t: Tài, x: Xỉu)!")
            return

        history = user_input.split()

        if not all(item in ["t", "x"] for item in history):
            await update.message.reply_text("Dãy lịch sử chỉ được chứa 't' (Tài) và 'x' (Xỉu).")
            return

        # Cập nhật dữ liệu thực tế
        history_data.extend(history)
        for item in history:
            real_data.append([len(real_data), label_encoder.transform([item])[0]])

        # Huấn luyện mô hình nếu có dữ liệu mới
        update_models()

        result = weighted_prediction(list(history_data))
        await update.message.reply_text(f"Kết quả dự đoán của tôi: {result}")

    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")

# Lệnh /add
async def add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = ' '.join(context.args)

        if not user_input:
            await update.message.reply_text("Vui lòng nhập kết quả thực tế (t: Tài, x: Xỉu)!")
            return

        new_data = user_input.split()

        if not all(item in ["t", "x"] for item in new_data):
            await update.message.reply_text("Kết quả chỉ được chứa 't' (Tài) và 'x' (Xỉu).")
            return

        # Cập nhật dữ liệu thực tế
        for item in new_data:
            real_data.append([len(real_data), label_encoder.transform([item])[0]])

        # Huấn luyện lại mô hình
        update_models()
        await update.message.reply_text(f"Dữ liệu thực tế đã được cập nhật: {new_data}")

    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")

# Lệnh /history
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
        "/txmd [md5]: Giải mã MD5 để tìm kết quả.\n"
        "/help: Hiển thị hướng dẫn."
    )

# Khởi tạo ứng dụng và các handler
async def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("tx", tx))
    app.add_handler(CommandHandler("txmd", txmd))
    app.add_handler(CommandHandler("add", add))
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CommandHandler("help", help_command))

    print("Bot đang chạy...")
    await app.run_polling()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())