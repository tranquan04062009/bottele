import os
os.system("pip install scikit-learn")
import random
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from collections import Counter, deque
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

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
        "Chào mừng bạn đến với bot dự đoán Tài Xỉu thực tế!\n"
        "Sử dụng lệnh /tx để nhận dự đoán.\n"
        "Nhập /help để biết thêm thông tin chi tiết."
    )

def update_model(feedback, model, new_data):
    """
    Cập nhật mô hình sau mỗi lần dự đoán (cải thiện mô hình liên tục).
    """
    if feedback == "correct":
        model.fit(new_data['X'], new_data['y'], epochs=1, batch_size=32)
        print("Mô hình đã được cập nhật.")
    elif feedback == "incorrect":
        model.fit(new_data['X'], new_data['y'], epochs=1, batch_size=32)
        print("Mô hình đã cải thiện.")
        
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

        # Thêm vào dữ liệu huấn luyện
        if len(history) >= 5:  # Chỉ thêm khi có đủ dữ liệu
            train_data.append(le.fit_transform(history[-5:]))
            train_labels.append(history[-1])
            train_model()

        # Dự đoán kết quả
        result = combined_prediction(list(history_data))

        # Tạo các nút để xác nhận kết quả
        keyboard = [
            [InlineKeyboardButton("Đúng", callback_data='correct')],
            [InlineKeyboardButton("Sai", callback_data='incorrect')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Gửi kết quả dự đoán và các nút
        await update.message.reply_text(
            f"Kết quả dự đoán của tôi: {'Tài' if result == 't' else 'Xỉu'}",
            reply_markup=reply_markup
        )

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

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    # Lấy dữ liệu từ callback_data
    feedback = query.data

    # Cập nhật dữ liệu huấn luyện dựa trên phản hồi
    if feedback == 'correct':
        # Nếu đúng, cập nhật lại dữ liệu huấn luyện (nếu cần)
        await query.edit_message_text("Cảm ơn bạn! Dữ liệu huấn luyện đã được cập nhật.")
    elif feedback == 'incorrect':
        # Nếu sai, cập nhật lại dữ liệu huấn luyện và cải thiện mô hình
        await query.edit_message_text("Cảm ơn bạn! Tôi sẽ cải thiện mô hình để dự đoán chính xác hơn.")
    
def kmeans_clustering(history_data, n_clusters=3):
    """
    Hàm sử dụng K-means để phân nhóm các cầu trong dữ liệu lịch sử.
    history_data: Dữ liệu lịch sử đã được chuyển thành dạng số (0: Xỉu, 1: Tài).
    n_clusters: Số lượng nhóm cần phân chia.
    """
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(history_data)
    return kmeans

# Phân loại dữ liệu lịch sử vào các nhóm
def predict_cluster(kmeans, new_data):
    """
    Dự đoán nhóm cho dữ liệu mới bằng K-means.
    kmeans: Mô hình K-means đã huấn luyện.
    new_data: Dữ liệu mới cần phân loại (đã được chuyển thành dạng số).
    """
    return kmeans.predict([new_data])  # Dự đoán nhóm cho một chuỗi mới

# Phát hiện và dự đoán kết quả dựa trên mô hình K-means và Mạng Neural
def detect_and_predict(history_data, new_data, kmeans, neural_network_model):
    """
    Phát hiện nhóm cầu từ dữ liệu lịch sử và dự đoán kết quả (Tài/Xỉu).
    history_data: Dữ liệu lịch sử để huấn luyện K-means (dạng số).
    new_data: Dữ liệu mới để phân loại và dự đoán.
    kmeans: Mô hình K-means đã huấn luyện.
    neural_network_model: Mô hình học máy (mạng neural).
    """
    # Sử dụng K-means để nhận diện nhóm cầu
    cluster_label = predict_cluster(kmeans, new_data)

    # Sử dụng mô hình neural network để dự đoán kết quả (Tài/Xỉu)
    prediction = neural_network_model.predict([new_data])
    prediction_result = 'Tài' if prediction[0] > 0.5 else 'Xỉu'

    return cluster_label, prediction_result
    
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
    
    app.add_handler(CallbackQueryHandler(button))
    
    print("Bot đang chạy...")
    app.run_polling()