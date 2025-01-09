import os
os.system("pip install scikit-learn")
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
    
def main_menu():
    keyboard = [
        [InlineKeyboardButton("Kết quả Tài/Xỉu", callback_data="tx")],
        [InlineKeyboardButton("Cập nhật dữ liệu thực tế", callback_data="add")],
        [InlineKeyboardButton("Xem lịch sử", callback_data="history")],
        [InlineKeyboardButton("Hướng dẫn", callback_data="help")]
    ]
    return InlineKeyboardMarkup(keyboard)

# Tạo menu Tài/Xỉu
def tx_menu():
    keyboard = [
        [InlineKeyboardButton("Tài", callback_data="tx_t")],
        [InlineKeyboardButton("Xỉu", callback_data="tx_x")],
        [InlineKeyboardButton("Xong", callback_data="finish_tx")]
    ]
    return InlineKeyboardMarkup(keyboard)

# Lệnh /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Chào mừng bạn đến với bot dự đoán Tài Xỉu! 😎\n"
        "Chọn một tùy chọn dưới đây để tiếp tục.",
        reply_markup=main_menu()
    )

# Lệnh /tx (dự đoán Tài/Xỉu)
async def tx(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.edit_message_text(
        text="Nhập lịch sử kết quả (t: Tài, x: Xỉu). Chọn Tài hoặc Xỉu nhé!",
        reply_markup=tx_menu()
    )

# Lệnh /add (cập nhật dữ liệu thực tế)
async def add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.edit_message_text(
        text="Nhập kết quả thực tế (t: Tài, x: Xỉu). Sau khi xong, nhấn 'Xong'!",
        reply_markup=tx_menu()
    )

# Lệnh /history (xem lịch sử)
async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not history_data:
        await update.callback_query.message.reply_text("Hiện tại chưa có dữ liệu lịch sử.")
    else:
        await update.callback_query.message.reply_text(f"Lịch sử gần nhất: {' '.join(history_data)}")

# Lệnh /help (hướng dẫn)
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.message.reply_text(
        "Hướng dẫn sử dụng bot:\n"
        "/tx: Dự đoán kết quả Tài/Xỉu.\n"
        "/add: Cập nhật dữ liệu thực tế.\n"
        "/history: Xem lịch sử gần nhất.\n"
        "/help: Hướng dẫn sử dụng bot.\n"
        "/start: khởi động bot.",
    )

# Xử lý các sự kiện từ menu
async def menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "tx":
        await tx(update, context)
    elif query.data == "add":
        await add(update, context)
    elif query.data == "history":
        await query.edit_message_text(
            text="Hiện tại chưa có dữ liệu lịch sử.",
            reply_markup=main_menu()
        )
    elif query.data == "help":
        await query.edit_message_text(
            text="Hướng dẫn sử dụng bot:\n"
                 "/tx: Dự đoán kết quả Tài/Xỉu.\n"
                 "/add: Cập nhật dữ liệu thực tế.\n"
                 "/history: Xem lịch sử gần nhất.\n"
                 "/help: Hướng dẫn sử dụng bot.\n"
                 "/start: khởi động bot.",
            reply_markup=main_menu()
        )

# Xử lý kết quả dự đoán Tài/Xỉu (Đúng/Sai)
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # Đáp lại callback để tránh lỗi timeout

    # Xử lý các nút trong menu Tài/Xỉu
    if query.data == "tx_t":
        history_data.append("Tài")
        result_text = f"ls hiện tại: {' '.join(history_data)}"
        await query.edit_message_text(
            text=result_text,
            reply_markup=tx_menu()
        )
    elif query.data == "tx_x":
        history_data.append("Xỉu")
        result_text = f"ls hiện tại: {' '.join(history_data)}"
        await query.edit_message_text(
            text=result_text,
            reply_markup=tx_menu()
        )
    elif query.data == "finish_tx":
        # Tạo dự đoán dựa trên lịch sử
        result = combined_prediction(history_data)
        result_text = f"Bot dự đoán kết quả: {result}"
        # Hiển thị nút "Đúng" và "Sai" sau khi dự đoán
        buttons = [
            [
                InlineKeyboardButton("Đúng", callback_data="correct"),
                InlineKeyboardButton("Sai", callback_data="incorrect"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(buttons)
        await query.edit_message_text(
            text=result_text,
            reply_markup=reply_markup
        )
        return
    else:
        result_text = "Không rõ hành động này. Vui lòng thử lại."

    # Cập nhật tin nhắn hiện tại
    await query.edit_message_text(
        text=result_text,
        reply_markup=tx_menu()
    )

# Xử lý kết quả đúng/sai
async def correct_incorrect_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "correct":
        await query.message.reply_text("Chúc mừng! 🎉 Kết quả chính xác, thêm vào dữ liệu huấn luyện.")
        # Cập nhật mô hình (thêm vào dữ liệu huấn luyện)
        train_data.append([history_data[-5:]])
        train_labels.append("t" if query.data == "t" else "x")
        train_model()
    else:
        await query.message.reply_text("Không sao, lần sau sẽ chính xác hơn! 😅")

    # Quay lại menu chính
    await query.message.reply_text(
        "Chọn một tùy chọn dưới đây để tiếp tục.",
        reply_markup=main_menu()
    )

# Main
def main():
    application = ApplicationBuilder().token(TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("tx", tx))
    application.add_handler(CommandHandler("add", add))
    application.add_handler(CommandHandler("history", history))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(menu_handler))
    application.add_handler(CallbackQueryHandler(menu_handler, pattern="^(tx|add|history|help)$"))  # Menu chính
    application.add_handler(CallbackQueryHandler(button_handler, pattern="^(tx_t|tx_x|finish_tx)$"))  # Tài/Xỉu
    application.add_handler(CallbackQueryHandler(correct_incorrect_handler, pattern="^(correct|incorrect)$"))
    application.run_polling()

if __name__ == "__main__":
    main()