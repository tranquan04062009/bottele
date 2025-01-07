import os
os.system("pip install scikit-learn")
import random
from collections import Counter, deque
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

# Token bot Telegram
TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise ValueError("Vui lòng đặt biến môi trường TELEGRAM_TOKEN chứa token bot!")

# Bộ nhớ lịch sử
history_data = deque(maxlen=100)  # Lưu tối đa 100 kết quả Tài/Xỉu
dice_data = deque(maxlen=100)     # Lưu tối đa 100 kết quả súc sắc

# ==============================
# Các hàm hỗ trợ
# ==============================

def detect_pattern(history):
    """
    Phát hiện kiểu cầu từ lịch sử:
    - Cầu 1-1: Xen kẽ Tài - Xỉu
    - Cầu bệt: Lặp lại liên tục Tài hoặc Xỉu
    """
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
    """
    Dự đoán dựa trên trọng số Tài/Xỉu từ lịch sử và tính tỷ lệ phần trăm.
    """
    if not history:
        return random.choice(["t", "x"]), 50.0, 50.0  # Nếu không có lịch sử, dự đoán ngẫu nhiên với tỷ lệ 50/50

    counter = Counter(history)
    prob_tai = counter["t"] / len(history) * 100  # Tính tỷ lệ phần trăm Tài
    prob_xiu = counter["x"] / len(history) * 100  # Tính tỷ lệ phần trăm Xỉu

    prediction = "t" if prob_tai > prob_xiu else "x"
    return prediction, prob_tai, prob_xiu

def combine_predictions(history, dice_values):
    """
    Kết hợp dự đoán từ lịch sử và tổng súc sắc.
    """
    prediction, prob_tai, prob_xiu = weighted_prediction(history)
    total_points = sum(dice_values) if dice_values else 0
    dice_prediction = "t" if total_points % 2 == 0 else "x"

    # Ưu tiên dự đoán từ lịch sử nếu có xu hướng rõ ràng
    final_prediction = prediction if prediction == dice_prediction else dice_prediction
    return final_prediction, prob_tai, prob_xiu

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

        history_data.extend(history)  # Cập nhật lịch sử
        prediction, prob_tai, prob_xiu = weighted_prediction(history)
        pattern = detect_pattern(list(history_data))

        # Tạo nút xác nhận
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

        dice_data.extend(dice_values)  # Cập nhật dữ liệu súc sắc
        prediction, prob_tai, prob_xiu = combine_predictions(list(history_data), dice_values)
        pattern = detect_pattern(list(history_data))

        # Tạo nút xác nhận
        buttons = InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ Đúng", callback_data=f"correct|{prediction}"), 
             InlineKeyboardButton("❌ Sai", callback_data=f"wrong|{prediction}")]
        ])

        await update.message.reply_text(
            f"Dự đoán: {'Tài' if prediction == 't' else 'Xỉu'}\n"
            f"Tỷ lệ phần trăm Tài: {prob_tai:.2f}%\n"
            f"Tỷ lệ phần trăm Xỉu: {prob_xiu:.2f}%\n"
            f"Phát hiện cầu: {pattern}\n"
            f"Dữ liệu súc sắc: {', '.join(map(str, dice_values))}",
            reply_markup=buttons
        )
    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")

# Lệnh /add: Thêm dữ liệu mới
async def add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = ' '.join(context.args)
        if not user_input:
            await update.message.reply_text("Vui lòng nhập dữ liệu dạng: 't x t | 12 6 8'.")
            return

        parts = user_input.split("|")
        if len(parts) != 2:
            await update.message.reply_text("Dữ liệu không hợp lệ! Nhập dạng 't x t | 12 6 8'.")
            return

        history = parts[0].strip().split()
        dice_values = list(map(int, parts[1].strip().split()))

        if not all(item in ["t", "x"] for item in history):
            await update.message.reply_text("Lịch sử chỉ được chứa 't' hoặc 'x'.")
            return
        if not all(isinstance(i, int) for i in dice_values):
            await update.message.reply_text("Dữ liệu súc sắc phải là số nguyên.")
            return

        history_data.extend(history)
        dice_data.extend(dice_values)

        await update.message.reply_text("Dữ liệu đã được thêm thành công!")
    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")

# Lệnh /history: Xem lịch sử
async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not history_data and not dice_data:
        await update.message.reply_text("Chưa có dữ liệu lịch sử.")
    else:
        await update.message.reply_text(
            f"Lịch sử gần nhất:\n"
            f"Tài/Xỉu: {' '.join(history_data)}\n"
            f"Súc sắc: {', '.join(map(str, dice_data))}"
        )

# Lệnh xử lý callback từ button
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
    app.add_handler(CommandHandler("tx", tx))
    app.add_handler(CommandHandler("txs", txs))
    app.add_handler(CommandHandler("add", add))
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CallbackQueryHandler(handle_callback))

    print("Bot đang chạy...")
    app.run_polling()