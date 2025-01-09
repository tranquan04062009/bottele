import os
import random
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
from collections import Counter, deque
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from io import BytesIO
import json
import time
import sqlite3


TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise ValueError("Biến môi trường TELEGRAM_TOKEN không tìm thấy!")

BOT_NAME = "Bot Tài Xỉu Thông Minh"
DATABASE_NAME = "tx_feedback.db"
DATA_PERSISTENT_PATH = "bot_data.json"

history_data = deque(maxlen=400)
train_data = []
train_labels = []
le = LabelEncoder()
scaler = StandardScaler()

feedback_weights = {'correct': 1.0, 'incorrect': -0.75}
strategy_weights = {'deterministic': 0.7, 'cluster': 0.6, 'machine_learning': 1.4, 'probability': 0.4, 'streak': 0.4, 'statistical': 0.2}
last_prediction = {'result': None, 'strategy': None, 'model': None}
user_feedback_history = deque(maxlen=1000)

model_logistic = LogisticRegression(random_state=42, solver='liblinear')
model_svm = SVC(kernel='linear', probability=True, random_state=42)
model_sgd = SGDClassifier(loss='log_loss', random_state=42)
model_rf = RandomForestClassifier(random_state=42)
model_nb = GaussianNB()

model_calibrated_svm = CalibratedClassifierCV(model_svm, method='isotonic', cv=5)

model_kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
models_to_calibrate = [model_logistic, model_sgd, model_rf]
calibrated_models = {}


def create_feedback_table():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_type TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_user_feedback(feedback):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO user_feedback (feedback_type) VALUES (?)", (feedback,))
    conn.commit()
    conn.close()


def load_data_state():
    global strategy_weights, last_prediction, user_feedback_history, history_data
    if os.path.exists(DATA_PERSISTENT_PATH):
        with open(DATA_PERSISTENT_PATH, 'r') as f:
            loaded_data = json.load(f)
            strategy_weights = loaded_data.get("strategy_weights", strategy_weights)
            last_prediction = loaded_data.get("last_prediction", last_prediction)
            user_feedback_history = deque(loaded_data.get("user_feedback_history", []), maxlen=1000)
            history_data = deque(loaded_data.get("history_data", []), maxlen=400)
        print("Đã tải dữ liệu trạng thái.")


def save_data_state():
    global strategy_weights, last_prediction, user_feedback_history, history_data
    data = {
        "strategy_weights": strategy_weights,
        "last_prediction": last_prediction,
        "user_feedback_history": list(user_feedback_history),
        "history_data": list(history_data)
    }
    try:
        with open(DATA_PERSISTENT_PATH, 'w') as f:
            json.dump(data, f)
            print("Đã lưu dữ liệu trạng thái.")
    except Exception as e:
        print(f"Lỗi khi lưu dữ liệu: {e}")


def save_current_history_image():
    if not history_data:
        return
    chart_image = generate_history_chart(history_data)
    ts = time.time()
    name = f"chart_tx_{ts}.png"
    with open(name, "wb") as file:
        file.write(chart_image.read())
    print(f"Đã lưu biểu đồ: {name}")

def generate_history_chart(history):
    if not history:
        return None
    labels, values = zip(*Counter(history).items())
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=['skyblue', 'salmon'])
    for i, v in enumerate(values):
        plt.text(labels[i], v + 0.1, str(v), ha='center', va='bottom')
    plt.xlabel('Kết quả (T: Tài, X: Xỉu)', fontsize=12)
    plt.ylabel('Tần suất', fontsize=12)
    plt.title('Phân Bố Kết Quả', fontsize=14)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return buffer


def calculate_probabilities(history):
    if not history:
        return {"t": 0.5, "x": 0.5}
    counter = Counter(history)
    total = len(history)
    prob_tai = counter["t"] / total
    prob_xiu = counter["x"] / total
    return {"t": prob_tai, "x": prob_xiu}


def apply_probability_threshold(prob_dict, threshold_t=0.55, threshold_x=0.45):
    return "t" if prob_dict["t"] > threshold_t else "x" if prob_dict["x"] > threshold_x else None


def statistical_prediction(history, bias=0.5):
    if not history:
        return random.choice(["t", "x"])
    counter = Counter(history)
    total = len(history)
    if total == 0:
        return random.choice(["t", "x"])
    prob_tai = counter["t"] / total
    prob_xiu = counter["x"] / total
    return "t" if (random.random() < prob_tai * (1 + bias) / 2) else "x" if random.random() < (
                prob_xiu * (1 + bias) / 2) else random.choice(["t", "x"])


def prepare_data_for_models(history):
    if len(history) < 5:
        return None, None
    encoded_history_5 = le.fit_transform(history[-5:])
    features = np.array([encoded_history_5])
    X = scaler.fit_transform(features)
    labels = le.transform([history[-1]])
    y = np.array(labels)
    return X, y


def train_all_models():
    if len(train_data) < 10:
        return
    X, Y = [], []
    for history in train_data:
        features, label = prepare_data_for_models(history)
        if features is not None and label is not None:
            X.append(features[0])
            Y.append(label[0])

    if len(X) > 1 and len(Y) > 1:
        X=np.array(X)
        Y=np.array(Y)
        for model in models_to_calibrate:
            try:

                model.fit(X, Y)
                calibrated_models[model] = model
            except ValueError:
                pass
        model_svm.fit(X, Y)
        model_calibrated_svm.fit(X, Y)
    

def ml_prediction(history):
    if len(train_data) < 10:
        return statistical_prediction(history)

    features, label = prepare_data_for_models(history)
    if features is None:
        return None

    model_svm_prob = model_calibrated_svm.predict_proba(features)
    svm_prediction_label = model_calibrated_svm.predict(features)

    log_prob, log_label = _predict_probabilty(calibrated_models.get(model_logistic, model_logistic), features)
    sgd_prob, sgd_label = _predict_probabilty(calibrated_models.get(model_sgd, model_sgd), features)
    rf_prob, rf_label = _predict_probabilty(calibrated_models.get(model_rf, model_rf), features)

    tai_probabilities_average = []
    xiu_probabilities_average = []

    if not np.isnan(log_prob["t"]):
        tai_probabilities_average.append(log_prob["t"])
    if not np.isnan(sgd_prob["t"]):
        tai_probabilities_average.append(sgd_prob["t"])
    if not np.isnan(rf_prob["t"]):
        tai_probabilities_average.append(rf_prob["t"])
    if not np.isnan(log_prob["x"]):
        xiu_probabilities_average.append(log_prob["x"])
    if not np.isnan(sgd_prob["x"]):
        xiu_probabilities_average.append(sgd_prob["x"])
    if not np.isnan(rf_prob["x"]):
        xiu_probabilities_average.append(rf_prob["x"])

    average_prob_t = np.mean(tai_probabilities_average) if tai_probabilities_average else 0
    average_prob_x = np.mean(xiu_probabilities_average) if xiu_probabilities_average else 0

    avg_probabilty = {"t": average_prob_t, "x": average_prob_x}
    svm_label = le.inverse_transform(svm_prediction_label)[0]

    predicted_outcome = apply_probability_threshold(avg_probabilty, 0.52, 0.48)
    if predicted_outcome:
        return predicted_outcome
    else:
       return svm_label


def _predict_probabilty(model, features):
    if hasattr(model, 'predict_proba'):
        try:
            probs = model.predict_proba(features)[0]
            labels = le.inverse_transform(model.predict(features))
            prob_dictionary = dict(zip(le.classes_, probs))
            return prob_dictionary, labels[0]
        except ValueError:
            return {"t": float('NaN'), "x": float('NaN')}, None
    return {"t": float('NaN'), "x": float('NaN')}, None


def cluster_analysis(history):
    if len(history) < 5:
        return None
    encoded_history = le.fit_transform(history)
    reshaped_history = encoded_history.reshape(-1, 1)
    try:
        model_kmeans.fit(reshaped_history)
    except ValueError:
        return None
    last_five = le.transform(history[-5:])
    last_five = last_five.reshape(1, -1)
    if model_kmeans.predict(last_five[0].reshape(-1, 1))[0] == 0:
        counter = Counter(history[-5:])
        if counter["t"] > counter["x"]:
            return 't'
        else:
            return 'x'
    elif model_kmeans.predict(last_five[0].reshape(-1, 1))[0] == 1:
        if history[-1] == 't':
            return 'x'
        else:
            return 't'


def analyze_real_data(history):
    if len(history) < 3:
        return None
    if all(item == history[0] for item in history):
        return history[0]
    if all(history[i] != history[i + 1] for i in range(len(history) - 1)):
        return "t" if history[-1] == "x" else "x"
    return None


def deterministic_algorithm(history):
    if len(history) < 4:
        return None
    if history[-1] == history[-2] == history[-3] and history[-1] == 't':
        return 'x'
    if history[-1] == history[-2] == history[-3] and history[-1] == 'x':
        return 't'
    if history[-1] != history[-2] and history[-2] != history[-3] and history[-3] != history[-4]:
        return "t" if history[-1] == "x" else "x"
    return None


def adjust_strategy_weights(feedback, strategy):
    global strategy_weights
    weight_change = feedback_weights.get(feedback, 0.0)
    strategy_weights[strategy] += weight_change * strategy_weights[strategy] * 0.15
    strategy_weights[strategy] = min(max(strategy_weights[strategy], 0.01), 2.0)
    return strategy_weights

def combined_prediction(history):
    global last_prediction
    strategy = None
    algo_prediction = deterministic_algorithm(history)
    if algo_prediction:
        strategy = "deterministic"
        last_prediction.update({'strategy': strategy, 'result': algo_prediction})
        return algo_prediction
    
    cluster_prediction = cluster_analysis(history)
    if cluster_prediction:
       strategy = "cluster"
       last_prediction.update({'strategy': strategy, 'result': cluster_prediction})
       return cluster_prediction
    ml_pred = ml_prediction(history)
    if ml_pred:
        strategy = "machine_learning"
        last_prediction.update({'strategy': strategy, 'result': ml_pred})
        return ml_pred
    probability_dict = calculate_probabilities(history)
    probability_pred = apply_probability_threshold(probability_dict)
    if probability_pred:
        strategy = "probability"
        last_prediction.update({'strategy': strategy, 'result': probability_pred})
        return probability_pred
    streak_prediction = analyze_real_data(history)
    if streak_prediction:
        strategy = "streak"
        last_prediction.update({'strategy': strategy, 'result': streak_prediction})
        return streak_prediction
    strategy = "statistical"
    last_prediction.update({'strategy': strategy, 'result': statistical_prediction(history, 0.3)})
    return statistical_prediction(history, 0.3)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Chào mừng bạn đến với {BOT_NAME}!\n"
        "Sử dụng /tx để dự đoán, /add để thêm kết quả.\n"
        "Nhập /help để xem hướng dẫn, /history để xem lịch sử, /chart để xem biểu đồ hoặc /logchart để lưu biểu đồ.",
        parse_mode=ParseMode.MARKDOWN
    )


async def tx(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = ' '.join(context.args)
        if not user_input:
            await update.message.reply_text("Vui lòng nhập dãy lịch sử (t: Tài, x: Xỉu).")
            return
        history = user_input.split()
        if not all(item in ["t", "x"] for item in history):
            await update.message.reply_text("Dữ liệu không hợp lệ. Lịch sử chỉ chứa 't' (Tài) hoặc 'x' (Xỉu).")
            return
        history_data.extend(history)
        if len(history) >= 5:
            train_data.append(list(history_data))
            train_labels.append(history[-1])
        train_all_models()
        result = combined_prediction(list(history_data))
        last_prediction["model"] = BOT_NAME
        keyboard = [
            [InlineKeyboardButton("Đúng", callback_data='correct')],
            [InlineKeyboardButton("Sai", callback_data='incorrect')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        formatted_result = f"Kết quả dự đoán từ {BOT_NAME} : *{'Tài' if result == 't' else 'Xỉu'}* "
        await update.message.reply_text(formatted_result,reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        await update.message.reply_text(f"Lỗi: {e}")


async def add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = ' '.join(context.args)
        if not user_input:
            await update.message.reply_text("Vui lòng nhập kết quả thực tế (t: Tài, x: Xỉu)!")
            return
        new_data = user_input.split()
        if not all(item in ["t", "x"] for item in new_data):
            await update.message.reply_text("Dữ liệu không hợp lệ. Kết quả chỉ chứa 't' (Tài) hoặc 'x' (Xỉu).")
            return
        history_data.extend(new_data)
        for i in range(len(new_data) - 5 + 1):
            train_data.append(list(history_data))
            train_labels.append(new_data[i + 4])
        train_all_models()
        await update.message.reply_text(f"Đã cập nhật dữ liệu: {new_data}")
    except Exception as e:
        await update.message.reply_text(f"Lỗi: {e}")

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    feedback = query.data
    global  user_feedback_history
    if last_prediction.get("strategy") is None or last_prediction.get('result') is None or  last_prediction.get('model')  is None :
        await query.edit_message_text("Không thể ghi nhận phản hồi. Vui lòng thử lại sau.")
        return
    if feedback == 'correct':
        user_feedback_history.append({'result': last_prediction['result'], 'strategy': last_prediction['strategy'],
                                      'feedback': 'correct', 'timestamp': time.time()})
        save_user_feedback('correct')
        await query.edit_message_text("Cảm ơn! Phản hồi đã được ghi nhận.")
    elif feedback == 'incorrect':
       user_feedback_history.append({'result': last_prediction['result'], 'strategy': last_prediction['strategy'],
                                    'feedback': 'incorrect', 'timestamp': time.time()})
       save_user_feedback('incorrect')
       await query.edit_message_text("Cảm ơn! Tôi sẽ cố gắng cải thiện.")
    adjust_strategy_weights(feedback, last_prediction["strategy"])

async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not history_data:
        await update.message.reply_text("Chưa có dữ liệu lịch sử.")
    else:
        await update.message.reply_text(f"Lịch sử gần đây: {' '.join(history_data)}")

async def chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chart_image = generate_history_chart(history_data)
    if chart_image is None:
        await update.message.reply_text("Không có dữ liệu lịch sử để hiển thị biểu đồ.")
        return
    await update.message.reply_photo(photo=chart_image, caption='Biểu đồ kết quả.')


async def logchart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_current_history_image()
    await update.message.reply_text("Đã lưu biểu đồ vào máy chủ.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Hướng dẫn sử dụng *{BOT_NAME}*:\n"
        "/tx [dãy lịch sử]: Dự đoán kết quả Tài/Xỉu.\n"
        "/add [kết quả]: Cập nhật kết quả thực tế.\n"
        "/history : Xem lịch sử gần đây.\n"
        "/chart : Xem biểu đồ kết quả.\n"
        "/logchart : Lưu biểu đồ kết quả vào máy chủ.\n"
         "Ví dụ:\n"
        "- /tx t t x t x\n"
        "- /add t x x t t", parse_mode=ParseMode.MARKDOWN)


if __name__ == "__main__":
    create_feedback_table()
    load_data_state()
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("tx", tx))
    app.add_handler(CommandHandler("add", add))
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("chart", chart))
    app.add_handler(CommandHandler("logchart", logchart))
    app.add_handler(CallbackQueryHandler(button))
    print("Bot đang hoạt động...")
    app.run_polling()
    save_data_state()
    print ("Bot đã dừng.")