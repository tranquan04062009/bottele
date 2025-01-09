import os
import random
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
from telegram.constants import ParseMode
from collections import Counter, deque
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

BOT_NAME = "👑 Bot Tài Xỉu Pro 👑"
DATABASE_NAME = "tx_feedback.db"
DATA_PERSISTENT_PATH = "bot_data.json"
HISTORY_DATA_PATH = "cau.json"

history_data = deque(maxlen=600)
train_data = []
train_labels = []
le = LabelEncoder()
scaler = StandardScaler()
poly = PolynomialFeatures(degree=2, include_bias=False)

feedback_weights = {'correct': 1.3, 'incorrect': -0.90}
strategy_weights = {'deterministic': 0.75, 'cluster': 0.70, 'machine_learning': 1.55, 'probability': 0.50, 'streak': 0.55, 'statistical': 0.30, 'boosting':1.25, 'mathematical': 0.65, 'statistical_algo': 0.45,'statistical_analysis':0.40, 'numerical_analysis':0.35, 'search_algorithm': 0.55, 'automatic_program':0.75, 'theorem_algo':0.60, 'evolutionary_algo': 0.95, 'reading_opportunity': 0.35}
last_prediction = {'result': None, 'strategy': None, 'model': None}
user_feedback_history = deque(maxlen=1000)
sentimental_analysis= {}


model_logistic = LogisticRegression(random_state=42, solver='liblinear',C = 1.1 , penalty = "l1")
model_svm = SVC(kernel='linear', probability=True, random_state=42 , C=1.4)
model_sgd = SGDClassifier(loss='log_loss', random_state=42 , alpha=0.01)
model_rf = RandomForestClassifier(random_state=42 ,n_estimators=150,max_depth = 9 ,min_samples_split = 2 )
model_gb = GradientBoostingClassifier(n_estimators=110, learning_rate=0.1 , max_depth= 6, random_state=42,subsample=0.9)
model_nb = GaussianNB()

model_calibrated_svm = CalibratedClassifierCV(model_svm, method='isotonic', cv=5)
model_kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
models_to_calibrate = [model_logistic, model_sgd, model_rf, model_gb]
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
    try:
       cursor.execute("INSERT INTO user_feedback (feedback_type) VALUES (?)", (feedback,))
       conn.commit()
    except sqlite3.Error as e :
       print("database exception during user saving on Feedback ",e)
    finally :
        conn.close()
def load_cau_data():
  global history_data
  if os.path.exists(HISTORY_DATA_PATH):
        try:
           with open(HISTORY_DATA_PATH, 'r') as f:
               loaded_data = json.load(f)
               history_data = deque(loaded_data, maxlen=600)
        except Exception as e:
              print(f"Could not open file for cau data:  {e}")

def save_cau_data():
   global history_data
   try:
     with open(HISTORY_DATA_PATH, 'w') as f:
            json.dump(list(history_data), f)
            print("Đã lưu dữ liệu cầu vào file cau.json.")
   except Exception as e :
       print(f"Could not save file for cau data  {e}")
def load_data_state():
    global strategy_weights, last_prediction, user_feedback_history, history_data, calibrated_models, train_data, train_labels , sentimental_analysis
    if os.path.exists(DATA_PERSISTENT_PATH):
         try:
             with open(DATA_PERSISTENT_PATH, 'r') as f:
               loaded_data = json.load(f)
               strategy_weights = loaded_data.get("strategy_weights", strategy_weights)
               last_prediction = loaded_data.get("last_prediction", last_prediction)
               user_feedback_history = deque(loaded_data.get("user_feedback_history", []), maxlen=1000)
               history_data = deque(loaded_data.get("history_data", []), maxlen=600)
               train_data = loaded_data.get("train_data", [])
               train_labels = loaded_data.get("train_labels", [])
               calibrated_models = loaded_data.get("calibrated_models",calibrated_models )
               sentimental_analysis=loaded_data.get ("sentimental_analysis", sentimental_analysis)
               print("Đã tải dữ liệu trạng thái từ file.")
         except Exception as e:
             print(f"Could not open file  {e}")
def generate_history_chart(history):
    if not history:
       return None
    labels, values = zip(*Counter(history).items())
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=['skyblue', 'salmon'])
    for i, v in enumerate(values):
        plt.text(labels[i], v + 0.1, str(v), ha='center', va='bottom')
    plt.xlabel('🎲 Kết quả (T: Tài, X: Xỉu)', fontsize=12)
    plt.ylabel('📊 Tần suất', fontsize=12)
    plt.title('📈 Phân Bố Kết Quả Gần Nhất', fontsize=14)
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
    if len(history) < 10:
          return None, None
    if len(set(history[-10:])) < 2:
          return None, None
    encoded_history = le.fit_transform(history[-10:])
    features = np.array([encoded_history],dtype=np.float64 )
    features_poly= poly.fit_transform(features)
    X = scaler.fit_transform(features_poly)
    labels = le.transform([history[-1]])
    y = np.array(labels ,dtype = np.int64)
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
     if len(X) > 1 and len(Y) > 1 :
            X = np.array(X)
            Y= np.array(Y)
            for model in models_to_calibrate:
               try:
                 model.fit(X, Y)
                 calibrated_models[model] = model
               except ValueError as ve :
                 print(f" model {model} error for data parameters : {ve}" )
                 pass

            model_svm.fit(X, Y)
            try:
               model_calibrated_svm.fit(X,Y)
            except ValueError as ve:
               print (f"Model Calibration Exception : {ve} skip model training")
def ml_prediction(history):
    if len(train_data) < 10:
        return statistical_prediction(history)
    features, label = prepare_data_for_models(history)
    if features is None:
       return statistical_prediction(history)
    try :
      model_svm_prob = model_calibrated_svm.predict_proba(features)
      svm_prediction_label = model_calibrated_svm.predict(features)
    except Exception as e :
         print (f"SVM prediction Model Exception  {e}, fallback Statistical method...")
         return statistical_prediction(history)
    log_prob, log_label = _predict_probabilty(calibrated_models.get(model_logistic, model_logistic), features)
    sgd_prob, sgd_label = _predict_probabilty(calibrated_models.get(model_sgd, model_sgd), features)
    rf_prob, rf_label = _predict_probabilty(calibrated_models.get(model_rf, model_rf), features)
    gb_prob, gb_label  = _predict_probabilty(calibrated_models.get(model_gb, model_gb), features)
    tai_probabilities_average = []
    xiu_probabilities_average = []
    if not np.isnan(log_prob["t"]): tai_probabilities_average.append(log_prob["t"])
    if not np.isnan(sgd_prob["t"]): tai_probabilities_average.append(sgd_prob["t"])
    if not np.isnan(rf_prob["t"]):  tai_probabilities_average.append(rf_prob["t"])
    if not np.isnan(gb_prob["t"]):  tai_probabilities_average.append(gb_prob["t"])
    if not np.isnan(log_prob["x"]):  xiu_probabilities_average.append(log_prob["x"])
    if not np.isnan(sgd_prob["x"]):   xiu_probabilities_average.append(sgd_prob["x"])
    if not np.isnan(rf_prob["x"]):   xiu_probabilities_average.append(rf_prob["x"])
    if not np.isnan(gb_prob["x"]) :  xiu_probabilities_average.append(gb_prob["x"])
    average_prob_t = np.mean(tai_probabilities_average) if tai_probabilities_average else 0
    average_prob_x = np.mean(xiu_probabilities_average) if xiu_probabilities_average else 0
    avg_probabilty = {"t": average_prob_t, "x": average_prob_x}
    svm_label = le.inverse_transform(svm_prediction_label)[0]
    predicted_outcome = apply_probability_threshold(avg_probabilty, 0.53, 0.47)
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
          except ValueError as ve :
               print (f"Model issue with probability : {ve} for {model}")
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
def mathematical_calculation(history):
     if not history or len(history) < 5 :
          return None
     sequence = "".join(history[-5:])
     t_count= sequence.count("t")
     x_count= sequence.count("x")
     if abs(t_count - x_count) > 2:
          return  "x" if t_count > x_count  else "t"
     else:
        return  "t" if random.random() < 0.6 else  "x"  if random.random() > 0.3 else None
def statistical_algorithm(history):
   if not history or len(history) < 10:
     return None
   last_10=history[-10:]
   count_t= last_10.count("t")
   count_x= last_10.count("x")
   if count_t == count_x :
    return "t" if random.random() < 0.5 else  "x"
   diff = abs(count_t - count_x)
   if diff >=3:
         return "t" if count_t < count_x else "x"
   else:
     return   "t" if random.random() < 0.55 else  "x" if random.random() > 0.45 else None
def statistical_analysis(history):
     if not history or len(history) < 8:
          return None
     last_8 = history[-8:]
     counter = Counter(last_8)
     total=len(last_8)
     prob_t= counter['t']/total if 't' in counter else 0
     prob_x= counter['x']/total  if 'x' in counter else 0
     if  abs(prob_t - prob_x) > 0.3:
           return 't' if prob_x > prob_t else 'x'
     else :
        return 't' if random.random() < 0.5 else 'x' if random.random() > 0.40  else  None
def numerical_analysis(history):
   if not history or len(history) < 6:
     return None
   numeric_representation= [1 if h=='t' else 0  for h in history[-6:]]
   sum_val = sum(numeric_representation)
   average_val= sum_val / len(numeric_representation)
   if average_val >= 0.70:
          return 't' if random.random() < 0.6 else  "x"
   elif average_val < 0.30:
       return 'x' if random.random() > 0.4 else "t"
   else :
        return None
def search_algorithm(history):
     if not history or len(history) < 7 :
          return None
     sequence_last_7 = "".join(history[-7:])
     patterns= ["ttx", "xtt","txt" ,"xxt","xtx","txx"]
     for pattern in patterns:
          if pattern in sequence_last_7:
              return 'x' if pattern[0] == "t"  else 't'
     return None
def automatic_programming(history):
     if not history or len(history) < 5:
          return None
     last_5 = "".join(history[-5:])
     if  last_5 in ['ttttt', "xxxxx"]:
         return  'x' if  last_5[0] == "t" else 't'
     if  last_5 in ['xtxtx', 'txtxt', 'txxtx' , "xttxx"  ] :
          return  't' if  last_5[0] == "x"  else "x"
     return None
def theorem_algorithm(history):
    if not history or len(history) < 6:
       return None
    last_6 = history[-6:]
    t_count= last_6.count('t')
    x_count= last_6.count('x')
    if  (t_count ==3 and x_count==3 ) or  ( t_count==4 and x_count==2)  or  (t_count==2 and x_count==4 ):
          return  "t" if random.random() < 0.5 else  "x"
    if t_count > x_count :
       return "x"
    elif x_count > t_count:
          return "t"
    return None
def evolutionary_algorithm(history):
   if not history or len(history) < 10:
     return None
   sequence= "".join(history[-10:])
   if   "ttttt" in sequence  or  "xxxxx" in sequence  :
     return "x" if sequence[0] == 't' else  "t"
   t_counts=[sequence.count(f't{"t" * i }') for i in range (1,3)]
   x_counts = [sequence.count(f'x{"x" * i }')  for i in range (1,3)]
   if  sum(t_counts) > sum(x_counts)  :
        return 'x'  if  random.random() > 0.2  else  None
   elif   sum(x_counts) > sum(t_counts):
         return 't' if random.random() > 0.2  else None
   return  None
def reading_opportunity(history):
     if not history or len(history) < 5 :
          return  None
     last_5_seq = "".join(history[-5:])
     if last_5_seq.count('t') == 3  and last_5_seq.count("x")==2:
           return  "x" if random.random() < 0.6 else  "t"
     if last_5_seq.count('x') == 3 and  last_5_seq.count("t") ==2 :
         return  "t"  if random.random() < 0.6 else "x"
     if "xtxt" in last_5_seq  or 'txtx' in last_5_seq :
            return   "t" if random.random() >0.4 else "x" if random.random() > 0.2  else  None
     return None
def combined_prediction(history):
     global last_prediction
     strategy=None
     algo_prediction = deterministic_algorithm(history)
     if algo_prediction:
          strategy = "deterministic"
          last_prediction.update({'strategy': strategy, 'result': algo_prediction})
          return  algo_prediction
     cluster_prediction= cluster_analysis(history)
     if cluster_prediction:
            strategy= "cluster"
            last_prediction.update({'strategy':strategy , 'result':cluster_prediction})
            return cluster_prediction
     ml_pred = ml_prediction(history)
     if ml_pred:
        strategy="machine_learning"
        last_prediction.update( { 'strategy': strategy,'result' : ml_pred})
        return ml_pred
     probability_dict =calculate_probabilities(history)
     probability_pred = apply_probability_threshold(probability_dict)
     if(probability_pred) :
         strategy= "probability"
         last_prediction.update({'strategy':strategy,'result': probability_pred})
         return probability_pred
     streak_prediction = analyze_real_data(history)
     if streak_prediction:
         strategy = "streak"
         last_prediction.update({'strategy':strategy, 'result':streak_prediction })
         return  streak_prediction
     math_prediction = mathematical_calculation(history)
     if math_prediction :
        strategy= "mathematical"
        last_prediction.update( { 'strategy': strategy,'result' : math_prediction})
        return  math_prediction
     statistical_algo_pred =  statistical_algorithm(history)
     if statistical_algo_pred:
         strategy= "statistical_algo"
         last_prediction.update({'strategy':strategy,'result': statistical_algo_pred})
         return statistical_algo_pred
     statistical_analysis_pred= statistical_analysis(history)
     if statistical_analysis_pred:
         strategy = "statistical_analysis"
         last_prediction.update({'strategy':strategy,'result' : statistical_analysis_pred })
         return  statistical_analysis_pred
     numerical_analysis_pred = numerical_analysis(history)
     if numerical_analysis_pred :
         strategy= "numerical_analysis"
         last_prediction.update({'strategy':strategy,'result':numerical_analysis_pred })
         return  numerical_analysis_pred
     search_algo_pred= search_algorithm(history)
     if search_algo_pred :
       strategy="search_algorithm"
       last_prediction.update({'strategy':strategy , 'result':search_algo_pred})
       return search_algo_pred
     automatic_prog_pred = automatic_programming(history)
     if automatic_prog_pred:
       strategy= "automatic_program"
       last_prediction.update({'strategy':strategy, 'result':automatic_prog_pred })
       return  automatic_prog_pred
     theorem_algo_pred = theorem_algorithm(history)
     if theorem_algo_pred:
       strategy="theorem_algo"
       last_prediction.update({'strategy':strategy, 'result':theorem_algo_pred })
       return theorem_algo_pred
     evolutionary_algo_pred = evolutionary_algorithm(history)
     if  evolutionary_algo_pred :
       strategy= "evolutionary_algo"
       last_prediction.update({'strategy':strategy,'result': evolutionary_algo_pred})
       return evolutionary_algo_pred
     reading_opportunity_pred = reading_opportunity(history)
     if  reading_opportunity_pred:
         strategy="reading_opportunity"
         last_prediction.update({'strategy':strategy, 'result': reading_opportunity_pred})
         return reading_opportunity_pred
     strategy = "boosting"
     last_prediction.update({'strategy':strategy,'result':statistical_prediction(history, 0.3) })
     return statistical_prediction(history, 0.3)
def calculate_training_status():
    total_predictions = len(user_feedback_history)
    if total_predictions == 0:
        return { "status" : "🤖 Chưa đủ dữ liệu.","accuracy" : 0 , "intelligence": 0  }
    correct_predictions = sum(1 for fb in user_feedback_history if fb['feedback'] == 'correct')
    accuracy_percentage = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    intelligence_level = np.mean(list(strategy_weights.values())) * 25 if  strategy_weights else 0
    status_report = {
      "status": "💪 Bot đang được huấn luyện.",
      "accuracy":  accuracy_percentage ,
        "intelligence" : intelligence_level if  intelligence_level <=100 else 100
    }
    return status_report

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
     start_text =  "✨ Chào mừng bạn đến với *{BOT_NAME}*!\n\n" \
                "🎲 Sử dụng /tx [dãy lịch sử] để nhận dự đoán Tài/Xỉu.\n" \
             "➕ Sử dụng /add [kết quả] để thêm kết quả thực tế.\n" \
              "🔄 Sử dụng /update để cập nhật toàn bộ lịch sử cược.\n" \
                "💾 Sử dụng /save để lưu dữ liệu hiện tại.\n" \
               "📜 Nhập /history để xem lịch sử cược gần nhất.\n" \
               "📊 Nhập /chart để xem biểu đồ tần suất.\n" \
               "💾 Nhập /logchart để lưu biểu đồ hiện tại vào server.\n" \
             "🧐 Nhập /status để xem trạng thái và độ thông minh bot\n\n" \
              "Bạn có thể bắt đầu sử dụng bằng cách nhập các lệnh trên, để trải nghiệm!\n"
     await update.message.reply_text(start_text.format(BOT_NAME=BOT_NAME), parse_mode=ParseMode.MARKDOWN)
async def update_cau(update: Update, context: ContextTypes.DEFAULT_TYPE):
      save_cau_data()
      await update.message.reply_text("🔄 Đã cập nhật dữ liệu cầu vào file.")
async def save_bot_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
      try:
        save_data_state()
        await update.message.reply_text("💾 Đã lưu dữ liệu bot vào file.")
      except Exception as e: # added parameter checks for exception output
         print(f"Error during saving data state {e}")
         await update.message.reply_text("💾 Không thể lưu dữ liệu")
async def tx(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_input = ' '.join(context.args)
        if not user_input:
             await update.message.reply_text("📝 Vui lòng nhập dãy lịch sử (t: Tài, x: Xỉu).")
             return
        history = user_input.split()
        if not all(item in ["t", "x"] for item in history):
           await update.message.reply_text("🚫 Dữ liệu không hợp lệ. Lịch sử chỉ chứa 't' (Tài) hoặc 'x' (Xỉu).")
           return
        history_data.extend(history)
        if len(history) >= 5:
              train_data.append(list(history_data))
              train_labels.append(history[-1])
        train_all_models()
        result = combined_prediction(list(history_data))
        last_prediction["model"] = BOT_NAME
        keyboard = [
             [InlineKeyboardButton("✅ Đúng", callback_data='correct')],
            [InlineKeyboardButton("❌ Sai", callback_data='incorrect')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        formatted_result = f"🔮 Kết quả dự đoán từ *{BOT_NAME}*: *{'✨Tài✨' if result == 't' else '🖤Xỉu🖤'}* "
        await update.message.reply_text(formatted_result, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Lỗi: {e}")
async def add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
       user_input = ' '.join(context.args)
       if not user_input:
         await update.message.reply_text("📝 Vui lòng nhập kết quả thực tế (t: Tài, x: Xỉu)!")
         return
       new_data = user_input.split()
       if not all(item in ["t", "x"] for item in new_data):
          await update.message.reply_text("🚫 Dữ liệu không hợp lệ. Kết quả chỉ chứa 't' (Tài) hoặc 'x' (Xỉu).")
          return
       for item in new_data :
         if item not in ["t", "x"]:
            await update.message.reply_text("🚫 Dữ liệu không hợp lệ. Kết quả chỉ chứa 't' (Tài) hoặc 'x' (Xỉu).")
            return
       history_data.extend(new_data)
       for i in range(len(new_data) - 5 + 1):
          train_data.append(list(history_data))
          train_labels.append(new_data[i + 4])
       train_all_models()
       await update.message.reply_text(f"➕ Đã cập nhật dữ liệu: {new_data}")
    except Exception as e:
          await update.message.reply_text(f"⚠️ Lỗi: {e}")
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    feedback = query.data
    global  user_feedback_history
    if last_prediction.get("strategy") is None or last_prediction.get('result') is None or  last_prediction.get('model')  is None :
       await query.edit_message_text("⚠️ Không thể ghi nhận phản hồi. Vui lòng thử lại sau.")
       return
    if feedback == 'correct':
        user_feedback_history.append({'result': last_prediction['result'], 'strategy': last_prediction['strategy'],
                                      'feedback': 'correct', 'timestamp': time.time()})
        save_user_feedback('correct')
        await query.edit_message_text("✅ Cảm ơn! Phản hồi đã được ghi nhận.")
    elif feedback == 'incorrect':
        user_feedback_history.append({'result': last_prediction['result'], 'strategy': last_prediction['strategy'],
                                   'feedback': 'incorrect', 'timestamp': time.time()})
        save_user_feedback('incorrect')
        await query.edit_message_text("❌ Cảm ơn! Tôi sẽ cố gắng cải thiện.")
    adjust_strategy_weights(feedback, last_prediction["strategy"])
async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not history_data:
       await update.message.reply_text("📜 Chưa có dữ liệu lịch sử.")
    else:
      await update.message.reply_text(f"📜 Lịch sử gần đây: {' '.join(history_data)}")
async def chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chart_image = generate_history_chart(history_data)
        if chart_image is None:
        await update.message.reply_text("📊 Không có dữ liệu để hiển thị biểu đồ.")
        return
    try:
       await update.message.reply_photo(photo=chart_image, caption="📈 Biểu đồ tần suất kết quả.")
    except Exception as e :
         print (f"error sending chart , error : {e}")
         await update.message.reply_text("📊 Không thể tạo biểu đồ. Lỗi không xác định")
async def logchart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        save_current_history_image()
        await update.message.reply_text("💾 Đã lưu biểu đồ vào máy chủ.")
    except Exception as e :
        print (f"Could not logchart: {e}")
        await update.message.reply_text("💾 Không thể lưu biểu đồ.")
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
     help_text = f"✨ Hướng dẫn sử dụng *{BOT_NAME}*:\n\n" \
                f"   🎲 /tx [dãy lịch sử]: Nhận dự đoán kết quả Tài/Xỉu.\n" \
                f"   ➕ /add [kết quả]: Cập nhật kết quả thực tế.\n" \
                f"   🔄 /update: Cập nhật toàn bộ lịch sử cược vào file.\n" \
                 f"  💾  /save: Lưu dữ liệu bot hiện tại vào file\n" \
               f"   📜 /history : Xem lịch sử gần đây.\n" \
                f"   📊 /chart : Xem biểu đồ tần suất.\n" \
              f"   💾 /logchart : Lưu biểu đồ vào máy chủ.\n" \
                f"   🧐 /status : Xem trạng thái huấn luyện và độ chính xác của bot.\n\n" \
               f"     _Ví dụ:_\n"  \
              f"         - /tx t t x t x\n"  \
              f"         - /add t x x t t"
     await update.message.reply_text(help_text.format(BOT_NAME=BOT_NAME), parse_mode=ParseMode.MARKDOWN)
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    training_report = calculate_training_status()
    formatted_message = (
        f"🤖 Trạng thái *{BOT_NAME}*:\n\n"
       f"   📊 Tình trạng: {training_report['status']}\n"
        f"   ✅ Độ chính xác: *{training_report['accuracy']:.2f}%*\n"
        f"   🧠 Mức độ thông minh: *{training_report['intelligence']:.2f}/100*\n"
    )
    await update.message.reply_text(formatted_message, parse_mode=ParseMode.MARKDOWN)
if __name__ == "__main__":
     create_feedback_table()
     load_data_state()
     load_cau_data()
     app = ApplicationBuilder().token(TOKEN).build()
     app.add_handler(CommandHandler("start", start))
     app.add_handler(CommandHandler("tx", tx))
     app.add_handler(CommandHandler("add", add))
     app.add_handler(CommandHandler("history", history))
     app.add_handler(CommandHandler("help", help_command))
     app.add_handler(CommandHandler("chart", chart))
     app.add_handler(CommandHandler("logchart", logchart))
     app.add_handler(CommandHandler("status", status))
     app.add_handler(CommandHandler("update", update_cau))
     app.add_handler(CommandHandler("save", save_bot_data))
     app.add_handler(CallbackQueryHandler(button))
     print("Bot đang hoạt động...")
     app.run_polling()
     print ("Bot đã dừng.")