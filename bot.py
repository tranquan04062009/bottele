import os
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
from scipy.optimize import minimize
import math

# Lấy token từ biến môi trường
TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise ValueError("Vui lòng đặt biến môi trường TELEGRAM_TOKEN chứa token bot!")

# Bộ nhớ lịch sử thực tế
history_data = deque(maxlen=100)
real_data = []

# Sử dụng LabelEncoder để mã hóa 'Tài' và 'Xỉu'
label_encoder = LabelEncoder()
label_encoder.fit(["Tài", "Xỉu"])

# Các mô hình học máy
logistic_model = LogisticRegression()
decision_tree_model = DecisionTreeClassifier()
random_forest_model = RandomForestClassifier(n_estimators=100)
svm_model = SVC()
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

def simulated_annealing(f, bounds, max_iter=1000, initial_temp=1000, cooling_rate=0.95):
    """Tối ưu hóa với thuật toán Simulated Annealing."""
    x = np.random.uniform(bounds[0], bounds[1])  # Giá trị khởi tạo
    best_x = x
    best_f = f(x)
    temperature = initial_temp

    for i in range(max_iter):
        # Sinh giá trị mới ngẫu nhiên
        new_x = x + np.random.uniform(-1, 1)
        new_x = np.clip(new_x, bounds[0], bounds[1])
        new_f = f(new_x)

        # Quyết định có chấp nhận giá trị mới không
        delta = new_f - best_f
        if delta < 0 or np.exp(-delta / temperature) > random.random():
            x = new_x
            if new_f < best_f:
                best_x, best_f = new_x, new_f

        # Giảm nhiệt độ
        temperature *= cooling_rate

    return best_x, best_f

def particle_swarm_optimization(f, bounds, num_particles=30, max_iter=100):
    """Tối ưu hóa với thuật toán Particle Swarm Optimization (PSO)."""
    dim = len(bounds)
    particles = np.random.uniform(bounds[0], bounds[1], (num_particles, dim))
    velocities = np.random.uniform(-1, 1, (num_particles, dim))
    personal_best = particles.copy()
    personal_best_scores = np.array([f(p) for p in personal_best])
    global_best = personal_best[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)

    for _ in range(max_iter):
        for i in range(num_particles):
            # Cập nhật vận tốc và vị trí
            r1, r2 = random.random(), random.random()
            velocities[i] += r1 * (personal_best[i] - particles[i]) + r2 * (global_best - particles[i])
            particles[i] += velocities[i]

            # Cập nhật cá nhân tốt nhất
            score = f(particles[i])
            if score < personal_best_scores[i]:
                personal_best[i] = particles[i]
                personal_best_scores[i] = score

            # Cập nhật toàn cục tốt nhất
            if score < global_best_score:
                global_best = particles[i]
                global_best_score = score

    return global_best, global_best_score

def gradient_descent(f, gradient_f, start_point, learning_rate=0.01, max_iter=1000):
    """Tối ưu hóa với thuật toán Gradient Descent."""
    x = np.array(start_point)
    for _ in range(max_iter):
        grad = np.array(gradient_f(x))
        x = x - learning_rate * grad
        if np.linalg.norm(grad) < 1e-6:  # Dừng khi gradient rất nhỏ
            break
    return x, f(x)

def weighted_prediction(history):
    """
    Dự đoán dựa trên trọng số các kết quả gần đây.
    Tăng trọng số cho các kết quả gần nhất trong lịch sử.
    """
    if len(real_data) > 1:
        X = np.array([i for i in range(len(history))]).reshape(-1, 1)
        logistic_pred = logistic_model.predict(X)
        decision_tree_pred = decision_tree_model.predict(X)
        random_forest_pred = random_forest_model.predict(X)
        svm_pred = svm_model.predict(X)
        ann_pred = ann_model.predict(X)

        # Tính toán kết quả tối ưu từ các mô hình
        predictions = [logistic_pred[-1], decision_tree_pred[-1], random_forest_pred[-1], svm_pred[-1], ann_pred[-1]]
        majority_vote = np.bincount(predictions).argmax()

        return label_encoder.inverse_transform([majority_vote])[0]
    return random.choice(['Tài', 'Xỉu'])

# Lệnh /optimize
async def optimize(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        def sample_function(x):
            return (x - 3) ** 2 + 4  # Hàm mẫu cần tối ưu hóa

        bounds = [-10, 10]

        # Simulated Annealing
        sa_result, sa_score = simulated_annealing(sample_function, bounds)
        # Particle Swarm Optimization
        pso_result, pso_score = particle_swarm_optimization(sample_function, bounds)
        # Gradient Descent
        gd_result, gd_score = gradient_descent(sample_function, lambda x: 2 * (x - 3), [0])

        await update.message.reply_text(
            f"Kết quả tối ưu hóa:\n"
            f"Simulated Annealing: Giá trị = {sa_result}, Điểm số = {sa_score}\n"
            f"Particle Swarm Optimization: Giá trị = {pso_result}, Điểm số = {pso_score}\n"
            f"Gradient Descent: Giá trị = {gd_result}, Điểm số = {gd_score}\n"
        )
    except Exception as e:
        await update.message.reply_text(f"Đã xảy ra lỗi: {e}")

# Lệnh /help cập nhật
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hướng dẫn sử dụng bot:\n"
        "/tx [dãy lịch sử]: Dự đoán kết quả Tài/Xỉu.\n"
        "/add [kết quả]: Cập nhật kết quả thực tế.\n"
        "/history: Xem lịch sử gần đây.\n"
        "/txmd [md5]: Giải mã MD5 để tìm kết quả.\n"
        "/optimize: Tìm hiểu chiến lược tối ưu hóa.\n"
    )

# Khởi chạy bot
if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("tx", tx))
    app.add_handler(CommandHandler("add", add))
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CommandHandler("txmd", txmd))
    app.add_handler(CommandHandler("optimize", optimize))
    app.add_handler(CommandHandler("help", help_command))

    print("Bot đang chạy...")
    app.run_polling()