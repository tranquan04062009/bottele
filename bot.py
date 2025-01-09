import os
import logging
import re
import time
import threading
import schedule
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
import warnings
from scipy import stats
import math
from urllib.parse import urljoin, urlparse
from queue import Queue
from datetime import datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackContext, CallbackQueryHandler

# Configure logging
LOG_FILE = 'bot_log.txt'
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

# Set default paths
DATA_FILE = 'game_data.csv'
MODEL_UPDATE_INTERVAL_MINUTES = 10
WEB_DATA_FILE = 'web_data.txt'

# Global application object
application = None

# --- Helper Functions ---

def create_feedback_keyboard(prediction_id):
    """Create an inline keyboard for feedback."""
    keyboard = [
        [
            InlineKeyboardButton("Correct", callback_data=f"feedback_correct_{prediction_id}"),
            InlineKeyboardButton("Incorrect", callback_data=f"feedback_incorrect_{prediction_id}"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


# --- Handler Functions ---

async def start(update: Update, context: CallbackContext):
    """Handles the /start command"""
    try:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Bot dự đoán game đã sẵn sàng. Sử dụng /help để xem hướng dẫn.")
    except Exception as e:
        logging.error(f"Error in start handler: {str(e)}")

async def help(update: Update, context: CallbackContext):
    """Handles the /help command"""
    try:
        help_text = """
Các lệnh có sẵn:
/predict - Dự đoán kết quả tiếp theo
/stats - Xem thống kê chi tiết
/pattern - Phân tích mẫu
/trend - Xem xu hướng hiện tại
/analyze - Phân tích toàn diện
/history - Xem lịch sử dự đoán
/accuracy - Xem độ chính xác
/url <web_url> <selector> - Thu thập dữ liệu từ trang web
/tx <number> <number> <number> - Nhập dữ liệu thủ công (cách nhau bởi khoảng trắng)
            """
        await context.bot.send_message(chat_id=update.effective_chat.id, text=help_text)
    except Exception as e:
        logging.error(f"Error in help handler: {str(e)}")

async def handle_tx(update: Update, context: CallbackContext):
    """Handles the /tx command to input historical data"""
    try:
         parts = update.message.text.split(' ', 1)
         if len(parts) < 2:
               await context.bot.send_message(chat_id=update.effective_chat.id, text="Vui lòng cung cấp các số cách nhau bằng khoảng trắng sau lệnh /tx")
               return
         numbers_text = parts[1].strip()
         numbers = []
         for num_str in numbers_text.split():
             if num_str.isdigit():
                 numbers.append(int(num_str))
             else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Vui lòng chỉ nhập các số hợp lệ. Dữ liệu bị bỏ qua")
                return
         if numbers:
            for number in numbers:
               predictor.record_game_result(number)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Đã ghi nhận {len(numbers)} số: {numbers}")
         else:
              await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Không tìm thấy số nào")

    except Exception as e:
            logging.error(f"Error in handle_tx: {str(e)}")
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Lỗi xử lý lệnh /tx. Vui lòng kiểm tra dữ liệu đầu vào")

async def handle_url(update: Update, context: CallbackContext):
    """Handles the /url command"""
    try:
         parts = update.message.text.split(' ', 2)
         if len(parts) < 3:
              await context.bot.send_message(chat_id=update.effective_chat.id, text="Vui lòng cung cấp URL và CSS selector. Ví dụ: `/url <url> <css selector>`")
              return
         url, selector = parts[1], parts[2]
         await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Đang thu thập dữ liệu từ {url} sử dụng selector `{selector}`...")
         await predictor.collect_data_from_url(url, selector, update, context)

    except IndexError:
         await context.bot.send_message(chat_id=update.effective_chat.id, text="Vui lòng cung cấp một URL hợp lệ và selector sau lệnh /url.")
    except Exception as e:
        logging.error(f"Error in handle_url: {str(e)}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Có lỗi xảy ra khi xử lý lệnh URL: {e}")


async def handle_stats(update: Update, context: CallbackContext):
    """Handles the /stats command to show simple statistics"""
    try:
       if not predictor.historical_data.empty:
         total_count = len(predictor.historical_data)
         correct_count = len(predictor.historical_data[predictor.historical_data['feedback'] == 'correct'])
         incorrect_count = len(predictor.historical_data[predictor.historical_data['feedback'] == 'incorrect'])

         stats_text = f"""
📊 Thống kê dữ liệu:
Tổng số lượng dự đoán: {total_count}
Số dự đoán chính xác: {correct_count}
Số dự đoán không chính xác: {incorrect_count}
"""
         await context.bot.send_message(chat_id=update.effective_chat.id, text=stats_text)
       else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Không có dữ liệu để hiển thị thống kê.")
    except Exception as e:
        logging.error(f"Error in handle_stats: {str(e)}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Có lỗi xảy ra khi hiển thị thống kê")


async def handle_history(update: Update, context: CallbackContext):
    """Handles the /history command"""
    try:
      if not predictor.historical_data.empty:
           history = predictor.historical_data.tail(10).to_string()
           await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Lịch sử dữ liệu gần nhất (10):\n {history}")
      else:
           await context.bot.send_message(chat_id=update.effective_chat.id, text="Không có lịch sử dự đoán để hiển thị.")
    except Exception as e:
        logging.error(f"Error in handle_history: {str(e)}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Lỗi hiển thị lịch sử dự đoán.")

async def handle_prediction(update: Update, context: CallbackContext):
    """Handles the /predict command"""
    try:
        if len(predictor.historical_data) < 10:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Cần ít nhất 10 số để dự đoán chính xác.")
            return

        numbers = predictor.historical_data['result'].tolist()
        # Tổng hợp các phương pháp dự đoán
        math_pred = predictor.mathematical_prediction(numbers)
        stat_analysis = predictor.statistical_analysis(numbers)
        ml_pred = predictor.machine_learning_prediction(numbers)
        patterns = predictor.pattern_analysis(numbers)
        evolution_result = predictor.evolutionary_algorithm(numbers)
        opportunities = predictor.opportunity_analysis(numbers)

        # Create report
        report = f"""
📊 Báo cáo dự đoán:

1. Phân tích toán học:
- Trung bình: {math_pred['basic_stats']['mean']:.2f}
- Độ lệch chuẩn: {math_pred['basic_stats']['std']:.2f}
- Khoảng tin cậy: [{math_pred['confidence_interval'][0]:.2f}, {math_pred['confidence_interval'][1]:.2f}]

2. Phân tích thống kê:
- Xu hướng: {stat_analysis['trend_analysis']['trend_direction']}
- Độ mạnh xu hướng: {stat_analysis['trend_analysis']['trend_strength']:.2f}

3. Dự đoán máy học:
- Ensemble: {ml_pred['ensemble_prediction']:.2f}
- Độ tin cậy: {ml_pred['confidence']['confidence_score']:.2f}

4. Mẫu phổ biến:
{patterns['repeating_patterns']}

5. Cơ hội:
- Điểm cơ hội: {opportunities['opportunity_score']:.2f}/100
- Xu hướng: {opportunities['current_trend']['direction']}
- Độ mạnh: {opportunities['current_trend']['strength']:.2f}

🎯 Dự đoán cuối cùng: {ml_pred['ensemble_prediction']:.2f}
⚠️ Độ tin cậy: {ml_pred['confidence']['confidence_score']*100:.2f}%
        """

        prediction_id = len(predictor.historical_data)
        keyboard = create_feedback_keyboard(prediction_id)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=report, reply_markup=keyboard)
    except Exception as e:
        logging.error(f"Error in handle_prediction: {str(e)}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Có lỗi xảy ra trong quá trình dự đoán.")

async def handle_feedback(update: Update, context: CallbackContext):
    """Handles feedback from the buttons."""
    try:
        query = update.callback_query
        await query.answer()

        feedback_data = query.data.split('_') # feedback_correct_{prediction_id} or feedback_incorrect_{prediction_id}
        feedback_type = feedback_data[1]
        prediction_id = int(feedback_data[2])

        if prediction_id < len(predictor.historical_data):
            predictor.record_feedback(prediction_id, feedback_type)
            await query.edit_message_text(text=f"Đã nhận phản hồi: {feedback_type} cho dự đoán {prediction_id}")
            if feedback_type == "correct":
                 predictor.update_models() # Update if the model is correct
        else:
             await query.edit_message_text(text="Dự đoán không hợp lệ. Có thể đã bị xoá.")
    except Exception as e:
         logging.error(f"Error in handle_feedback: {str(e)}")
         await query.edit_message_text(text="Lỗi xử lý phản hồi. Vui lòng thử lại.")


class GamePredictor:
    def __init__(self, bot_token):
         self.application = ApplicationBuilder().token(bot_token).build()
         self.game_data = []
         self.historical_data = pd.DataFrame()
         self.models = {
            'lr': LinearRegression(),
            'rf': RandomForestRegressor(),
            'svr': SVR(kernel='rbf')
         }
         self.scaler = StandardScaler()
         self.load_game_data()

    def start_bot(self):
        self.schedule_data_collection()
        self.schedule_model_updates()
        self.application.add_handler(CallbackQueryHandler(handle_feedback))
        self.application.run_polling() # Use global application object

    def load_game_data(self):
        """Load game data from CSV file"""
        try:
            if os.path.exists(DATA_FILE):
                self.historical_data = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])
                logging.info(f"Loaded {len(self.historical_data)} records from {DATA_FILE}")
            else:
                logging.info(f"{DATA_FILE} not found, starting with empty data.")
        except Exception as e:
            logging.error(f"Error loading game data: {str(e)}")

    def save_game_data(self):
        """Save game data to CSV file"""
        try:
            self.historical_data.to_csv(DATA_FILE, index=False)
            logging.info(f"Saved {len(self.historical_data)} records to {DATA_FILE}")
        except Exception as e:
            logging.error(f"Error saving game data: {str(e)}")

    def collect_data(self):
       """Scheduled data collection (example: random number between 1 and 6)"""
       # Generate a random game result
       new_result = np.random.randint(1, 7) # simulate a dice roll
       self.record_game_result(new_result)

    def record_game_result(self, result):
        """Record a single game result to the data"""
        timestamp = datetime.now()
        new_data = pd.DataFrame({'result': [result], 'timestamp': [timestamp]})
        self.historical_data = pd.concat([self.historical_data, new_data], ignore_index=True) # Add data
        self.save_game_data() # Save to file
        logging.info(f"Recorded game result: {result} at {timestamp}")

    def record_feedback(self, prediction_id, feedback_type):
       """Record feedback for a specific prediction"""
       try:
          self.historical_data.loc[prediction_id, 'feedback'] = feedback_type
          self.save_game_data()
          logging.info(f"Recorded feedback: {feedback_type} for prediction {prediction_id}")
       except Exception as e:
            logging.error(f"Error recording feedback: {str(e)}")

    async def collect_data_from_url(self, url, selector, update: Update, context: CallbackContext):
        """Thu thập và xử lý dữ liệu từ URL"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Attempt to get data using user selector
            elements = soup.select(selector)

            if elements:
                logging.info(f"Found data with user-provided selector '{selector}' on {url}")
            else:
                # Fallback with some common CSS selector if user specified selector fail to get data
                common_selectors = [
                    'span.bet-history__item-result', # for https://68gb2025.ink/?code=10853170
                    'p', 'span', 'div', 'li', '.result', '#result',
                    '.text-result', 'div.item-session.has-result .text',
                     '.history__item .result', 'div[class*="bet-history-"] .value-result',
                ]

                for sel in common_selectors:
                     elements = soup.select(sel)
                     if elements:
                         selector = sel
                         logging.warning(f"Could not find data with '{selector}', using fallback selector '{sel}' on {url}")
                         break
                else: # If no common selectors work
                    logging.error(f"Could not find data using any selector on {url}")
                    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Không tìm thấy dữ liệu trên {url} với selector đã cho, hoặc selector mặc định.")
                    return


            text_data = ' '.join([el.get_text().strip() for el in elements]) #Extract numbers from text
            numbers = self.extract_numbers_from_text(text_data)

            # Record valid numbers
            if numbers:
               for number in numbers:
                   self.record_game_result(number)
               logging.info(f"Extracted number from web: {numbers}")
            else:
               logging.warning(f"No numbers found on {url}")
               await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Không có số nào được tìm thấy trên {url}")
        except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching URL {url}: {str(e)}")
                await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Lỗi khi truy cập URL: {str(e)}")
        except Exception as e:
             logging.error(f"Error collecting data from {url}: {str(e)}")
             await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Lỗi không xác định: {str(e)}")


    def extract_numbers_from_text(self, text):
        """Extract numbers from text using regular expression"""
        try:
            numbers = re.findall(r'\b\d+\b', text)  # Find whole number (e.g., prevent getting 23 in 123)
            return [int(num) for num in numbers]
        except Exception as e:
            logging.error(f"Error extracting numbers: {str(e)}")
            return []

    def schedule_data_collection(self):
        """Lên lịch thu thập dữ liệu mỗi 1 phút"""
        def run_schedule():
            while True:
                schedule.run_pending()
                time.sleep(1)

        schedule.every(1).minutes.do(self.collect_data)
        threading.Thread(target=run_schedule).start()

    def schedule_model_updates(self):
       """Lên lịch cập nhật models"""
       def run_model_updates():
           while True:
               schedule.run_pending()
               time.sleep(1)

       schedule.every(MODEL_UPDATE_INTERVAL_MINUTES).minutes.do(self.update_models)
       threading.Thread(target=run_model_updates).start()

    def update_models(self):
        """Cập nhật các model ML"""
        try:
            if len(self.historical_data) < 10:
                logging.info("Not enough data to train models.")
                return

            # Filter for correct feedback
            correct_data = self.historical_data[self.historical_data['feedback'] == 'correct']
            if len(correct_data) < 5: # At least 5 examples before we update
                logging.info("Not enough 'correct' feedback to train models.")
                return

            X = self.prepare_features(correct_data)
            y = correct_data['result'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Cập nhật các model
            self.models['lr'].fit(X_train, y_train)
            self.models['rf'].fit(X_train, y_train)
            self.models['svr'].fit(X_train, y_train)

            logging.info("Models updated successfully")

        except Exception as e:
            logging.error(f"Model update error: {str(e)}")

    def prepare_features(self, df):
        """Chuẩn bị features cho ML"""
        df = df.copy()

        # Tạo features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['rolling_mean'] = df['result'].rolling(window=5).mean()
        df['rolling_std'] = df['result'].rolling(window=5).std()

        # One-hot encoding cho categorical features
        features = pd.get_dummies(df[['hour', 'day_of_week']])
        features = features.join(df[['rolling_mean', 'rolling_std']])

        # Fill any remaining NaN with 0
        features.fillna(0, inplace=True)

        return self.scaler.fit_transform(features)

    def mathematical_prediction(self, numbers):
        """1. Tính toán toán học nâng cao"""
        results = {
            'basic_stats': {
                'mean': np.mean(numbers),
                'median': np.median(numbers),
                'std': np.std(numbers),
                'variance': np.var(numbers)
            },
            'distribution': {
                'skewness': stats.skew(numbers),
                'kurtosis': stats.kurtosis(numbers)
            },
            'probability': self.calculate_probability(numbers),
            'confidence_interval': stats.t.interval(alpha=0.95, df=len(numbers)-1,
                                                 loc=np.mean(numbers),
                                                 scale=stats.sem(numbers))
        }
        return results

    def calculate_probability(self, numbers):
        """Tính xác suất chi tiết"""
        total = len(numbers)
        counter = Counter(numbers)
        basic_prob = {num: count/total for num, count in counter.items()}

        # Tính xác suất có điều kiện
        conditional_prob = defaultdict(dict)
        for i in range(len(numbers)-1):
            current = numbers[i]
            next_num = numbers[i+1]
            if current not in conditional_prob:
                conditional_prob[current] = defaultdict(int)
            conditional_prob[current][next_num] += 1

        # Chuẩn hóa xác suất có điều kiện
        for current in conditional_prob:
            total = sum(conditional_prob[current].values())
            for next_num in conditional_prob[current]:
                conditional_prob[current][next_num] /= total

        return {
            'basic_probability': basic_prob,
            'conditional_probability': dict(conditional_prob)
        }

    def statistical_analysis(self, numbers):
        """2. Phân tích thống kê nâng cao"""
        analysis = {
            'descriptive_stats': pd.Series(numbers).describe().to_dict(),
            'quartiles': {
                'Q1': np.percentile(numbers, 25),
                'Q2': np.percentile(numbers, 50),
                'Q3': np.percentile(numbers, 75),
                'IQR': np.percentile(numbers, 75) - np.percentile(numbers, 25)
            },
            'distribution_tests': {
                'normality': stats.normaltest(numbers),
                'uniformity': stats.kstest(numbers, 'uniform')
            },
            'trend_analysis': self.analyze_trend(numbers)
        }
        return analysis

    def analyze_trend(self, numbers):
        """Phân tích xu hướng"""
        diff = np.diff(numbers)
        return {
            'trend_direction': 'increasing' if np.mean(diff) > 0 else 'decreasing',
            'trend_strength': abs(np.mean(diff)),
            'volatility': np.std(diff),
            'momentum': sum(1 for x in diff if x > 0) / len(diff)
        }

    def machine_learning_prediction(self, numbers):
        """3. Dự đoán máy học nâng cao"""
        if len(numbers) < 10:
            return None

        # Chuẩn bị data
        X = np.array([numbers[i:i+5] for i in range(len(numbers)-5)])
        y = np.array(numbers[5:])

        # Train multiple models
        predictions = {}
        for name, model in self.models.items():
            model.fit(X, y)
            pred = model.predict([numbers[-5:]])[0]
            predictions[name] = pred

        # Ensemble prediction
        ensemble_pred = np.mean(list(predictions.values()))

        return {
            'individual_predictions': predictions,
            'ensemble_prediction': ensemble_pred,
            'confidence': self.calculate_prediction_confidence(predictions)
        }

    def calculate_prediction_confidence(self, predictions):
        """Tính độ tin cậy của dự đoán"""
        values = list(predictions.values())
        return {
            'std_dev': np.std(values),
            'range': max(values) - min(values),
            'confidence_score': 1 / (1 + np.std(values))
        }

    def pattern_analysis(self, numbers):
        """6. Phân tích mẫu nâng cao"""
        patterns = {
            'sequences': self.find_sequences(numbers),
            'repeating_patterns': self.find_repeating_patterns(numbers),
            'cycle_analysis': self.analyze_cycles(numbers)
        }
        return patterns

    def find_sequences(self, numbers):
        """Tìm các chuỗi số"""
        sequences = []
        current_seq = [numbers[0]]

        for i in range(1, len(numbers)):
            if numbers[i] == numbers[i-1] + 1:
                current_seq.append(numbers[i])
            else:
                if len(current_seq) > 1:
                    sequences.append(current_seq)
                current_seq = [numbers[i]]

        return sequences

    def find_repeating_patterns(self, numbers):
        """Tìm mẫu lặp lại"""
        patterns = {}
        for length in range(2, 6):
            for i in range(len(numbers) - length):
                pattern = tuple(numbers[i:i+length])
                if pattern in patterns:
                    patterns[pattern] += 1
                else:
                    patterns[pattern] = 1

        return dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5])

    def analyze_cycles(self, numbers):
        """Phân tích chu kỳ"""
        fft = np.fft.fft(numbers)
        frequencies = np.fft.fftfreq(len(numbers))
        dominant_cycles = sorted([(abs(fft[i]), 1/abs(freq))
                                for i, freq in enumerate(frequencies)
                                if freq > 0], reverse=True)[:3]
        return dominant_cycles

    def evolutionary_algorithm(self, numbers):
        """9. Thuật toán tiến hóa nâng cao"""
        class Individual:
            def __init__(self, genes):
                self.genes = genes
                self.fitness = 0

            def calculate_fitness(self, target):
                self.fitness = -abs(sum(self.genes) - target)
                return self.fitness

        class Population:
            def __init__(self, size, gene_length):
                self.individuals = [Individual(np.random.randint(0, 10, gene_length))
                                  for _ in range(size)]

            def evolve(self, generations, target):
                for gen in range(generations):
                    # Tính fitness
                    for ind in self.individuals:
                        ind.calculate_fitness(target)

                    # Sắp xếp theo fitness
                    self.individuals.sort(key=lambda x: x.fitness, reverse=True)

                    # Chọn lọc
                    next_gen = self.individuals[:len(self.individuals)//2]

                    # Lai ghép
                    while len(next_gen) < len(self.individuals):
                        parent1, parent2 = np.random.choice(next_gen, 2)
                        crossover_point = np.random.randint(0, len(parent1.genes))
                        child_genes = np.concatenate([parent1.genes[:crossover_point],
                                                    parent2.genes[crossover_point:]])
                        next_gen.append(Individual(child_genes))

                    # Đột biến
                    for ind in next_gen[1:]:
                        if np.random.random() < 0.1:
                            mutation_point = np.random.randint(0, len(ind.genes))
                            ind.genes[mutation_point] = np.random.randint(0, 10)

                    self.individuals = next_gen

                return self.individuals[0]

        # Sử dụng thuật toán tiến hóa
        target_sum = sum(numbers[-5:])
        pop = Population(size=50, gene_length=5)
        best_individual = pop.evolve(generations=100, target=target_sum)

        return {
            'predicted_sequence': best_individual.genes.tolist(),
            'fitness': best_individual.fitness,
            'target_sum': target_sum
        }

    def opportunity_analysis(self, numbers):
        """10. Phân tích cơ hội nâng cao"""
        analysis = {
            'current_trend': self.analyze_current_trend(numbers),
            'momentum_indicators': self.calculate_momentum(numbers),
            'volatility_analysis': self.analyze_volatility(numbers),
            'opportunity_score': self.calculate_opportunity_score(numbers)
        }
        return analysis

    def analyze_current_trend(self, numbers):
        """Phân tích xu hướng hiện tại"""
        if len(numbers) < 2:
            return None

        recent_numbers = numbers[-10:]
        trend = {
            'direction': 'up' if recent_numbers[-1] > recent_numbers[0] else 'down',
            'strength': abs(recent_numbers[-1] - recent_numbers[0]),
            'consistency': sum(1 for i in range(1, len(recent_numbers))
                             if (recent_numbers[i] - recent_numbers[i-1] > 0) ==
                                (recent_numbers[-1] > recent_numbers[0])) / (len(recent_numbers)-1)
        }
        return trend

    def calculate_momentum(self, numbers):
        """Tính momentum"""
        return {
            'roc': (numbers[-1] - numbers[0]) / numbers[0] if numbers[0] != 0 else 0,
            'acceleration': np.diff(np.diff(numbers)).mean(),
            'moving_average': pd.Series(numbers).rolling(window=5).mean().iloc[-1]
        }

    def analyze_volatility(self, numbers):
        """Phân tích biến động"""
        returns = np.diff(numbers) / numbers[:-1]
        return {
            'historical_volatility': np.std(returns) * np.sqrt(252),
            'average_true_range': sum(abs(high - low)
                                    for high, low in zip(numbers[1:], numbers[:-1])) / (len(numbers)-1)
        }

    def calculate_opportunity_score(self, numbers):
        """Tính điểm cơ hội"""
        trend = self.analyze_current_trend(numbers)
        momentum = self.calculate_momentum(numbers)
        volatility = self.analyze_volatility(numbers)

        score = 0
        if trend['direction'] == 'up':
            score += trend['strength'] * trend['consistency']
        score += momentum['roc'] * 100
        score -= volatility['historical_volatility']

        return max(min(score, 100), 0)  # Normalize to 0-100


def main():
    """Main entry point for the bot"""
    global application
    bot_token = os.environ.get("BOT_TOKEN")  # Get the bot token from environment variable
    if not bot_token:
       logging.error("Bot token not found. Make sure to set the BOT_TOKEN environment variable.")
       return

    predictor = GamePredictor(bot_token)
    application = predictor.application # set the application globally

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help))
    application.add_handler(CommandHandler("url", handle_url))
    application.add_handler(CommandHandler("predict", handle_prediction))
    application.add_handler(CommandHandler("tx", handle_tx))
    application.add_handler(CommandHandler("stats", handle_stats))
    application.add_handler(CommandHandler("history", handle_history))


    logging.info("Bot is starting...")  # Notify that the bot is starting
    predictor.start_bot()  # Start the bot
    logging.info("Bot has started successfully.")

if __name__ == "__main__":
    main()