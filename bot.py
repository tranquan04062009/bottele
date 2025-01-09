import telebot
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
import logging
import schedule
import time
import threading
from scipy import stats
import math
import os
os.system(
# Cấu hình logging
logging.basicConfig(filename='bot_log.txt', level=logging.INFO)
warnings.filterwarnings('ignore')

class GamePredictor:
    def __init__(self, bot_token):
        self.bot = telebot.TeleBot(bot_token)
        self.game_data = []
        self.historical_data = pd.DataFrame()
        
    def start(self):
        self.setup_handlers()
        self.schedule_data_collection()
        self.bot.infinity_polling()

    def setup_handlers(self):
        @self.bot.message_handler(commands=['start'])
        def send_welcome(message):
            self.bot.reply_to(message, "Bot dự đoán game đã sẵn sàng. Sử dụng /help để xem hướng dẫn.")

        @self.bot.message_handler(commands=['help'])
        def send_help(message):
            help_text = """
Các lệnh có sẵn:
/predict - Dự đoán kết quả tiếp theo
/stats - Xem thống kê chi tiết
/pattern - Phân tích mẫu
/trend - Xem xu hướng hiện tại
/analyze - Phân tích toàn diện
/history - Xem lịch sử dự đoán
/accuracy - Xem độ chính xác
/url <web_url> - Thu thập dữ liệu từ trang web
            """
            self.bot.reply_to(message, help_text)
        
        @self.bot.message_handler(commands=['url'])
        def handle_url(message):
            """Xử lý lệnh /url để thu thập dữ liệu từ trang web"""
            try:
                # Lấy URL từ lệnh
                url = message.text.split(' ', 1)[1]
                
                # Gọi hàm thu thập dữ liệu từ URL
                self.collect_data_from_url(url)
                self.bot.reply_to(message, f"Đang thu thập dữ liệu từ {url}...")
            except IndexError:
                self.bot.reply_to(message, "Vui lòng cung cấp một URL hợp lệ sau lệnh /url.")

    def collect_data_from_url(self, url):
        """Thu thập và xử lý dữ liệu từ URL"""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Giả sử ta lấy tất cả các đoạn văn bản từ trang web
            paragraphs = soup.find_all('p')
            text_data = ' '.join([para.get_text() for para in paragraphs])
            
            # Xử lý dữ liệu nếu cần (ví dụ: chuyển thành DataFrame hoặc thêm vào game_data)
            self.game_data.append(text_data)  # Thêm dữ liệu mới vào game_data
            logging.info(f"Thu thập dữ liệu từ {url}")
            
            # In ra một phần dữ liệu thu thập được
            logging.info(f"Thu thập {len(paragraphs)} đoạn văn bản từ trang web.")
            
            # Tiến hành phân tích hoặc dự đoán nếu cần
            self.analyze_data()
        
        except Exception as e:
            logging.error(f"Lỗi khi thu thập dữ liệu từ {url}: {str(e)}")

    def analyze_data(self):
        """Tiến hành phân tích dữ liệu đã thu thập (Ví dụ: phân tích mẫu, dự đoán)"""
        # Đơn giản thêm một bước phân tích mẫu từ dữ liệu đã thu thập
        if self.game_data:
            data_text = ' '.join(self.game_data)  # Kết hợp tất cả dữ liệu
            word_count = len(data_text.split())
            logging.info(f"Phân tích dữ liệu thu thập được, tổng số từ: {word_count}")
            # Tiến hành các phân tích hoặc dự đoán khác từ dữ liệu này
            # Bạn có thể triển khai các phân tích ở đây như phân tích từ ngữ, mô hình học máy, v.v.

    def schedule_data_collection(self):
        """Lên lịch thu thập dữ liệu mỗi 5 phút"""
        def run_schedule():
            while True:
                schedule.run_pending()
                time.sleep(1)

        schedule.every(5).minutes.do(self.collect_data)
        threading.Thread(target=run_schedule).start()

    def update_models(self):
        """Cập nhật các model ML"""
        try:
            X = self.prepare_features()
            y = self.historical_data['result'].values
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            # Cập nhật các model
            self.models['lr'] = LinearRegression().fit(X_train, y_train)
            self.models['rf'] = RandomForestRegressor().fit(X_train, y_train)
            self.models['svr'] = SVR(kernel='rbf').fit(X_train, y_train)
            
            logging.info("Models updated successfully")
            
        except Exception as e:
            logging.error(f"Model update error: {str(e)}")

    def prepare_features(self):
        """Chuẩn bị features cho ML"""
        df = self.historical_data.copy()
        
        # Tạo features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['rolling_mean'] = df['result'].rolling(window=5).mean()
        df['rolling_std'] = df['result'].rolling(window=5).std()
        
        # One-hot encoding cho categorical features
        features = pd.get_dummies(df[['hour', 'day_of_week']])
        features = features.join(df[['rolling_mean', 'rolling_std']])
        
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

    @bot.message_handler(commands=['predict'])
    def handle_prediction(self, message):
        """Xử lý lệnh dự đoán"""
        try:
            if len(self.game_data) < 10:
                self.bot.reply_to(message, "Cần ít nhất 10 số để dự đoán chính xác.")
                return

            # Tổng hợp các phương pháp dự đoán
            math_pred = self.mathematical_prediction(self.game_data)
            stat_analysis = self.statistical_analysis(self.game_data)
            ml_pred = self.machine_learning_prediction(self.game_data)
            patterns = self.pattern_analysis(self.game_data)
            evolution_result = self.evolutionary_algorithm(self.game_data)
            opportunities = self.opportunity_analysis(self.game_data)

            # Tạo báo cáo
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

            self.bot.reply_to(message, report)
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            self.bot.reply_to(message, "Có lỗi xảy ra trong quá trình dự đoán.")

if __name__ == "__main__":
    predictor = GamePredictor("YOUR_BOT_TOKEN")
    predictor.start()