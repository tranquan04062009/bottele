import logging
import os
import uuid
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, Filters, CallbackQueryHandler, ContextTypes
from PIL import Image, ImageFilter
import pytesseract
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import cv2
import pickle
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import List, Tuple

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
DATA_PATH = "data"
MODEL_PATH = "model.pth"
SCALER_PATH = "scaler.pkl"
BOT_DATA_PATH = "bot_data.pkl"
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

@dataclass
class BotData:
    history_data: List[np.ndarray] = field(default_factory=list)
    history_labels: List[int] = field(default_factory=list)
    model: nn.Module = None
    scaler: StandardScaler = None
    
def load_bot_data() -> BotData:
    if os.path.exists(BOT_DATA_PATH):
      with open(BOT_DATA_PATH, 'rb') as f:
        logging.info("Loaded bot data")
        return pickle.load(f)
    else:
        logging.info("Created new bot data")
        return BotData()
        
def save_bot_data(bot_data: BotData):
    with open(BOT_DATA_PATH, 'wb') as f:
      pickle.dump(bot_data, f)
    logging.info("Saved bot data")
    
def download_image(url, folder):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{uuid.uuid4()}.jpg")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return file_path
    return None

def enhance_image(path):
    try:
        img = cv2.imread(path)
        if img is None:
            logging.error(f"Could not read image from path: {path}")
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        enhanced_path = path[:-4] + "_enhanced" + path[-4:]
        cv2.imwrite(enhanced_path, thresh)
        return enhanced_path
    except Exception as e:
        logging.error(f"Error enhancing image: {e}")
        return None

def extract_numbers_from_image(path):
    enhanced_path = enhance_image(path)
    if not enhanced_path:
        return None
    try:
        img = Image.open(enhanced_path).convert('L')
        text = pytesseract.image_to_string(img, config="--psm 6")
        return [int(s) for s in text.split() if s.isdigit()]
    except Exception as e:
        logging.error(f"Error OCR: {e}")
        return None

def analyze_path(path):
    numbers = extract_numbers_from_image(path)
    if not numbers:
      return None
    if len(numbers) == 1:
        return [0]
    return [1 if numbers[i+1] > numbers[i] else -1 if numbers[i+1] < numbers[i] else 0 for i in range(len(numbers) - 1)]

def create_feature_vector(numbers, path):
    padded_numbers = numbers + [0] * (20 - len(numbers)) if len(numbers) < 20 else numbers[:20]
    padded_path = path + [0] * (20 - len(path)) if len(path) < 20 else path[:20]
    return np.array(padded_numbers + padded_path)

class TaiXiuDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class TaiXiuPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def train_model(model, scaler, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    train_dataset = TaiXiuDataset(X_train_scaled, y_train)
    test_dataset = TaiXiuDataset(X_test_scaled, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            X_batch = X_batch.unsqueeze(1)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            avg_loss = total_loss/len(train_loader)
            logging.info(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    model.eval()
    with torch.no_grad():
        test_loss = 0
        y_true = []
        y_pred = []
        for X_batch, y_batch in test_loader:
           X_batch = X_batch.unsqueeze(1)
           output = model(X_batch)
           loss = criterion(output, y_batch)
           test_loss += loss.item()
           _, predicted = torch.max(output, 1)
           y_true.extend(y_batch.tolist())
           y_pred.extend(predicted.tolist())
        avg_test_loss = test_loss / len(test_loader)
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        logging.info(f"Accuracy on test set: {accuracy * 100:.2f}%")
        logging.info(f"Test Loss: {avg_test_loss:.4f}")
        logging.info(f"Classification Report:\n {report}")
    return model, scaler

def load_or_create_model_and_scaler(input_size: int) -> Tuple[nn.Module, StandardScaler]:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
      with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
      model = torch.load(MODEL_PATH)
      logging.info("Loaded model and scaler")
      return model, scaler
    else:
      hidden_size = 128
      num_layers = 2
      output_size = 2
      model = TaiXiuPredictor(input_size, hidden_size, num_layers, output_size, dropout_rate=0.3)
      scaler = StandardScaler()
      logging.info("Created new model and scaler")
      return model, scaler
    
def save_model_and_scaler(model: nn.Module, scaler: StandardScaler):
    with open(SCALER_PATH, 'wb') as f:
      pickle.dump(scaler, f)
    torch.save(model, MODEL_PATH)
    logging.info("Saved model and scaler")

def predict_next_outcome(model, scaler, path) -> Tuple[str, float]:
    numbers = extract_numbers_from_image(path)
    path_analysis = analyze_path(path)
    if not numbers or path_analysis is None:
      return None, None
    feature_vector = create_feature_vector(numbers, path_analysis)
    scaled_vector = scaler.transform(feature_vector.reshape(1, -1))
    with torch.no_grad():
      input_tensor = torch.tensor(scaled_vector, dtype=torch.float32).unsqueeze(1)
      output = model(input_tensor)
      _, predicted = torch.max(output, 1)
    prediction_str = "Tài" if predicted.item() == 1 else "Xỉu"
    probabilities = torch.nn.functional.softmax(output, dim=1)[0].tolist()
    return prediction_str, max(probabilities)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Chào mừng đến với bot dự đoán Tài Xỉu! Gửi ảnh lịch sử để bắt đầu.')

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    path = download_image(update.message.photo[-1].file_path, DATA_PATH)
    if not path:
      await update.message.reply_text("Lỗi tải ảnh.")
      return
    prediction, confidence = predict_next_outcome(context.application.bot_data.model, context.application.bot_data.scaler, path)
    if not prediction:
        await update.message.reply_text("Lỗi dự đoán, không có đủ dữ liệu đầu vào.")
        return
    context.user_data['last_path'] = path
    context.user_data['last_prediction'] = prediction
    numbers = extract_numbers_from_image(path)
    path_analysis = analyze_path(path)
    context.user_data['last_feature_vector'] = create_feature_vector(numbers, path_analysis)
    message = (f"**Phân tích kết quả:**\n"
               f"- Các số đã nhận dạng: `{numbers}`\n"
               f"- Đường đi (1 tăng, -1 giảm, 0 ngang): `{path_analysis}`\n\n"
               f"**Kết quả dự đoán:**\n"
               f"- Dự đoán: **{prediction}**\n"
               f"- Độ tin cậy: `{confidence * 100:.2f}%`")
    keyboard = [[InlineKeyboardButton("✅ Đúng", callback_data='correct'),
                InlineKeyboardButton("❌ Sai", callback_data='incorrect')]]
    await update.message.reply_text(message, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))

async def feedback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    if 'last_feature_vector' not in context.user_data or 'last_prediction' not in context.user_data:
        await query.message.reply_text("Không có dữ liệu phản hồi, gửi hình trước.")
        return
    label = 1 if context.user_data['last_prediction'] == "Tài" else 0 if query.data == 'correct' else 1
    await query.message.reply_text("Cảm ơn phản hồi! Bot sẽ học.")
    bot_data = context.application.bot_data
    bot_data.history_data.append(context.user_data['last_feature_vector'])
    bot_data.history_labels.append(label)
    model, scaler = train_model(bot_data.model, bot_data.scaler, np.array(bot_data.history_data), np.array(bot_data.history_labels))
    bot_data.model = model
    bot_data.scaler = scaler
    save_model_and_scaler(model, scaler)
    save_bot_data(bot_data)
    
def main() -> None:
    if not TOKEN:
        logging.error("TELEGRAM_BOT_TOKEN not found in environment variables.")
        return

    bot_data = load_bot_data()
    if not bot_data.model:
      input_size = 40
      model, scaler = load_or_create_model_and_scaler(input_size)
      bot_data.model = model
      bot_data.scaler = scaler
      save_bot_data(bot_data)

    application = Application.builder().token(TOKEN).bot_data(bot_data).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(Filters.photo, handle_image))
    application.add_handler(CallbackQueryHandler(feedback_handler, pattern='correct|incorrect'))

    application.run_polling()

if __name__ == '__main__':
    main()