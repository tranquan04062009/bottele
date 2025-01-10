import logging
import os
import uuid
from typing import List, Tuple, Union, Optional
from dataclasses import dataclass, field
from io import BytesIO
import socket
from contextlib import suppress
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from PIL import Image
import pytesseract
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
from torch.utils.data import Dataset, DataLoader
from random import randint, choice


# Conditional import for cv2, it will set cv2 = None
try:
    import cv2
    logging.info("OpenCV available")
except ImportError as e:
    logging.error(f"OpenCV not available: {e}. Image enhance methods will be unavailable.")
    cv2 = None


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
  model: Optional[nn.Module] = None
  scaler: Optional[StandardScaler] = None
    

def load_bot_data() -> BotData:
  if os.path.exists(BOT_DATA_PATH):
    try:
      with open(BOT_DATA_PATH, 'rb') as f:
        logging.info("Loaded bot data from file")
        return pickle.load(f)
    except (IOError, OSError, pickle.PickleError) as e:
          logging.error(f"Error loading bot data: {e}, using default values")
          return BotData()
  else:
      logging.info("Creating new bot data as no file exists.")
      return BotData()

def save_bot_data(bot_data: BotData):
  try:
      with open(BOT_DATA_PATH, 'wb') as f:
          pickle.dump(bot_data, f)
      logging.info("Saved bot data to file.")
  except (IOError, OSError, pickle.PickleError) as e:
      logging.error(f"Error saving bot data to file {e}")


def download_image(url: str, folder: str) -> Optional[str]:
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{uuid.uuid4()}.jpg")
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return file_path
    except requests.exceptions.RequestException as e:
      logging.error(f"Error downloading image: {e}")
      return None
    except (IOError, OSError) as e:
         logging.error(f"Error saving file {file_path} : {e}")
         return None
       

def enhance_image(path: str) -> Optional[str]:
  if not cv2:
    try:
      img = Image.open(path)
      img = img.convert('L').resize((500,500), Image.Resampling.LANCZOS)
      enhanced_path = path[:-4] + "_enhanced" + path[-4:]
      img.save(enhanced_path)
      return enhanced_path
    except Exception as e:
        logging.error(f"Pillow fallback image enhancement Error {e}")
        return path

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
  except (IOError, OSError, cv2.error) as e:
      logging.error(f"Error enhancing image: {e}")
      return None


def extract_numbers_from_image(path: str) -> Optional[List[int]]:
    enhanced_path = enhance_image(path)
    if not enhanced_path:
        return None
    try:
        img = Image.open(enhanced_path).convert('L')
        text = pytesseract.image_to_string(img, config="--psm 6")
        return [int(s) for s in text.split() if s.isdigit()]
    except (IOError, OSError, pytesseract.TesseractError) as e:
        logging.error(f"Error OCR: {e}")
        return None
    except Exception as e: #catch other generic error related to Image or Tesseract call
      logging.error(f"Unexpected Error extract_numbers_from_image {e}")
      return None

def analyze_path(path: str) -> Optional[List[int]]:
  numbers = extract_numbers_from_image(path)
  if not numbers:
    return None
  if len(numbers) == 1:
      return [0]
  return [1 if numbers[i+1] > numbers[i] else -1 if numbers[i+1] < numbers[i] else 0 for i in range(len(numbers) - 1)]

def create_feature_vector(numbers: List[int], path: List[int]) -> np.ndarray:
    padded_numbers = numbers + [0] * (20 - len(numbers)) if len(numbers) < 20 else numbers[:20]
    padded_path = path + [0] * (20 - len(path)) if len(path) < 20 else path[:20]
    return np.array(padded_numbers + padded_path)

def validate_input_type(arg: any, expected_type: any, arg_name: str) -> None:
    if not isinstance(arg, expected_type):
       raise TypeError(f"{arg_name} must be {expected_type}, but get {type(arg)}")


class TaiXiuDataset(Dataset):
    def __init__(self, X: np.ndarray, y: List[int]):
        validate_input_type(X, np.ndarray, "X")
        validate_input_type(y, list, "y")

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

class TaiXiuPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout_rate: float = 0.2):
      super().__init__()
      validate_input_type(input_size, int, "input_size")
      validate_input_type(hidden_size, int, "hidden_size")
      validate_input_type(num_layers, int, "num_layers")
      validate_input_type(output_size, int, "output_size")
      validate_input_type(dropout_rate, float, "dropout_rate")
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
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        model = torch.load(MODEL_PATH)
        logging.info("Loaded model and scaler from file")
        return model, scaler
    except (IOError, OSError, pickle.PickleError, RuntimeError) as e:
      logging.error(f"Error loading model or scaler, creating a new one {e}")
  hidden_size = 128
  num_layers = 2
  output_size = 2
  model = TaiXiuPredictor(input_size, hidden_size, num_layers, output_size, dropout_rate=0.3)
  scaler = StandardScaler()
  logging.info("Created new model and scaler")
  return model, scaler
  

def save_model_and_scaler(model: nn.Module, scaler: StandardScaler):
  try:
      with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
      torch.save(model, MODEL_PATH)
      logging.info("Saved model and scaler to file")
  except (IOError, OSError, pickle.PickleError, RuntimeError) as e:
      logging.error(f"Error saving model and scaler {e}")

def predict_next_outcome(model, scaler, path) -> Tuple[Optional[str], Optional[float]]:
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
  try:
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
    except Exception as e:
         logging.error(f"Unexpected exception when handle image: {e}")
         await update.message.reply_text("Lỗi không xác định khi xử lý ảnh. Xin thử lại.")


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
  model, scaler = train_model(bot_data.model, bot_data.scaler, np.array(bot_data.history_data), bot_data.history_labels)
  bot_data.model = model
  bot_data.scaler = scaler
  save_model_and_scaler(model, scaler)
  save_bot_data(bot_data)


def prepare_training_data(num_samples:int):
    """Generates simulated training data."""
    numbers_list = []
    labels = []
    for i in range(num_samples):
      sequence = []
      current_number = randint(3,18)
      sequence.append(current_number)
      current_number = randint(3,18)
      sequence.append(current_number)
      for _ in range(randint(2, 20)): #Random length
          delta = choice([-1, 0, 1]) #random move in history graph up/down/stay
          current_number += delta

          if current_number <=3:
             current_number += choice([1, 2]) #force it into acceptable bound
          if current_number >=18:
            current_number -= choice([1,2])

          sequence.append(current_number)

      label = 1 if current_number > 10 else 0  #1 is Tai and 0 is Xiu
      labels.append(label)
      numbers_list.append(sequence)
        
    X_data = []
    for numbers in numbers_list:
        path = analyze_path(None, numbers)
        if not path:
           continue
        X_data.append(create_feature_vector(numbers, path) )
        
    return np.array(X_data, dtype=object), np.array(labels, dtype=object)



def main() -> None:
  if not TOKEN:
        logging.error("TELEGRAM_BOT_TOKEN not found in environment variables.")
        return

  bot_data = load_bot_data()
  if not bot_data.model:
      X, y = prepare_training_data(num_samples=500) #create more training sample with generated function.
      input_size = 40
      model, scaler = load_or_create_model_and_scaler(input_size)
      model, scaler = train_model(model, scaler, X, y)

      bot_data.model = model
      bot_data.scaler = scaler
      save_bot_data(bot_data)
  application = Application.builder().token(TOKEN).build()
  application.bot_data = bot_data

  application.add_handler(CommandHandler("start", start))
  application.add_handler(MessageHandler(filters.PHOTO, handle_image))
  application.add_handler(CallbackQueryHandler(feedback_handler, pattern='correct|incorrect'))
    
  application.run_polling()

if __name__ == '__main__':
  main()