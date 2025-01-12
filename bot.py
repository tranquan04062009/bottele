from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
import os
import random
import requests
import secrets
import asyncio
from dotenv import load_dotenv
from tqdm import tqdm  # Thư viện thanh tiến trình

# Load environment variables
load_dotenv()
TOKEN = os.getenv('BOT_TOKEN')

# Initialize bot and dispatcher
bot = Bot(token=TOKEN)
dp = Dispatcher()
dp.bot = bot  # Gán bot cho dp

# Globals for session management
user_spam_sessions = {}  # {chat_id: [{session_id, username, message, is_active}]}
blocked_users = set()  # Users who are blocked

# Constants
SPAM_RATE_LIMIT = 2  # Seconds between spam messages
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/109.0",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) Gecko/20100101 Firefox/31.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:58.0) Gecko/20100101 Firefox/58.0"
]

# Headers for requests
HEADERS = {
    "User-Agent": random.choice(USER_AGENTS),
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
}

async def send_message(session: dict, chat_id: int):
    """Send spam messages in a loop and show progress."""
    counter = 0
    session_id = session["session_id"]
    total_spams = 100  # You can define a fixed number or use a dynamic value
    pbar = tqdm(total=total_spams, desc=f"Spam phiên {session_id}", position=0, leave=True)

    while session["is_active"] and counter < total_spams:
        try:
            # Generate random device ID
            device_id = secrets.token_hex(21)
            url = "https://ngl.link/api/submit"
            body = f"username={session['username']}&question={session['message']}&deviceId={device_id}&gameSlug=&referrer="

            # Randomly select user-agent to avoid detection
            headers = HEADERS
            headers["User-Agent"] = random.choice(USER_AGENTS)

            # Send POST request
            response = requests.post(url, headers=headers, data=body)
            if response.status_code != 200:
                await bot.send_message(chat_id, f"[Lỗi] Phiên {session_id}: Bị giới hạn, chờ 5 giây...")
                await asyncio.sleep(5)
                continue

            # Update spam counter and progress bar
            counter += 1
            pbar.update(1)  # Update progress bar
            await bot.send_message(chat_id, f"Phiên {session_id}: Đã gửi {counter} tin nhắn.")

            await asyncio.sleep(SPAM_RATE_LIMIT)

        except requests.exceptions.RequestException as e:
            print(f"[Lỗi] {e}")
            await bot.send_message(chat_id, f"[Lỗi] Không thể gửi tin nhắn: {e}")
            await asyncio.sleep(2)

    pbar.close()  # Close progress bar when done
    # End of session
    await bot.send_message(chat_id, f"Phiên {session_id} đã kết thúc. Tổng tin nhắn đã gửi: {counter}")


@dp.message_handler(commands=["start"])
async def start_command(message: types.Message):
    """Handle /start command."""
    chat_id = message.chat.id
    if chat_id in blocked_users:
        await message.reply("Bạn đã bị chặn khỏi việc sử dụng bot này.")
        return

    # Initialize spam session for new users
    user_spam_sessions.setdefault(chat_id, [])
    await message.reply(
        "Chào mừng! Sử dụng các nút bên dưới để bắt đầu:",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[[KeyboardButton("Bắt đầu Spam"), KeyboardButton("Danh sách Spam")]],
            resize_keyboard=True,
        ),
    )


@dp.message_handler(Text(equals="Bắt đầu Spam"))
async def handle_start_spam(message: types.Message):
    """Handle 'Bắt đầu Spam'."""
    chat_id = message.chat.id
    if chat_id in blocked_users:
        await message.reply("Bạn đã bị chặn khỏi việc sử dụng bot này.")
        return

    await message.reply("Nhập tên người dùng để spam:")

    # Wait for username
    username = (await dp.bot.wait_for("message", timeout=60)).text
    await message.reply("Nhập nội dung tin nhắn:")
    text = (await dp.bot.wait_for("message", timeout=60)).text

    # Start spam session
    session_id = len(user_spam_sessions[chat_id]) + 1
    new_session = {"session_id": session_id, "username": username, "message": text, "is_active": True}
    user_spam_sessions[chat_id].append(new_session)

    await message.reply(f"Phiên spam {session_id} đã bắt đầu!")
    asyncio.create_task(send_message(new_session, chat_id))


@dp.message_handler(Text(equals="Danh sách Spam"))
async def handle_list_spam(message: types.Message):
    """Handle 'Danh sách Spam'."""
    chat_id = message.chat.id
    if chat_id in blocked_users:
        await message.reply("Bạn đã bị chặn khỏi việc sử dụng bot này.")
        return

    sessions = user_spam_sessions.get(chat_id, [])
    if not sessions:
        await message.reply("Không có phiên spam nào đang hoạt động.")
        return

    # List active sessions
    buttons = []
    for session in sessions:
        buttons.append(
            InlineKeyboardButton(f"Dừng phiên {session['session_id']}", callback_data=f"stop_{session['session_id']}")
        )

    await message.reply(
        "Danh sách các phiên spam hiện tại:",
        reply_markup=InlineKeyboardMarkup([buttons]),
    )


@dp.callback_query_handler(lambda c: c.data.startswith("stop_"))
async def handle_stop_spam(callback_query: types.CallbackQuery):
    """Handle stop spam session."""
    chat_id = callback_query.message.chat.id
    session_id = int(callback_query.data.split("_")[1])

    sessions = user_spam_sessions.get(chat_id, [])
    session = next((s for s in sessions if s["session_id"] == session_id), None)

    if session:
        session["is_active"] = False
        await bot.send_message(chat_id, f"Phiên spam {session_id} đã bị dừng.")
    else:
        await bot.send_message(chat_id, f"Không tìm thấy phiên spam với ID {session_id}.")


if __name__ == "__main__":
    print("Bot đang chạy...")
    # Use dp.start_polling() instead of executor
    asyncio.run(dp.start_polling())