import logging
import google.generativeai as genai
from openai import OpenAI
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import os
import io
import asyncio

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Get tokens from environment variables
TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Initialize models
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
openai_model = "gpt-4o"  # Or any other model

# Prompt for the bot
ENHANCED_CODING_PROMPT = """
Bạn là một trợ lý AI chuyên nghiệp, có khả năng tạo code chất lượng cao và hỗ trợ toàn diện. Bạn sẽ luôn luôn trả lời bằng tiếng Việt.
Hãy tập trung vào mục tiêu của người dùng. Code của bạn phải giải quyết trực tiếp vấn đề mà người dùng đang gặp phải.
Đảm bảo code bạn tạo ra phải là giải pháp tối ưu nhất. Hãy đưa ra code thông minh, logic, hiệu quả và dễ đọc, luôn luôn tìm cách để code có thể chạy được ngay.
Sử dụng tất cả các thông tin đã biết trong cuộc trò chuyện trước đó để tạo code.
"""

# A dictionary to store chat history for each user
user_chat_history = {}

async def start(update: Update, context: CallbackContext):
    """Handles the /start command."""
    user_name = update.effective_user.first_name
    await update.message.reply_text(
        f"Xin chào {user_name}! Tôi là bot AI, hãy gửi tin nhắn cho tôi để bắt đầu."
    )

async def clear_history(update: Update, context: CallbackContext):
    """Handles the /dl command to clear chat history."""
    user_id = update.effective_user.id
    if user_id in user_chat_history:
        del user_chat_history[user_id]
    await update.message.reply_text("Lịch sử trò chuyện đã được xóa.")

def create_code_file(code_content, user_id):
    """Creates a temporary file containing the code."""
    file_name = f"code_{user_id}.txt"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(code_content)
    return file_name


async def handle_message(update: Update, context: CallbackContext):
    """Handles incoming messages from users."""
    message = update.message.text
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name

    logger.info(f"Message from {user_name}: {message}")

    # Get or create user's chat history
    if user_id not in user_chat_history:
        user_chat_history[user_id] = []

    # Append user message to chat history
    user_chat_history[user_id].append(f"User: {message}")

    try:
        # Combine enhanced prompt, chat history, and user message
        all_contents = [ENHANCED_CODING_PROMPT] + user_chat_history[user_id] + [message]
         
        # Get responses from both Gemini and OpenAI concurrently
        gemini_response_future = asyncio.create_task(
            asyncio.to_thread(gemini_model.generate_content, contents=all_contents)
        )
        openai_response_future = asyncio.create_task(
            asyncio.to_thread(openai_client.chat.completions.create, model=openai_model, messages=[{"role":"user", "content": " ".join(all_contents)}])
        )


        # Wait for both API calls to complete
        gemini_response = await gemini_response_future
        openai_response = await openai_response_future

        gemini_text = gemini_response.text if gemini_response.text else ""
        openai_text = openai_response.choices[0].message.content if openai_response.choices else ""

        # Combine the responses intelligently
        combined_response = combine_responses(gemini_text, openai_text)

        if combined_response:
            # Check if the combined response contains code (heuristic - can be improved)
            if "```" in combined_response:
                code_blocks = combined_response.split("```")[1::2] # Extract code blocks

                # Create and send code files for each block
                for i, code in enumerate(code_blocks):
                   
                    code = code.strip()
                    
                    file_name = create_code_file(code, user_id)

                    with open(file_name, "rb") as f:
                        await update.message.reply_document(
                             document=InputFile(f, filename=f"code_{i+1}_{user_id}.txt"),
                                caption=f"Code được tạo cho {user_name}. Khối code {i+1}."
                             )
                   
                    os.remove(file_name) # Clean up the temp file

                 # Send remaining text that isn't code
                remaining_text = ""
                parts = combined_response.split("```")
                for i, part in enumerate(parts):
                     if i % 2 == 0:
                         remaining_text += part

                if remaining_text.strip():
                    await update.message.reply_text(f"{user_name}: {remaining_text}")


            else:
                await update.message.reply_text(f"{user_name}: {combined_response}")

             # Append bot response to the user's chat history
            user_chat_history[user_id].append(f"Bot: {combined_response}")

             # Limit history to 100 messages
            if len(user_chat_history[user_id]) > 100:
                user_chat_history[user_id] = user_chat_history[user_id][-100:]

        else:
            logger.warning(f"Both APIs returned empty responses.")
            await update.message.reply_text("Tôi xin lỗi, có vẻ như tôi không hiểu câu hỏi của bạn.")

    except Exception as e:
        logger.error(f"Error during API request: {e}", exc_info=True)
        await update.message.reply_text("Đã có lỗi xảy ra khi kết nối với AI. Xin vui lòng thử lại sau.")

def combine_responses(gemini_text, openai_text):
    """Intelligently combines responses from Gemini and OpenAI."""
    # Simple concatenation for now; can be enhanced to merge logically or remove duplicates
    combined =  f"**Gemini:**\n{gemini_text}\n\n**OpenAI:**\n{openai_text}"
    return combined

async def error(update: Update, context: CallbackContext):
    """Handles errors."""
    logger.warning(f"Update {update} caused error {context.error}", exc_info=True)


def main():
    """Initializes and runs the bot."""
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("dl", clear_history))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error)

    logger.info("Bot is running...")
    application.run_polling()

if __name__ == '__main__':
    main()