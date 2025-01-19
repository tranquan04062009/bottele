import logging
import google.generativeai as genai
from openai import OpenAI
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import os
import io
import requests # For handling images
from io import BytesIO # for reading image content

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
openai_model = "gpt-4-turbo-preview"  # Or any other model
openai_image_model = "dall-e-3" # Model for image generation

# Prompt for the bot
ENHANCED_CODING_PROMPT = """
Bạn là một trợ lý AI chuyên nghiệp, có khả năng tạo code chất lượng cao. Bạn sẽ luôn luôn trả lời bằng tiếng Việt.
Hãy tập trung vào mục tiêu của người dùng. Code của bạn phải giải quyết trực tiếp vấn đề mà người dùng đang gặp phải.
Đảm bảo code bạn tạo ra phải là giải pháp tối ưu nhất. Hãy đưa ra code thông minh, logic, hiệu quả và dễ đọc, luôn luôn tìm cách để code có thể chạy được ngay.
Sử dụng tất cả các thông tin đã biết trong cuộc trò chuyện trước đó để tạo code.
"""

# Prompt for image generation
IMAGE_GENERATION_PROMPT = """
Bạn là một trợ lý AI chuyên về tạo ảnh. Hãy tạo một bức ảnh theo yêu cầu của người dùng và miêu tả ngắn gọn nội dung bức ảnh đó.
Bạn sẽ luôn luôn trả lời bằng tiếng Việt.
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
    """Handles the /dl command to delete chat history."""
    user_id = update.effective_user.id
    if user_id in user_chat_history:
        del user_chat_history[user_id]
        await update.message.reply_text("Lịch sử chat của bạn đã được xóa.")
    else:
        await update.message.reply_text("Không có lịch sử chat nào để xóa.")

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
       
        # Determine which model to use (you can implement a more complex logic here)
        use_openai = "code" in message.lower() # Heuristic: Use OpenAI if "code" is in the message
        use_openai_image = "ảnh" in message.lower() or "hình" in message.lower() # Use DALL-E if image word in message

        if use_openai_image:
            image_prompt_content = [IMAGE_GENERATION_PROMPT] + [message]
            response = openai_client.images.generate(
              model=openai_image_model,
              prompt = " ".join(image_prompt_content),
              n=1, # Generate one image
              size="1024x1024"
             )
            
            image_url = response.data[0].url
            
            # Send the image via telegram
            response = requests.get(image_url)
            image =  BytesIO(response.content)
           
            await update.message.reply_photo(photo=image, caption="Ảnh được tạo bởi AI")

            # Append bot response to the user's chat history
            user_chat_history[user_id].append(f"Bot: Ảnh được tạo bởi AI")
            
        elif use_openai:
            response = openai_client.chat.completions.create(
              model=openai_model,
              messages=[{"role":"user", "content": " ".join(all_contents)}],
              )
            response_text = response.choices[0].message.content
        else:    
            # Use Gemini API with all the prompts and chat history
            response = gemini_model.generate_content(
                contents=all_contents
            )
            response_text = response.text

        if use_openai or not use_openai_image: #Avoid sending duplicate text response in image case
            if response_text:
                # Check if the response contains code (heuristic - can be improved)
                if "```" in response_text:
                    code_blocks = response_text.split("```")[1::2] # Extract code blocks

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
                    parts = response_text.split("```")
                    for i, part in enumerate(parts):
                        if i % 2 == 0:
                            remaining_text += part

                    if remaining_text.strip():
                        await update.message.reply_text(f"{user_name}: {remaining_text}")


                else:
                    await update.message.reply_text(f"{user_name}: {response_text}")

            # Append bot response to the user's chat history
            user_chat_history[user_id].append(f"Bot: {response_text}")

        

        # Limit history to 100 messages
        if len(user_chat_history[user_id]) > 100:
            user_chat_history[user_id] = user_chat_history[user_id][-100:]

    except Exception as e:
        logger.error(f"Error during API request: {e}", exc_info=True)
        await update.message.reply_text("Đã có lỗi xảy ra khi kết nối với AI. Xin vui lòng thử lại sau.")


async def error(update: Update, context: CallbackContext):
    """Handles errors."""
    logger.warning(f"Update {update} caused error {context.error}", exc_info=True)


def main():
    """Initializes and runs the bot."""
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("dl", clear_history)) # Add clear history command
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error)

    logger.info("Bot is running...")
    application.run_polling()

if __name__ == '__main__':
    main()