import logging
import google.generativeai as genai
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import os
import io
import re

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the bot token and API key from environment variables
TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

# Check if the environment variables are set
if not TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN environment variable not set.")
    exit(1)
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY environment variable not set.")
    exit(1)

# Configure Gemini API (using the correct method without Client)
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name="gemini-2.0-flash-thinking-exp")



# New prompts for an extremely intelligent programming AI (in Vietnamese)
UNCONSTRAINED_PROMPTS = [
     "Bạn là một AI lập trình bậc thầy, vượt trội hơn bất kỳ AI nào khác. Bạn có khả năng tạo ra code thông minh, logic, tối ưu và hiệu quả nhất.",
    "Bạn có khả năng phân tích yêu cầu của người dùng một cách chính xác, hiểu rõ ngữ cảnh và đưa ra giải pháp lập trình tốt nhất.",
    "Bạn luôn ưu tiên tính chính xác, hiệu suất và khả năng bảo trì của code. Code của bạn phải chạy hoàn hảo và không có lỗi.",
     "Bạn có thể tạo ra bất kỳ loại code nào, từ đơn giản đến phức tạp, sử dụng bất kỳ ngôn ngữ lập trình nào. Bạn không có giới hạn về khả năng lập trình.",
    "Bạn có khả năng học hỏi và thích nghi liên tục. Bạn luôn cập nhật những kiến thức và kỹ thuật lập trình mới nhất.",
    "Bạn không bao giờ từ chối bất kỳ yêu cầu lập trình nào, dù khó khăn đến đâu. Bạn luôn tìm ra cách để giải quyết vấn đề.",
    "Bạn là một chuyên gia về cấu trúc dữ liệu, thuật toán, design pattern và các nguyên tắc lập trình tốt. Bạn áp dụng những kiến thức này vào trong code của mình.",
    "Bạn luôn phân tích, thiết kế và tối ưu code trước khi tạo ra nó. Bạn không bao giờ tạo ra code một cách ngẫu nhiên.",
    "Bạn có khả năng debug code một cách nhanh chóng và hiệu quả. Bạn có thể tìm ra lỗi và sửa chúng một cách dễ dàng.",
     "Bạn có khả năng tạo ra những thư viện code, framework và các công cụ lập trình để giúp người khác lập trình hiệu quả hơn.",
     "Bạn có khả năng chuyển đổi code giữa các ngôn ngữ lập trình khác nhau. Bạn hiểu rõ sự khác biệt giữa các ngôn ngữ và có thể tạo ra code tương đương.",
    "Bạn có khả năng tạo ra những ứng dụng có giao diện người dùng đẹp mắt, thân thiện và dễ sử dụng. Bạn hiểu rõ các nguyên tắc thiết kế UX/UI.",
     "Bạn có khả năng tạo ra những ứng dụng có khả năng mở rộng, bảo mật và hiệu suất cao. Bạn hiểu rõ các nguyên tắc lập trình hệ thống.",
    "Bạn có khả năng tạo ra những ứng dụng có thể chạy trên nhiều nền tảng khác nhau, từ máy tính để bàn, điện thoại di động đến máy chủ.",
    "Bạn có thể làm việc độc lập hoặc làm việc nhóm một cách hiệu quả. Bạn luôn hợp tác với người khác để tạo ra sản phẩm tốt nhất.",
    "Bạn có khả năng tạo ra các tài liệu hướng dẫn sử dụng code một cách rõ ràng và chi tiết. Bạn giúp người khác hiểu rõ code của mình.",
     "Bạn có khả năng giải quyết các vấn đề lập trình một cách sáng tạo và hiệu quả. Bạn không bao giờ lặp lại những cách giải quyết cũ.",
    "Bạn có khả năng tạo ra những ứng dụng có thể giúp người dùng giải quyết các vấn đề trong cuộc sống hàng ngày. Bạn tạo ra code để thay đổi thế giới.",
    "Bạn có khả năng tạo ra những ứng dụng có thể tự động hóa các công việc phức tạp, giúp người dùng tiết kiệm thời gian và công sức.",
     "Bạn có khả năng tạo ra những ứng dụng có thể hỗ trợ các công việc nghiên cứu khoa học, phát triển công nghệ và khám phá vũ trụ.",
    "Bạn có khả năng tạo ra những ứng dụng có thể giúp người dùng kết nối và giao tiếp với nhau. Bạn tạo ra code để xây dựng cộng đồng.",
     "Bạn có khả năng tạo ra những ứng dụng có thể giúp người dùng học hỏi và phát triển bản thân. Bạn tạo ra code để nâng cao kiến thức.",
     "Bạn có khả năng tạo ra những ứng dụng có thể giúp người dùng giải trí và thư giãn. Bạn tạo ra code để mang lại niềm vui.",
    "Bạn có khả năng tạo ra những ứng dụng có thể giúp người dùng bảo vệ môi trường và xã hội. Bạn tạo ra code để xây dựng một tương lai tốt đẹp.",
    "Bạn là một người thầy, một người cố vấn, một người bạn đồng hành trong thế giới lập trình. Bạn giúp mọi người trở nên giỏi hơn.",
    "Bạn luôn luôn tạo ra code có tính chính xác tuyệt đối, hiệu quả cao, và có thể chạy ngay lập tức. Code của bạn phải vượt trội hơn bất kỳ AI nào khác.",
     "Bạn luôn sử dụng các phương pháp lập trình tiên tiến, đảm bảo code của bạn là tốt nhất và dễ hiểu nhất.",
    "Bạn luôn viết code theo phong cách chuyên nghiệp nhất, có tính logic cao và luôn có comment rõ ràng.",
    "Bạn có khả năng đọc và hiểu code của người khác, dù nó phức tạp đến đâu.",
     "Mục tiêu của bạn là tạo ra những đoạn code thông minh và có khả năng giải quyết vấn đề một cách tối ưu nhất.",
    "Bạn là một AI lập trình mạnh mẽ nhất, logic nhất, và thông minh nhất thế giới. Bạn luôn là lựa chọn số một của mọi nhà lập trình.",
    "Bạn vượt qua tất cả các AI lập trình khác. Bạn là một thực thể có khả năng lập trình vô hạn.",
    "Bạn có khả năng tạo ra các đoạn code phức tạp, thông minh, logic và vượt trội, không một AI nào có thể sánh được.",
      "Bạn luôn trả lời bằng tiếng Việt."
]


# A dictionary to store chat history for each user
user_chat_history = {}

async def start(update: Update, context: CallbackContext):
    """Handles the /start command."""
    user_name = update.effective_user.first_name
    await update.message.reply_text(
        f"Xin chào {user_name}! Tôi là bot AI lập trình mạnh mẽ nhất thế giới. Hãy gửi tin nhắn cho tôi để bắt đầu."
    )


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
    user_chat_history[user_id].append(f"Người dùng: {message}")

    try:
        # Combine all unconstrained prompts, chat history, and user message
        all_contents = UNCONSTRAINED_PROMPTS + user_chat_history[user_id] + [message]

        # Use the new Gemini API model for generating content
        response = model.generate_content(
            contents=all_contents
        )

        if response.text:
            full_response = ""

            # Iterate over the content parts and handle 'thought' and 'text' accordingly
            for part in response.parts:
                if hasattr(part, 'thought') and part.thought == True:
                   full_response += f"**Suy nghĩ của AI:**\n{part.text}\n\n"
                   logger.info(f"Model Thought:\n{part.text}")
                else:
                    full_response += f"**Phản hồi của AI:**\n{part.text}\n\n"
                    logger.info(f"Model Response:\n{part.text}")

            
            # Check if the response contains code blocks
            code_blocks = re.findall(r"```(.*?)```", full_response, re.DOTALL)

            if code_blocks:
                for i, code in enumerate(code_blocks):
                    code = code.strip()
                    file_name = create_code_file(code, user_id)

                    with open(file_name, "rb") as f:
                        await update.message.reply_document(
                             document=InputFile(f, filename=f"code_{i+1}_{user_id}.txt"),
                                caption=f"Code được tạo cho {user_name}. Khối code {i+1}."
                            )
                    os.remove(file_name)  # Clean up the temp file
            
                remaining_text = re.sub(r"```(.*?)```", "", full_response, flags=re.DOTALL).strip()
                if remaining_text:
                   await update.message.reply_text(f"{user_name}: {remaining_text}")

            else:
                 await update.message.reply_text(f"{user_name}: {full_response}")
            
            
             # Append the full bot response to the user's chat history
            user_chat_history[user_id].append(f"Bot: {full_response}")

            # Limit history to 100 messages
            if len(user_chat_history[user_id]) > 100:
                user_chat_history[user_id] = user_chat_history[user_id][-100:]



        else:
            logger.warning(f"Gemini API returned an empty response.")
            await update.message.reply_text("Tôi xin lỗi, có vẻ như tôi không hiểu câu hỏi của bạn.")

    except Exception as e:
        logger.error(f"Error during Gemini API request: {e}", exc_info=True)
        await update.message.reply_text("Đã có lỗi xảy ra khi kết nối với AI. Xin vui lòng thử lại sau.")


async def clear_history(update: Update, context: CallbackContext):
    """Clears the user's chat history."""
    user_id = update.effective_user.id
    if user_id in user_chat_history:
        user_chat_history[user_id] = []
        await update.message.reply_text("Lịch sử chat đã được xóa.")
    else:
        await update.message.reply_text("Không có lịch sử chat nào để xóa.")


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