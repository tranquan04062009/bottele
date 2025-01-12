import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import re
from typing import Dict, Optional
import logging
from datetime import datetime

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Conversation states
COOKIE, TARGET_ID, TIMEOUT, SPEED = range(4)

class FacebookApiVIP:
    def __init__(self, cookie: str):
        self.cookie = cookie
        self.user_id = self._extract_user_id(cookie)
        self.headers = self._generate_headers(cookie)
        self.session = self._create_session()
        self.fb_dtsg: Optional[str] = None
        self.jazoest: Optional[str] = None
        
    def _extract_user_id(self, cookie: str) -> str:
        try:
            return cookie.split('c_user=')[1].split(';')[0]
        except IndexError:
            raise ValueError("Invalid cookie: c_user not found")

    def _generate_headers(self, cookie: str) -> Dict[str, str]:
        return {
            'authority': 'mbasic.facebook.com',
            'cache-control': 'max-age=0',
            'sec-ch-ua': '"Google Chrome";v="120"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'upgrade-insecure-requests': '1',
            'origin': 'https://mbasic.facebook.com',
            'content-type': 'application/x-www-form-urlencoded',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-dest': 'document',
            'referer': 'https://mbasic.facebook.com/',
            'accept-language': 'en-US,en;q=0.9',
            'cookie': cookie
        }

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    async def get_user_info(self) -> tuple[Optional[str], Optional[str]]:
        if self.fb_dtsg and self.jazoest:
            return self.fb_dtsg, self.jazoest

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.get('https://mbasic.facebook.com/profile.php', headers=self.headers, timeout=10)
            )
            html = response.text
            
            self.fb_dtsg = re.search(r'name="fb_dtsg" value="(.*?)"', html).group(1)
            self.jazoest = re.search(r'name="jazoest" value="(.*?)"', html).group(1)
            name = re.search(r'<title>(.*?)</title>', html).group(1)
            
            return name, self.user_id
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            return None, None

    async def report_user(self, target_user_id: str, timeout: int, speed: float) -> list[str]:
        if not self.fb_dtsg or not self.jazoest:
            await self.get_user_info()

        reasons = {
            'spam': 'Spam',
            'violation': 'Community Standards Violation',
            'hate_speech': 'Hate Speech',
            'pornography': 'Adult Content',
            'harassment': 'Harassment',
            'impersonation': 'Fake Account',
            'personal_attack': 'Personal Attack',
            'terrorism': 'Terrorism',
            'violence': 'Violence',
            'intellectual_property': 'IP Violation',
            'false_information': 'Misinformation'
        }

        results = []
        for reason_code, reason_description in reasons.items():
            data = {
                'av': self.user_id,
                '__user': self.user_id,
                'fb_dtsg': self.fb_dtsg,
                'jazoest': self.jazoest,
                'target_user_id': target_user_id,
                'report_type': 'user',
                'reason': reason_code,
                'action': 'report'
            }
            
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.post(
                        'https://www.facebook.com/report/user',
                        headers=self.headers,
                        data=data,
                        timeout=timeout
                    )
                )
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                if response.status_code == 200:
                    msg = f"✅ [{timestamp}] Report sent: {reason_description}"
                else:
                    msg = f"❌ [{timestamp}] Failed: {reason_description} (Status: {response.status_code})"
                
                results.append(msg)
                await asyncio.sleep(speed)  # Control report speed
                
            except Exception as e:
                results.append(f"❌ Error with {reason_description}: {str(e)}")

        return results

class ReportBot:
    def __init__(self, token: str):
        self.application = Application.builder().token(token).build()
        self.fb_api: Optional[FacebookApiVIP] = None
        self.setup_handlers()

    def setup_handlers(self):
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler('start', self.start)],
            states={
                COOKIE: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.receive_cookie)],
                TARGET_ID: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.receive_target_id)],
                TIMEOUT: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.receive_timeout)],
                SPEED: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.receive_speed)]
            },
            fallbacks=[CommandHandler('cancel', self.cancel)]
        )

        self.application.add_handler(conv_handler)
        self.application.add_handler(CommandHandler("help", self.help_command))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        await update.message.reply_text(
            "🤖 Welcome to FB Report Bot!\n\n"
            "Please send your Facebook cookie (starting with 'c_user=')"
        )
        return COOKIE

    async def receive_cookie(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        cookie = update.message.text
        try:
            self.fb_api = FacebookApiVIP(cookie)
            name, user_id = await self.fb_api.get_user_info()
            
            if name and user_id:
                await update.message.reply_text(
                    f"✅ Successfully logged in as:\nName: {name}\nID: {user_id}\n\n"
                    "Please enter the target Facebook ID to report:"
                )
                return TARGET_ID
            else:
                await update.message.reply_text("❌ Invalid cookie. Please try again with /start")
                return ConversationHandler.END
                
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}\nPlease try again with /start")
            return ConversationHandler.END

    async def receive_target_id(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        context.user_data['target_id'] = update.message.text
        await update.message.reply_text(
            "Enter timeout value in seconds (5-30 recommended):"
        )
        return TIMEOUT

    async def receive_timeout(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        try:
            timeout = int(update.message.text)
            if timeout < 1:
                raise ValueError("Timeout must be positive")
            context.user_data['timeout'] = timeout
            await update.message.reply_text(
                "Enter speed between reports in seconds (0.5-3.0 recommended):"
            )
            return SPEED
        except ValueError:
            await update.message.reply_text("Please enter a valid number")
            return TIMEOUT

    async def receive_speed(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        try:
            speed = float(update.message.text)
            if speed < 0:
                raise ValueError("Speed must be positive")
                
            await update.message.reply_text("⏳ Starting report process...")
            
            results = await self.fb_api.report_user(
                context.user_data['target_id'],
                context.user_data['timeout'],
                speed
            )
            
            # Send results in chunks to avoid message length limits
            chunk_size = 10
            for i in range(0, len(results), chunk_size):
                chunk = results[i:i + chunk_size]
                await update.message.reply_text("\n".join(chunk))
            
            await update.message.reply_text(
                "✅ Report process completed!\n"
                "Use /start to report another account"
            )
            
            return ConversationHandler.END
            
        except ValueError:
            await update.message.reply_text("Please enter a valid number")
            return SPEED

    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        await update.message.reply_text("Operation cancelled. Use /start to begin again.")
        return ConversationHandler.END

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        help_text = (
            "📖 Bot Commands:\n\n"
            "/start - Start the report process\n"
            "/cancel - Cancel current operation\n"
            "/help - Show this help message\n\n"
            "🔹 How to use:\n"
            "1. Start with /start command\n"
            "2. Enter your Facebook cookie\n"
            "3. Enter target Facebook ID\n"
            "4. Set timeout (5-30 seconds)\n"
            "5. Set speed (0.5-3.0 seconds)\n\n"
            "⚠️ Use responsibly and in accordance with Facebook's terms of service."
        )
        await update.message.reply_text(help_text)

    def run(self):
        self.application.run_polling()

def main():
    # Replace 'YOUR_BOT_TOKEN' with your Telegram bot token
    bot = ReportBot('YOUR_BOT_TOKEN')
    bot.run()

if __name__ == '__main__':
    main()