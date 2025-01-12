import asyncio
import logging
import re
import json
import os
import secrets
import signal
from typing import Dict, Optional, Any, Tuple, Callable, List
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)
import aiohttp
from dataclasses import dataclass, field
from cryptography.fernet import Fernet
import sqlite3
from pydantic import BaseModel, validator, ValidationError


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


# Load configuration
def load_config() -> dict:
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
            return config
    except FileNotFoundError:
        logger.error(
            "config.json not found. Ensure it exists in the same folder as the script."
        )
        exit()
    except json.JSONDecodeError:
        logger.error("Error decoding config.json. Please check that its JSON syntax is correct.")
        exit()


config = load_config()

TELEGRAM_TOKEN = config.get("telegram_token")
if not TELEGRAM_TOKEN:
    logger.error("Telegram token not found in config.json")
    exit()


# Encryption Key (use a proper method for secure key management)
ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    ENCRYPTION_KEY = secrets.token_urlsafe(32)
    logger.warning(
        "No encryption key found in environment. Using randomly generated key. THIS IS NOT SECURE FOR PRODUCTION!"
    )
fernet = Fernet(ENCRYPTION_KEY)


# Custom Exceptions
class FacebookApiException(Exception):
    """Base class for exceptions related to Facebook API."""

    pass


class InvalidCookieError(FacebookApiException):
    """Exception raised when an invalid Facebook cookie is used."""

    pass


class UserInfoRetrievalError(FacebookApiException):
    """Exception raised if user info cannot be retrieved."""

    pass


class ReportError(FacebookApiException):
    """Exception raised during reporting a user."""

    pass


class TelegramBotError(Exception):
    """Base class for exceptions in Telegram Bot."""

    pass


class InvalidInputError(TelegramBotError):
    """Exception raised when invalid input is provided to telegram bot"""

    pass


# Data validation with Pydantic
class UserInput(BaseModel):
    target_id: str
    timeout: int

    @validator("target_id")
    def validate_target_id(cls, value):
        if not value.isdigit():
            raise ValueError("Invalid target ID format. Please enter a numerical ID.")
        return value

    @validator("timeout")
    def validate_timeout(cls, value):
        if value <= 0:
            raise ValueError("Timeout must be a positive number")
        return value


@dataclass
class UserState:
    state: str = "idle"
    cookie: Optional[str] = None
    target_id: Optional[str] = None
    timeout: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "UserState":
        return cls(**data)

    def to_dict(self) -> Dict:
        return {
            "state": self.state,
            "cookie": self.cookie,
            "target_id": self.target_id,
            "timeout": self.timeout,
        }


# Database setup
def create_connection():
    conn = sqlite3.connect("user_states.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS user_states (
            user_id INTEGER PRIMARY KEY,
            state TEXT,
            cookie TEXT,
            target_id TEXT,
            timeout INTEGER
        )
    """
    )
    conn.commit()
    return conn


def save_user_state_db(conn: sqlite3.Connection, user_id: int, user_state: UserState):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO user_states (user_id, state, cookie, target_id, timeout) VALUES (?, ?, ?, ?, ?)",
        (user_id, user_state.state, user_state.cookie, user_state.target_id, user_state.timeout),
    )
    conn.commit()


def load_user_state_db(
    conn: sqlite3.Connection, user_id: int
) -> Optional[UserState]:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT state, cookie, target_id, timeout FROM user_states WHERE user_id = ?",
        (user_id,),
    )
    result = cursor.fetchone()
    if result:
        state, cookie, target_id, timeout = result
        return UserState(
            state=state, cookie=cookie, target_id=target_id, timeout=timeout
        )
    return None


# User States Dictionary
user_states: Dict[int, UserState] = {}


def encrypt_cookie(cookie: str) -> str:
    try:
        return fernet.encrypt(cookie.encode()).decode()
    except Exception as e:
        logger.error(f"Error during cookie encryption: {e}")
        raise FacebookApiException(f"Cookie encryption failed: {e}") from e


def decrypt_cookie(encrypted_cookie: str) -> str:
    try:
        return fernet.decrypt(encrypted_cookie.encode()).decode()
    except Exception as e:
        logger.error(f"Error during cookie decryption: {e}")
        raise FacebookApiException(f"Cookie decryption failed: {e}") from e


class FacebookApiVIP:
    def __init__(self, cookie: str):
        self.cookie: str = cookie
        try:
            self.user_id: str = self._extract_user_id(cookie)
        except ValueError as e:
            raise InvalidCookieError("Failed to extract user ID from the cookie") from e
        self.headers: dict = self._generate_headers(cookie)
        self.fb_dtsg: Optional[str] = None
        self.jazoest: Optional[str] = None
        self.session: Optional[aiohttp.ClientSession] = None

    def _extract_user_id(self, cookie: str) -> str:
        try:
            return cookie.split("c_user=")[1].split(";")[0]
        except IndexError:
            logger.error("Cannot get user_id from cookie: c_user not found in cookie")
            raise ValueError("Invalid cookie format")

    def _generate_headers(self, cookie: str) -> dict:
        return {
            "authority": "mbasic.facebook.com",
            "cache-control": "max-age=0",
            "sec-ch-ua": '"Google Chrome";v="93", " Not;A Brand";v="99", "Chromium";v="93"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "upgrade-insecure-requests": "1",
            "origin": "https://mbasic.facebook.com",
            "content-type": "application/x-www-form-urlencoded",
            "user-agent": "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36",
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "sec-fetch-site": "same-origin",
            "sec-fetch-mode": "navigate",
            "sec-fetch-user": "?1",
            "sec-fetch-dest": "document",
            "referer": "https://mbasic.facebook.com/",
            "accept-language": "en-US,en;q=0.9",
            "cookie": cookie,
        }

    async def create_session(self) -> None:
        self.session = aiohttp.ClientSession(headers=self.headers)

    async def close_session(self) -> None:
        if self.session:
            await self.session.close()

    async def get_thongtin(self) -> Tuple[str, str]:
        if self.fb_dtsg and self.jazoest:
            logger.info("Using cached data.")
            return self.fb_dtsg, self.jazoest

        try:
            async with self.session.get(
                "https://mbasic.facebook.com/profile.php", timeout=10
            ) as response:
                home = await response.text()
                if not home:
                    raise UserInfoRetrievalError("Empty response from Facebook")
                self.fb_dtsg = re.search(r'name="fb_dtsg" value="(.*?)"', home).group(
                    1
                )
                self.jazoest = re.search(r'name="jazoest" value="(.*?)"', home).group(
                    1
                )
                ten = re.search(r"<title>(.*?)</title>", home).group(1)
                logger.info(f"Facebook name: {ten}, ID: {self.user_id}")
                return ten, self.user_id
        except AttributeError as e:
            logger.error(
                f"Unable to get user information: Required information not found in response: {e}"
            )
            raise UserInfoRetrievalError(
                "Required information not found in response"
            ) from e
        except Exception as e:
            logger.error(f"Error during get_thongtin: {e}")
            raise UserInfoRetrievalError(
                f"An error occurred during get_thongtin: {e}"
            ) from e

    async def report_user(self, target_user_id: str, timeout: int) -> None:
        if not self.fb_dtsg or not self.jazoest:
            logger.info("Data not loaded yet. Reloading...")
            try:
                await self.get_thongtin()
            except UserInfoRetrievalError as e:
                raise ReportError(f"Failed to retrieve user information: {e}") from e

        reasons = {
            "spam": "Spam",
            "violation": "Vi phạm quy tắc",
            "hate_speech": "Ngôn ngữ thù địch",
            "pornography": "Nội dung khiêu dâm",
            "harassment": "Ngược đãi",
            "impersonation": "Giả mạo",
            "personal_attack": "Tấn công cá nhân",
        }
        for reason_code, reason_description in reasons.items():
            data = {
                "av": self.user_id,
                "__user": self.user_id,
                "fb_dtsg": self.fb_dtsg,
                "jazoest": self.jazoest,
                "target_user_id": target_user_id,
                "report_type": "user",
                "reason": reason_code,
            }
            try:
                async with self.session.post(
                    "https://www.facebook.com/report/user", data=data, timeout=timeout
                ) as response:
                    if response.status == 200:
                        logger.info(
                            f"Successfully reported user {target_user_id} with reason '{reason_description}'."
                        )
                    else:
                        logger.error(
                            f"Report failed with reason '{reason_description}'. Response status code: {response.status}"
                        )
                        raise ReportError(f"Report failed: status code {response.status}")
            except aiohttp.ClientError as e:
                logger.error(
                    f"Error executing report with reason '{reason_description}': {e}"
                )
                raise ReportError(
                    f"Network error while reporting: {e}, with reason: {reason_description}"
                ) from e
            except Exception as e:
                logger.error(
                    f"Unexpected error while reporting with reason '{reason_description}': {e}"
                )
                raise ReportError(
                    f"An unexpected error occurred: {e}, with reason {reason_description}"
                ) from e


async def handle_report_request(
    user_id: int, state_data: UserState, context: ContextTypes.DEFAULT_TYPE
) -> None:
    try:
        if not state_data.cookie:
            raise InvalidInputError(
                "Cookie not found in state, something wrong with flow."
            )

        decrypted_cookie = decrypt_cookie(state_data.cookie)
        api = FacebookApiVIP(decrypted_cookie)

        async with api.session or aiohttp.ClientSession():
            await api.create_session()
            await api.get_thongtin()
            await api.report_user(state_data.target_id, state_data.timeout)
        await context.bot.send_message(
            chat_id=user_id, text="Report process completed."
        )

    except InvalidInputError as e:
        logger.error(f"Error during report for user {user_id}: Invalid input: {e}")
        await context.bot.send_message(
            chat_id=user_id,
            text=f"Invalid input: {e}. Please make sure the process is correctly done with correct information.",
        )
    except FacebookApiException as e:
        logger.error(f"Error during report for user {user_id}: Facebook API error: {e}")
        await context.bot.send_message(
            chat_id=user_id,
            text=f"An error occurred during the Facebook API call: {e} Please try again.",
        )
    except Exception as e:
        logger.error(f"Unexpected error during report for user {user_id}: {e}")
        await context.bot.send_message(
            chat_id=user_id,
            text=f"An unexpected error occurred: {e}. Please try again.",
        )
    finally:
        state_data.state = "idle"
        conn = create_connection()
        save_user_state_db(conn, user_id, state_data)
        conn.close()

# decorator for command validation for private chat
def private_chat_only(func: Callable):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_chat.type != "private":
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="This command can only be used in private chats."
            )
            return
        return await func(update, context)
    return wrapper

@private_chat_only
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Hello! I'm ready to report Facebook users. Use /cookie to set your cookie and then /report to start.",
    )

@private_chat_only
async def cookie_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_states.setdefault(user_id, UserState()).state = "waiting_for_cookie"
    conn = create_connection()
    save_user_state_db(conn, user_id, user_states.get(user_id))
    conn.close()
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Please send your Facebook cookie. It will be encrypted and stored temporarily.",
    )

@private_chat_only
async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    state_data = user_states.get(user_id)

    if not state_data or not state_data.cookie:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Please set your Facebook cookie first using /cookie.",
        )
        return

    state_data.state = "waiting_for_target_id"
    conn = create_connection()
    save_user_state_db(conn, user_id, state_data)
    conn.close()
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Please enter the ID of the person you want to report.",
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    message_text = update.message.text
    state_data = user_states.get(user_id)
    conn = create_connection()

    if not state_data:
        state_data = load_user_state_db(conn, user_id)
        if state_data:
            user_states[user_id] = state_data
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Please use /start to begin.",
            )
            conn.close()
            return

    try:
        if state_data.state == "waiting_for_cookie":
            encrypted_cookie = encrypt_cookie(message_text)
            state_data.cookie = encrypted_cookie
            state_data.state = "idle"
            save_user_state_db(conn, user_id, state_data)
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Your cookie has been received and encrypted. Use /report to proceed.",
            )
        elif state_data.state == "waiting_for_target_id":
            state_data.target_id = message_text
            state_data.state = "waiting_for_timeout"
            save_user_state_db(conn, user_id, state_data)
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Please enter the timeout for the report request (in seconds)",
            )
        elif state_data.state == "waiting_for_timeout":
            validated_input = UserInput(
                target_id=state_data.target_id, timeout=message_text
            )
            state_data.timeout = validated_input.timeout
            state_data.state = "processing_report"
            save_user_state_db(conn, user_id, state_data)
            asyncio.create_task(handle_report_request(user_id, state_data, context))
    except ValidationError as e:
        logger.error(f"Validation error for user {user_id}: {e}")
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text=f"Validation Error: {e}"
        )
    except Exception as e:
        logger.error(f"Unexpected error while processing message for user {user_id}: {e}")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"An unexpected error occurred, please try again.",
        )
    finally:
        conn.close()


async def shutdown(app: Application) -> None:
    logger.info("Shutting down bot gracefully...")
    await app.stop()
    logger.info("Bot stopped successfully.")


async def setup_application(token: str) -> Application:
    conn = create_connection()
    conn.close()  # close for now since it might not be used immediately.
    application = ApplicationBuilder().token(token).build()

    start_handler = CommandHandler("start", start_command)
    cookie_handler = CommandHandler("cookie", cookie_command)
    report_handler = CommandHandler("report", report_command)
    message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)

    application.add_handler(start_handler)
    application.add_handler(cookie_handler)
    application.add_handler(report_handler)
    application.add_handler(message_handler)
    return application


def run_application(application: Application):
    loop = asyncio.get_event_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig, lambda: asyncio.create_task(shutdown(application))
        )

    try:
        loop.run_until_complete(application.run_polling())
    except KeyboardInterrupt:
        pass
    finally:
        if not loop.is_closed():
            loop.run_until_complete(application.stop())
            loop.close()

def main() -> None:
    try:
        application = asyncio.run(setup_application(TELEGRAM_TOKEN))
        run_application(application)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Failed to run bot: {e}")


if __name__ == "__main__":
    main()