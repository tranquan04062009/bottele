import asyncio
import logging
import re
import os
import signal
from typing import Dict, Optional, Any, Tuple, Callable, List
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackQueryHandler,
)
import aiohttp
from dataclasses import dataclass, field
import sqlite3
from pydantic import BaseModel, field_validator, ValidationError

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Telegram token from environment variable
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    logger.error(
        "Telegram token không được tìm thấy trong biến môi trường TELEGRAM_TOKEN."
    )
    exit()

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
    report_speed: int

    @field_validator("target_id")
    def validate_target_id(cls, value):
        if not value.isdigit():
            raise ValueError("ID mục tiêu không hợp lệ. Vui lòng nhập ID số.")
        return value

    @field_validator("timeout")
    def validate_timeout(cls, value):
        if value <= 0:
            raise ValueError("Thời gian chờ phải là một số dương.")
        return value

    @field_validator("report_speed")
    def validate_report_speed(cls, value):
        if value < 0:
            raise ValueError("Tốc độ báo cáo phải là một số dương.")
        return value


@dataclass
class UserState:
    state: str = "idle"
    cookie: Optional[str] = None
    target_id: Optional[str] = None
    timeout: Optional[int] = None
    report_speed: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "UserState":
        return cls(**data)

    def to_dict(self) -> Dict:
        return {
            "state": self.state,
            "cookie": self.cookie,
            "target_id": self.target_id,
            "timeout": self.timeout,
            "report_speed": self.report_speed,
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
            timeout INTEGER,
            report_speed INTEGER
        )
    """
    )
    conn.commit()
    return conn


def save_user_state_db(conn: sqlite3.Connection, user_id: int, user_state: UserState):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO user_states (user_id, state, cookie, target_id, timeout, report_speed) VALUES (?, ?, ?, ?, ?, ?)",
        (
            user_id,
            user_state.state,
            user_state.cookie,
            user_state.target_id,
            user_state.timeout,
            user_state.report_speed,
        ),
    )
    conn.commit()


def load_user_state_db(
    conn: sqlite3.Connection, user_id: int
) -> Optional[UserState]:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT state, cookie, target_id, timeout, report_speed FROM user_states WHERE user_id = ?",
        (user_id,),
    )
    result = cursor.fetchone()
    if result:
        state, cookie, target_id, timeout, report_speed = result
        return UserState(
            state=state,
            cookie=cookie,
            target_id=target_id,
            timeout=timeout,
            report_speed=report_speed
        )
    return None


# User States Dictionary
user_states: Dict[int, UserState] = {}


class FacebookApiVIP:
    def __init__(self, cookie: str):
        self.cookie: str = cookie
        try:
            self.user_id: str = self._extract_user_id(cookie)
        except ValueError as e:
            raise InvalidCookieError("Không thể trích xuất ID người dùng từ cookie") from e
        self.headers: dict = self._generate_headers(cookie)
        self.fb_dtsg: Optional[str] = None
        self.jazoest: Optional[str] = None
        self.session: Optional[aiohttp.ClientSession] = None

    def _extract_user_id(self, cookie: str) -> str:
        try:
            return cookie.split("c_user=")[1].split(";")[0]
        except IndexError:
            logger.error("Không thể lấy user_id từ cookie: c_user không tìm thấy")
            raise ValueError("Định dạng cookie không hợp lệ")

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
            logger.info("Đang sử dụng dữ liệu đã lưu.")
            return self.fb_dtsg, self.jazoest

        try:
            async with self.session.get(
                "https://mbasic.facebook.com/profile.php", timeout=10
            ) as response:
                home = await response.text()
                if not home:
                    raise UserInfoRetrievalError(
                        "Phản hồi trống từ Facebook"
                    )
                self.fb_dtsg = re.search(r'name="fb_dtsg" value="(.*?)"', home).group(
                    1
                )
                self.jazoest = re.search(r'name="jazoest" value="(.*?)"', home).group(
                    1
                )
                ten = re.search(r"<title>(.*?)</title>", home).group(1)
                logger.info(f"Tên Facebook: {ten}, ID: {self.user_id}")
                return ten, self.user_id
        except AttributeError as e:
            logger.error(
                f"Không thể lấy thông tin người dùng: Không tìm thấy thông tin bắt buộc trong phản hồi: {e}"
            )
            raise UserInfoRetrievalError(
                "Không tìm thấy thông tin bắt buộc trong phản hồi"
            ) from e
        except Exception as e:
            logger.error(f"Lỗi trong quá trình lấy thông tin: {e}")
            raise UserInfoRetrievalError(
                f"Đã xảy ra lỗi trong quá trình lấy thông tin: {e}"
            ) from e

    async def report_user(self, target_user_id: str, timeout: int, report_speed: int) -> None:
        if not self.fb_dtsg or not self.jazoest:
            logger.info("Dữ liệu chưa được tải. Đang tải lại...")
            try:
                await self.get_thongtin()
            except UserInfoRetrievalError as e:
                raise ReportError(f"Không thể lấy thông tin người dùng: {e}") from e

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
                            f"Đã báo cáo thành công người dùng {target_user_id} với lý do '{reason_description}'."
                        )
                    else:
                        logger.error(
                            f"Báo cáo thất bại với lý do '{reason_description}'. Mã trạng thái phản hồi: {response.status}"
                        )
                        raise ReportError(
                            f"Báo cáo thất bại: mã trạng thái {response.status}"
                        )
                await asyncio.sleep(report_speed) # Add delay after each report.
            except aiohttp.ClientError as e:
                logger.error(
                    f"Lỗi khi thực hiện báo cáo với lý do '{reason_description}': {e}"
                )
                raise ReportError(
                    f"Lỗi mạng khi báo cáo: {e}, với lý do: {reason_description}"
                ) from e
            except Exception as e:
                logger.error(
                    f"Lỗi không mong muốn khi báo cáo với lý do '{reason_description}': {e}"
                )
                raise ReportError(
                    f"Đã xảy ra lỗi không mong muốn: {e}, với lý do {reason_description}"
                ) from e


async def handle_report_request(
    user_id: int, state_data: UserState, context: ContextTypes.DEFAULT_TYPE
) -> None:
    try:
        if not state_data.cookie:
            raise InvalidInputError(
                "Không tìm thấy cookie trong dữ liệu, có lỗi trong quy trình."
            )
        api = FacebookApiVIP(state_data.cookie)

        async with api.session or aiohttp.ClientSession():
            await api.create_session()
            await api.get_thongtin()
            await api.report_user(state_data.target_id, state_data.timeout, state_data.report_speed)
        await context.bot.send_message(
            chat_id=user_id, text="Quá trình báo cáo đã hoàn tất."
        )

    except InvalidInputError as e:
        logger.error(f"Lỗi trong quá trình báo cáo cho người dùng {user_id}: Đầu vào không hợp lệ: {e}")
        await context.bot.send_message(
            chat_id=user_id,
            text=f"Đầu vào không hợp lệ: {e}. Vui lòng đảm bảo quá trình được thực hiện đúng với thông tin chính xác.",
        )
    except FacebookApiException as e:
        logger.error(f"Lỗi trong quá trình báo cáo cho người dùng {user_id}: Lỗi Facebook API: {e}")
        await context.bot.send_message(
            chat_id=user_id,
            text=f"Đã xảy ra lỗi trong quá trình gọi API Facebook: {e} Vui lòng thử lại.",
        )
    except Exception as e:
        logger.error(f"Lỗi không mong muốn trong quá trình báo cáo cho người dùng {user_id}: {e}")
        await context.bot.send_message(
            chat_id=user_id,
            text=f"Đã xảy ra lỗi không mong muốn: {e}. Vui lòng thử lại.",
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
                text="Lệnh này chỉ có thể được sử dụng trong các cuộc trò chuyện riêng tư.",
            )
            return
        return await func(update, context)
    return wrapper


@private_chat_only
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [InlineKeyboardButton("Bắt đầu", callback_data="start_report")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Xin chào! Tôi đã sẵn sàng để báo cáo người dùng Facebook. Bạn muốn bắt đầu chứ?",
        reply_markup=reply_markup
    )


async def handle_start_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Tuyệt vời! Hãy sử dụng /cookie để thiết lập cookie Facebook của bạn, sau đó sử dụng /report để bắt đầu báo cáo."
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
        text="Vui lòng gửi cookie Facebook của bạn. Nó sẽ được lưu trữ tạm thời.",
    )


@private_chat_only
async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    state_data = user_states.get(user_id)

    if not state_data or not state_data.cookie:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Vui lòng thiết lập cookie Facebook của bạn trước bằng lệnh /cookie.",
        )
        return

    state_data.state = "waiting_for_target_id"
    conn = create_connection()
    save_user_state_db(conn, user_id, state_data)
    conn.close()
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Vui lòng nhập ID của người bạn muốn báo cáo.",
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
                text="Vui lòng sử dụng lệnh /start để bắt đầu.",
            )
            conn.close()
            return

    try:
        if state_data.state == "waiting_for_cookie":
            state_data.cookie = message_text
            state_data.state = "idle"
            save_user_state_db(conn, user_id, state_data)
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Cookie của bạn đã được nhận. Sử dụng lệnh /report để tiếp tục.",
            )
        elif state_data.state == "waiting_for_target_id":
            state_data.target_id = message_text
            state_data.state = "waiting_for_timeout"
            save_user_state_db(conn, user_id, state_data)
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Vui lòng nhập thời gian chờ cho yêu cầu báo cáo (tính bằng giây)",
            )
        elif state_data.state == "waiting_for_timeout":
           state_data.timeout = int(message_text)
           state_data.state = "waiting_for_report_speed"
           save_user_state_db(conn, user_id, state_data)
           await context.bot.send_message(
               chat_id=update.effective_chat.id,
               text="Tuyệt! Bây giờ, hãy nhập tốc độ báo cáo (thời gian chờ giữa các lần báo cáo, tính bằng giây)",
           )
        elif state_data.state == "waiting_for_report_speed":
            validated_input = UserInput(
                target_id=state_data.target_id, timeout=state_data.timeout, report_speed = message_text
            )
            state_data.report_speed = validated_input.report_speed
            state_data.state = "processing_report"
            save_user_state_db(conn, user_id, state_data)
            asyncio.create_task(handle_report_request(user_id, state_data, context))
    except ValidationError as e:
         logger.error(f"Lỗi xác thực cho người dùng {user_id}: {e}")
         await context.bot.send_message(
           chat_id=update.effective_chat.id, text=f"Lỗi xác thực: {e}"
         )
    except ValueError as e:
         logger.error(f"Lỗi đầu vào cho người dùng {user_id}: {e}")
         await context.bot.send_message(
            chat_id=update.effective_chat.id, text=f"Lỗi đầu vào: {e}."
         )
    except Exception as e:
        logger.error(f"Lỗi không mong muốn khi xử lý tin nhắn cho người dùng {user_id}: {e}")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Đã xảy ra lỗi không mong muốn, vui lòng thử lại.",
        )
    finally:
        conn.close()


async def shutdown(app: Application) -> None:
    logger.info("Đang tắt bot một cách êm ái...")
    await app.stop()
    logger.info("Bot đã dừng thành công.")



async def main() -> None:
        application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

        start_handler = CommandHandler("start", start_command)
        start_button_handler = CallbackQueryHandler(handle_start_button, pattern="start_report")
        cookie_handler = CommandHandler("cookie", cookie_command)
        report_handler = CommandHandler("report", report_command)
        message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)

        application.add_handler(start_handler)
        application.add_handler(start_button_handler)
        application.add_handler(cookie_handler)
        application.add_handler(report_handler)
        application.add_handler(message_handler)

        loop = asyncio.get_event_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig, lambda: asyncio.create_task(shutdown(application))
            )
        try:
           await application.run_polling()
        except Exception as e:
             logger.error(f"Không thể chạy bot: {e}")
        finally:
             if not loop.is_closed():
                 await application.stop()
                 loop.close()



if __name__ == "__main__":
    asyncio.run(main())