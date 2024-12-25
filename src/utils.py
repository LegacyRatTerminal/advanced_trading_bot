# src/utils.py
import logging
import asyncio
from telegram import Bot

class Logger:
    @staticmethod
    def setup_logging():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='logs/trading_bot.log'
        )
        return logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id

    async def send_message(self, message):
        """
        Async method to send telegram message
        """
        try:
            bot = Bot(token=self.token)
            await bot.send_message(chat_id=self.chat_id, text=message)
        except Exception as e:
            print(f"Telegram notification failed: {e}")
