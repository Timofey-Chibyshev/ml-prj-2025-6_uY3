import asyncio
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage

from config import TOKEN
from utils.db import init_db
from handlers.user import user_router
from handlers.admin import admin_router


async def main():
    print("Initializing database...")
    await init_db()
    print("Database OK")

    bot = Bot(
        token=TOKEN,
        default=DefaultBotProperties(parse_mode="HTML")
    )

    dp = Dispatcher(storage=MemoryStorage())

    dp.include_router(user_router)
    dp.include_router(admin_router)

    print("Bot started!")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
