from aiogram import Router, F
from aiogram.types import Message

from utils.db import async_session
from utils.models.user import User
from utils.models.photo import Photo
from utils.models.model import MLModel
from sqlalchemy import select, func
import os

from utils.crud import set_active_model

ADMIN_ID = int(os.getenv("ADMIN_ID", 0))

admin_router = Router()


def is_admin(message: Message) -> bool:
    return message.from_user.id == ADMIN_ID


@admin_router.message(F.text == "/admin")
async def admin_start(message: Message):
    if not is_admin(message):
        return await message.answer("У вас нет доступа ❌")

    await message.answer("Добро пожаловать в админ-панель! ⚙")


@admin_router.message(F.text == "/stats")
async def admin_stats(message: Message):
    if not is_admin(message):
        return await message.answer("У вас нет доступа ❌")

    async with async_session() as session:
        users = await session.execute(select(func.count(User.id)))
        photos = await session.execute(select(func.count(Photo.id)))

        await message.answer(
            f"📊 Статистика:\n"
            f"👤 Пользователей: {users.scalar()}\n"
            f"📷 Фото: {photos.scalar()}"
        )


@admin_router.message(F.text == "/models")
async def admin_models(message: Message):
    if not is_admin(message):
        return await message.answer("У вас нет доступа ❌")

    async with async_session() as session:
        result = await session.execute(select(MLModel))
        models = result.scalars().all()

    if not models:
        return await message.answer("Модели отсутствуют.")

    msg = "📦 Доступные модели:\n\n"
    for m in models:
        msg += f"• <b>{m.name}</b> {'(активная)' if m.is_active else ''}\n"

    await message.answer(msg)


@admin_router.message(F.text.startswith("/set_model"))
async def admin_set_model(message: Message):
    if not is_admin(message):
        return await message.answer("У вас нет доступа ❌")

    try:
        _, model_name = message.text.split()
    except ValueError:
        return await message.answer("Использование: /set_model <name>")

    await set_active_model(model_name)
    await message.answer(f"Активная модель: <b>{model_name}</b>")
