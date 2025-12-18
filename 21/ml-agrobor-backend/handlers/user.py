from aiogram import Router, F
from aiogram.types import Message, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

from utils.crud import (
    get_or_create_user,
    add_photo,
    update_feedback,
    get_active_model
)

import os
from datetime import datetime

from utils.ml import load_model, predict_image


class UserFlow(StatesGroup):
    choosing_culture = State()
    choosing_mode = State()
    waiting_photo = State()


user_router = Router()


@user_router.message(F.text == "/start")
async def cmd_start(message: Message, state: FSMContext):
    user = await get_or_create_user(message.from_user.id, message.from_user.username)
    await state.clear()

    kb = InlineKeyboardBuilder()
    kb.button(text="🌾 Пшеница", callback_data="culture:wheat")

    await message.answer(
        f"Привет, <b>{user.username or 'друг'}</b>! 👋\n\n"
        f"Я помогу диагностировать болезни колоса пшеницы.\n"
        f"Выбери культуру:",
        reply_markup=kb.as_markup()
    )

    await state.set_state(UserFlow.choosing_culture)


@user_router.callback_query(F.data.startswith("culture:"))
async def choose_culture(callback: CallbackQuery, state: FSMContext):
    _, culture = callback.data.split(":")
    await state.update_data(culture=culture)

    kb = InlineKeyboardBuilder()
    kb.button(text="🩺 Диагностика болезней", callback_data="mode:diagnostics")
    kb.button(text="⬅️ Назад", callback_data="back:start")
    kb.adjust(1)

    await callback.message.edit_text(
        f"Вы выбрали культуру: <b>Пшеница</b> 🌾\n\n"
        f"Теперь выберите режим:",
        reply_markup=kb.as_markup()
    )
    await callback.answer()
    await state.set_state(UserFlow.choosing_mode)


@user_router.callback_query(F.data.startswith("mode:"))
async def choose_mode(callback: CallbackQuery, state: FSMContext):
    _, mode = callback.data.split(":")
    await state.update_data(mode=mode)

    kb = InlineKeyboardBuilder()
    kb.button(text="⬅️ Назад", callback_data="back:culture")

    await callback.message.edit_text(
        "Отлично! 🩺\n"
        "Теперь отправьте <b>фото колоса пшеницы</b>.\n\n"
        "📌 Советы:\n"
        "• Хорошее освещение\n"
        "• Колос должен занимать большую часть кадра\n"
        "• Фото должно быть резким",
        reply_markup=kb.as_markup()
    )

    await callback.answer()
    await state.set_state(UserFlow.waiting_photo)


@user_router.callback_query(F.data.startswith("back:"))
async def go_back(callback: CallbackQuery, state: FSMContext):
    target = callback.data.split(":")[1]

    if target == "start":
        await cmd_start(callback.message, state)

    elif target == "culture":
        kb = InlineKeyboardBuilder()
        kb.button(text="🌾 Пшеница", callback_data="culture:wheat")
        kb.button(text="⬅️ Назад", callback_data="back:start")
        kb.adjust(1)

        await callback.message.edit_text(
            "Выберите культуру:",
            reply_markup=kb.as_markup()
        )
        await state.set_state(UserFlow.choosing_culture)

    await callback.answer()


@user_router.message(UserFlow.waiting_photo, F.photo)
async def handle_photo(message: Message, state: FSMContext):
    tg_photo = message.photo[-1]

    processing_msg = await message.answer("🔍 Обрабатываю фото...")

    os.makedirs("data/photos", exist_ok=True)
    filename = f"{message.from_user.id}_{datetime.now().timestamp()}.jpg"
    filepath = os.path.join("data/photos", filename)

    file = await message.bot.get_file(tg_photo.file_id)
    await message.bot.download_file(file.file_path, destination=filepath)

    active_model = await get_active_model()
    if not active_model:
        prediction = "❌ Нет активной модели!"
    else:
        model = await load_model(active_model.name, active_model.path)
        prediction = await predict_image(model, filepath)

    photo_id = await add_photo(
        user_id=message.from_user.id,
        path=filepath,
        prediction=prediction
    )

    kb = InlineKeyboardBuilder()
    kb.button(text="🔁 Отправить другое фото", callback_data="mode:diagnostics")
    kb.button(text="🌾 Выбрать культуру", callback_data="back:start")
    kb.button(text="🩺 Выбрать режим", callback_data="back:culture")
    kb.button(text="👍 Верно", callback_data=f"fb:{photo_id}:good")
    kb.button(text="👎 Неверно", callback_data=f"fb:{photo_id}:bad")
    kb.adjust(1)

    await processing_msg.edit_text(
        f"📷 Фото получено!\n\n"
        f"🧠 <b>Результат анализа:</b>\n"
        f"<i>{prediction}</i>\n\n"
        f"Что дальше?",
        reply_markup=kb.as_markup()
    )


@user_router.callback_query(F.data.startswith("fb:"))
async def feedback_handler(callback: CallbackQuery):
    _, photo_id, fb = callback.data.split(":")
    photo_id = int(photo_id)

    if fb == "good":
        await update_feedback(photo_id, "correct")
        await callback.answer("Спасибо! 👍")
    else:
        await update_feedback(photo_id, "wrong")
        await callback.answer("Спасибо за обратную связь! 👎")

    await callback.message.edit_reply_markup()
