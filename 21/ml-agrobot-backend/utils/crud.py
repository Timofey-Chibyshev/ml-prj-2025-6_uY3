from utils.db import async_session
from utils.models.user import User
from utils.models.photo import Photo
from utils.models.model import MLModel
from sqlalchemy import select, update


async def get_or_create_user(user_id: int, username: str | None = None) -> User:
    async with async_session() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if user:
            return user

        new_user = User(id=user_id, username=username or "")
        session.add(new_user)
        await session.commit()
        return new_user


async def add_photo(user_id: int, path: str, prediction: str):
    async with async_session() as session:
        photo = Photo(
            user_id=user_id,
            photo_path=path,
            prediction=prediction
        )
        session.add(photo)
        await session.commit()
        return photo.id


async def update_feedback(photo_id: int, feedback: str):
    async with async_session() as session:
        await session.execute(
            update(Photo)
            .where(Photo.id == photo_id)
            .values(feedback=feedback)
        )
        await session.commit()


async def get_last_photo(user_id: int) -> Photo | None:
    async with async_session() as session:
        result = await session.execute(
            select(Photo)
            .where(Photo.user_id == user_id)
            .order_by(Photo.id.desc())
        )
        return result.scalar_one_or_none()


async def get_active_model() -> MLModel:
    async with async_session() as session:
        result = await session.execute(select(MLModel).where(MLModel.is_active == True))
        return result.scalar_one_or_none()


async def set_active_model(name: str):
    async with async_session() as session:
        await session.execute(update(MLModel).values(is_active=False))
        await session.execute(
            update(MLModel).where(MLModel.name == name).values(is_active=True)
        )
        await session.commit()
