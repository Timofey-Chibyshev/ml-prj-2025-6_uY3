from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from config import DATABASE_URL


class Base(DeclarativeBase):
    pass


engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True
)

async_session = async_sessionmaker(
    engine,
    expire_on_commit=False,
    autoflush=False
)


async def init_db():
    """Создание всех таблиц (при первом запуске)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
