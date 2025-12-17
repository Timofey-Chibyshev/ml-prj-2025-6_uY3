from sqlalchemy import Column, Integer, String
from utils.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=False)
    username = Column(String, nullable=True)
    history = Column(String, default="")

    def __repr__(self):
        return f"<User id={self.id}>"
