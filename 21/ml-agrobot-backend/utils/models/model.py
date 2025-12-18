from sqlalchemy import Column, Integer, String, Boolean
from utils.db import Base


class MLModel(Base):
    __tablename__ = "ml_models"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    path = Column(String, nullable=False)
    is_active = Column(Boolean, default=False)
    accuracy = Column(String, nullable=True)

    def __repr__(self):
        return f"<MLModel name={self.name} active={self.is_active}>"
