from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from utils.db import Base


class Photo(Base):
    __tablename__ = "photos"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    photo_path = Column(String, nullable=False)
    prediction = Column(String, nullable=False)
    feedback = Column(String, nullable=True)

    user = relationship("User")

    def __repr__(self):
        return f"<Photo id={self.id} user={self.user_id}>"
