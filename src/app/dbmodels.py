from sqlalchemy import Column, Integer, String, Text
from app.database import Base, engine


class RequestHistory(Base):
    __tablename__ = "request_history"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(Text)
    result_summary = Column(Text)


Base.metadata.create_all(bind=engine)
