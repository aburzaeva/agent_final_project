import uuid
from datetime import datetime, date

from sqlalchemy import (
    Column, String, Float, Integer, Date, DateTime, Text, ForeignKey, create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=True)
    calorie_goal = Column(Float, default=2000.0)
    protein_goal = Column(Float, default=150.0)
    fat_goal = Column(Float, default=70.0)
    carb_goal = Column(Float, default=250.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    meals = relationship("MealEntry", back_populates="user", cascade="all, delete-orphan")


class MealEntry(Base):
    __tablename__ = "meal_entries"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("user_profiles.id"), nullable=False)
    meal_date = Column(Date, default=date.today, index=True)
    meal_type = Column(String, nullable=True)  # breakfast, lunch, dinner, snack
    description = Column(Text, nullable=False)
    calories = Column(Float, default=0.0)
    protein = Column(Float, default=0.0)
    fat = Column(Float, default=0.0)
    carbs = Column(Float, default=0.0)
    fiber = Column(Float, default=0.0)
    sugar = Column(Float, default=0.0)
    sodium = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("UserProfile", back_populates="meals")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)  # user / assistant
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_engine(database_url: str = "sqlite:///./data/food_tracker.db"):
    return create_engine(database_url, echo=False)


def init_db(database_url: str = "sqlite:///./data/food_tracker.db"):
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(engine=None, database_url: str = "sqlite:///./data/food_tracker.db"):
    if engine is None:
        engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
