from datetime import date, datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session as DBSession

from .models import UserProfile, MealEntry, ChatMessage, get_session_factory, init_db


class SessionStore:
    """Manages user sessions, meal logging, and chat history."""

    def __init__(self, database_url: str = "sqlite:///./data/food_tracker.db"):
        engine = init_db(database_url)
        self._session_factory = get_session_factory(engine)

    def _session(self) -> DBSession:
        return self._session_factory()

    def get_or_create_user(self, session_id: str) -> dict:
        with self._session() as db:
            user = db.query(UserProfile).filter_by(session_id=session_id).first()
            if not user:
                user = UserProfile(session_id=session_id)
                db.add(user)
                db.commit()
                db.refresh(user)
            return {
                "id": user.id,
                "session_id": user.session_id,
                "name": user.name,
                "calorie_goal": user.calorie_goal,
                "protein_goal": user.protein_goal,
                "fat_goal": user.fat_goal,
                "carb_goal": user.carb_goal,
            }

    def update_user_goals(
        self,
        session_id: str,
        calorie_goal: Optional[float] = None,
        protein_goal: Optional[float] = None,
        fat_goal: Optional[float] = None,
        carb_goal: Optional[float] = None,
        name: Optional[str] = None,
    ) -> dict:
        with self._session() as db:
            user = db.query(UserProfile).filter_by(session_id=session_id).first()
            if not user:
                user = UserProfile(session_id=session_id)
                db.add(user)

            if calorie_goal is not None:
                user.calorie_goal = calorie_goal
            if protein_goal is not None:
                user.protein_goal = protein_goal
            if fat_goal is not None:
                user.fat_goal = fat_goal
            if carb_goal is not None:
                user.carb_goal = carb_goal
            if name is not None:
                user.name = name

            db.commit()
            db.refresh(user)
            return {
                "calorie_goal": user.calorie_goal,
                "protein_goal": user.protein_goal,
                "fat_goal": user.fat_goal,
                "carb_goal": user.carb_goal,
                "name": user.name,
            }

    def log_meal(
        self,
        session_id: str,
        description: str,
        calories: float,
        protein: float,
        fat: float,
        carbs: float,
        meal_type: Optional[str] = None,
        fiber: float = 0.0,
        sugar: float = 0.0,
        sodium: float = 0.0,
    ) -> dict:
        with self._session() as db:
            user = db.query(UserProfile).filter_by(session_id=session_id).first()
            if not user:
                user = UserProfile(session_id=session_id)
                db.add(user)
                db.commit()
                db.refresh(user)

            entry = MealEntry(
                user_id=user.id,
                description=description,
                calories=calories,
                protein=protein,
                fat=fat,
                carbs=carbs,
                meal_type=meal_type,
                fiber=fiber,
                sugar=sugar,
                sodium=sodium,
            )
            db.add(entry)
            db.commit()
            return {
                "id": entry.id,
                "description": entry.description,
                "calories": entry.calories,
                "protein": entry.protein,
                "fat": entry.fat,
                "carbs": entry.carbs,
                "meal_type": entry.meal_type,
                "meal_date": str(entry.meal_date),
            }

    def get_daily_summary(self, session_id: str, target_date: Optional[date] = None) -> dict:
        target_date = target_date or date.today()
        with self._session() as db:
            user = db.query(UserProfile).filter_by(session_id=session_id).first()
            if not user:
                return {"meals": [], "totals": _empty_totals(), "goals": _default_goals()}

            meals = (
                db.query(MealEntry)
                .filter_by(user_id=user.id, meal_date=target_date)
                .order_by(MealEntry.created_at)
                .all()
            )

            totals = {
                "calories": sum(m.calories for m in meals),
                "protein": sum(m.protein for m in meals),
                "fat": sum(m.fat for m in meals),
                "carbs": sum(m.carbs for m in meals),
                "fiber": sum(m.fiber for m in meals),
                "sugar": sum(m.sugar for m in meals),
                "sodium": sum(m.sodium for m in meals),
            }
            goals = {
                "calorie_goal": user.calorie_goal,
                "protein_goal": user.protein_goal,
                "fat_goal": user.fat_goal,
                "carb_goal": user.carb_goal,
            }
            remaining = {
                "calories": max(0, user.calorie_goal - totals["calories"]),
                "protein": max(0, user.protein_goal - totals["protein"]),
                "fat": max(0, user.fat_goal - totals["fat"]),
                "carbs": max(0, user.carb_goal - totals["carbs"]),
            }

            return {
                "date": str(target_date),
                "meals": [
                    {
                        "description": m.description,
                        "meal_type": m.meal_type,
                        "calories": m.calories,
                        "protein": m.protein,
                        "fat": m.fat,
                        "carbs": m.carbs,
                    }
                    for m in meals
                ],
                "totals": totals,
                "goals": goals,
                "remaining": remaining,
            }

    def get_weekly_stats(self, session_id: str) -> dict:
        today = date.today()
        week_ago = today - timedelta(days=7)
        with self._session() as db:
            user = db.query(UserProfile).filter_by(session_id=session_id).first()
            if not user:
                return {"days": []}

            meals = (
                db.query(MealEntry)
                .filter(
                    MealEntry.user_id == user.id,
                    MealEntry.meal_date >= week_ago,
                    MealEntry.meal_date <= today,
                )
                .all()
            )

            days: dict[date, dict] = {}
            for m in meals:
                d = m.meal_date
                if d not in days:
                    days[d] = {"calories": 0, "protein": 0, "fat": 0, "carbs": 0}
                days[d]["calories"] += m.calories
                days[d]["protein"] += m.protein
                days[d]["fat"] += m.fat
                days[d]["carbs"] += m.carbs

            return {
                "days": [
                    {"date": str(d), **vals}
                    for d, vals in sorted(days.items())
                ],
                "goals": {
                    "calorie_goal": user.calorie_goal,
                    "protein_goal": user.protein_goal,
                    "fat_goal": user.fat_goal,
                    "carb_goal": user.carb_goal,
                },
            }

    def save_chat_message(self, session_id: str, role: str, content: str):
        with self._session() as db:
            msg = ChatMessage(session_id=session_id, role=role, content=content)
            db.add(msg)
            db.commit()

    def get_chat_history(self, session_id: str, limit: int = 50) -> list[dict]:
        with self._session() as db:
            messages = (
                db.query(ChatMessage)
                .filter_by(session_id=session_id)
                .order_by(ChatMessage.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {"role": m.role, "content": m.content}
                for m in reversed(messages)
            ]


def _empty_totals():
    return {"calories": 0, "protein": 0, "fat": 0, "carbs": 0, "fiber": 0, "sugar": 0, "sodium": 0}


def _default_goals():
    return {"calorie_goal": 2000, "protein_goal": 150, "fat_goal": 70, "carb_goal": 250}
