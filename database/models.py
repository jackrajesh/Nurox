from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime, Boolean, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()


# ============================
# PLAN TYPES
# ============================

class PlanType(str, enum.Enum):
    FREE       = "free"
    PRO        = "pro"
    ENTERPRISE = "enterprise"


# ============================
# PLAN LIMITS (Central config)
# ============================

PLAN_LIMITS = {
    "free": {
        "daily_debates":   5,
        "monthly_debates": 50,
        "rate_per_minute": 3,
    },
    "pro": {
        "daily_debates":   100,
        "monthly_debates": 2000,
        "rate_per_minute": 20,
    },
    "enterprise": {
        "daily_debates":   -1,   # -1 = unlimited
        "monthly_debates": -1,
        "rate_per_minute": 100,
    },
}


# ============================
# USER TABLE
# ============================

class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String, unique=True, index=True)
    email           = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    plan            = Column(String, default="free")   # free | pro | enterprise
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

    debates = relationship("DebateHistory", back_populates="user")
    usage   = relationship("UsageTracking", back_populates="user", uselist=False)


# ============================
# DEBATE HISTORY TABLE
# ============================

class DebateHistory(Base):
    __tablename__ = "debate_history"

    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("users.id"))
    question     = Column(Text)
    final_answer = Column(Text)
    mode         = Column(String, default="general")
    created_at   = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="debates")


# ============================
# USAGE TRACKING TABLE
# ============================

class UsageTracking(Base):
    __tablename__ = "usage_tracking"

    id                   = Column(Integer, primary_key=True, index=True)
    user_id              = Column(Integer, ForeignKey("users.id"), unique=True)

    # Daily
    debates_today        = Column(Integer, default=0)
    daily_reset_at       = Column(DateTime, default=datetime.utcnow)

    # Monthly
    debates_this_month   = Column(Integer, default=0)
    monthly_reset_at     = Column(DateTime, default=datetime.utcnow)

    # Rate limiting (per minute)
    requests_this_minute = Column(Integer, default=0)
    minute_window_start  = Column(DateTime, default=datetime.utcnow)

    # Lifetime
    total_debates        = Column(Integer, default=0)

    user = relationship("User", back_populates="usage")
