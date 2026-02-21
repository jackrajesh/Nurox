"""
NUROX V6.3 - Usage Limiter Service
"""

from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from database.models import User, UsageTracking, PLAN_LIMITS


class UsageLimiter:

    def __init__(self, db: Session):
        self.db = db

    def _get_tracking(self, user: User) -> UsageTracking:
        tracking = self.db.query(UsageTracking).filter_by(user_id=user.id).first()
        if not tracking:
            tracking = UsageTracking(user_id=user.id)
            self.db.add(tracking)
            self.db.commit()
            self.db.refresh(tracking)
        return tracking

    def _reset_windows(self, tracking: UsageTracking):
        now = datetime.utcnow()

        if now - tracking.daily_reset_at >= timedelta(hours=24):
            tracking.debates_today  = 0
            tracking.daily_reset_at = now

        if now - tracking.monthly_reset_at >= timedelta(days=30):
            tracking.debates_this_month = 0
            tracking.monthly_reset_at   = now

        if now - tracking.minute_window_start >= timedelta(minutes=1):
            tracking.requests_this_minute = 0
            tracking.minute_window_start  = now

    def check_and_consume(self, user: User) -> dict:
        plan     = user.plan or "free"
        limits   = PLAN_LIMITS[plan]
        tracking = self._get_tracking(user)

        self._reset_windows(tracking)

        # Rate limit per minute
        if tracking.requests_this_minute >= limits["rate_per_minute"]:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error":   "rate_limit_exceeded",
                    "message": f"Slow down! Max {limits['rate_per_minute']} requests/minute on {plan} plan.",
                    "upgrade": plan != "enterprise"
                }
            )

        # Daily limit
        daily_limit = limits["daily_debates"]
        if daily_limit != -1 and tracking.debates_today >= daily_limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error":     "daily_limit_exceeded",
                    "message":   f"Daily limit of {daily_limit} debates reached on {plan} plan.",
                    "used":      tracking.debates_today,
                    "limit":     daily_limit,
                    "resets_in": "24 hours",
                    "upgrade":   plan != "enterprise"
                }
            )

        # Monthly limit
        monthly_limit = limits["monthly_debates"]
        if monthly_limit != -1 and tracking.debates_this_month >= monthly_limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error":     "monthly_limit_exceeded",
                    "message":   f"Monthly limit of {monthly_limit} debates reached on {plan} plan.",
                    "used":      tracking.debates_this_month,
                    "limit":     monthly_limit,
                    "resets_in": "30 days",
                    "upgrade":   plan != "enterprise"
                }
            )

        # All good — consume one unit
        tracking.debates_today        += 1
        tracking.debates_this_month   += 1
        tracking.requests_this_minute += 1
        tracking.total_debates        += 1
        self.db.commit()

        return {
            "plan":          plan,
            "used_today":    tracking.debates_today,
            "daily_limit":   daily_limit,
            "used_monthly":  tracking.debates_this_month,
            "monthly_limit": monthly_limit,
        }
