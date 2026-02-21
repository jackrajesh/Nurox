"""
NUROX V6.3 - Admin Panel Routes
Production Secure Version
"""

import os
import secrets
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy.orm import Session

from database.connection import SessionLocal
from database.models import User, UsageTracking, DebateHistory

router = APIRouter()
security = HTTPBasic()


# ============================
# ADMIN CREDENTIALS (ENV BASED)
# ============================

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

if not ADMIN_USERNAME or not ADMIN_PASSWORD:
    raise RuntimeError("Admin credentials not set in environment variables.")


# ============================
# DATABASE DEPENDENCY
# ============================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================
# VERIFY ADMIN
# ============================

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_user = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_pass = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)

    if not (correct_user and correct_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Wrong credentials.",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True


# ============================
# GET ALL USERS
# ============================

@router.get("/shadow-admin/users")
def admin_get_users(
    db: Session = Depends(get_db),
    _: bool = Depends(verify_admin)
):
    users = db.query(User).all()
    result = []

    for u in users:
        tracking = db.query(UsageTracking).filter_by(user_id=u.id).first()

        result.append({
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "plan": u.plan,
            "is_active": u.is_active,
            "created_at": str(u.created_at),
            "debates_today": tracking.debates_today if tracking else 0,
            "debates_this_month": tracking.debates_this_month if tracking else 0,
            "total_debates": tracking.total_debates if tracking else 0,
        })

    return result


# ============================
# UPGRADE USER PLAN
# ============================

@router.post("/shadow-admin/upgrade")
def admin_upgrade_user(
    username: str,
    plan: str,
    db: Session = Depends(get_db),
    _: bool = Depends(verify_admin)
):
    if plan not in ["free", "pro", "enterprise"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid plan. Use: free | pro | enterprise"
        )

    user = db.query(User).filter(User.username == username).first()

    if not user:
        raise HTTPException(
            status_code=404,
            detail=f"User '{username}' not found."
        )

    old_plan = user.plan
    user.plan = plan
    db.commit()

    return {
        "message": f"✅ {username} upgraded from {old_plan} → {plan}",
        "username": username,
        "new_plan": plan,
    }


# ============================
# DISABLE USER
# ============================

@router.post("/shadow-admin/disable")
def admin_disable_user(
    username: str,
    db: Session = Depends(get_db),
    _: bool = Depends(verify_admin)
):
    user = db.query(User).filter(User.username == username).first()

    if not user:
        raise HTTPException(
            status_code=404,
            detail=f"User '{username}' not found."
        )

    user.is_active = False
    db.commit()

    return {"message": f"🚫 {username} has been disabled."}


# ============================
# PLATFORM STATS
# ============================

@router.get("/shadow-admin/stats")
def admin_stats(
    db: Session = Depends(get_db),
    _: bool = Depends(verify_admin)
):
    total_users = db.query(User).count()
    total_debates = db.query(DebateHistory).count()

    free_users = db.query(User).filter(User.plan == "free").count()
    pro_users = db.query(User).filter(User.plan == "pro").count()
    enterprise = db.query(User).filter(User.plan == "enterprise").count()

    return {
        "total_users": total_users,
        "total_debates": total_debates,
        "plan_breakdown": {
            "free": free_users,
            "pro": pro_users,
            "enterprise": enterprise,
        }
    }