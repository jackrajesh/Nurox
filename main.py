from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
import httpx
import re
import numpy as np
import os
import logging

from config import GROQ_API_KEY_AI1, GROQ_API_KEY_AI2, MODEL_NAME
from database.connection import engine
from database.models import Base, User, DebateHistory
from auth.routes import router as auth_router, get_current_user, get_db
from services.usage_limiter import UsageLimiter
from admin.routes import router as admin_router


# -------------------------
# Logging (Production Safe)
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------
# App Init
# -------------------------
app = FastAPI(title="NUROX V6.3 Intelligence Platform")


# -------------------------
# CORS (Production Ready)
# -------------------------
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Startup Event (DB Init)
# -------------------------
@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)


# -------------------------
# Routers
# -------------------------
app.include_router(auth_router)
app.include_router(admin_router)


# -------------------------
# Schemas
# -------------------------
class DebateRequest(BaseModel):
    question: str


class DebateMessage(BaseModel):
    role: str
    content: str


class DebateResponse(BaseModel):
    mode: str
    transcript: List[DebateMessage]
    deterministic: Optional[str]
    simulation: Optional[str]
    simulation_data: Optional[List[float]]
    risk_alerts: Optional[str]
    final_answer: str
    authority: str
    confidence: str
    usage: dict


# -------------------------
# LLM Caller (Stable Version)
# -------------------------
async def call_llm(api_key, system_prompt, messages, temperature=0.3):
    timeout = httpx.Timeout(60.0, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "system", "content": system_prompt}] + messages,
                "temperature": temperature,
                "max_tokens": 900,
            },
        )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"LLM Error: {response.text}")

    data = response.json()

    if not data.get("choices"):
        raise HTTPException(status_code=500, detail="Invalid LLM response structure.")

    return data["choices"][0]["message"]["content"].strip()


# -------------------------
# Utility Logic
# -------------------------
def detect_mode(question: str):
    keywords = ["risk", "reward", "win rate", "break", "transaction", "slippage"]
    return "quant" if any(k in question.lower() for k in keywords) else "general"


def deterministic_engine(question: str):
    nums = list(map(float, re.findall(r"\d+\.?\d*", question)))

    if len(nums) < 2:
        return None, None

    risk = nums[0]
    reward = nums[1]
    transaction = nums[2] if len(nums) >= 3 else 0
    slippage = nums[3] if len(nums) >= 4 else 0

    net_win = reward - transaction - slippage
    net_loss = -risk - transaction
    denom = net_win - net_loss

    if denom == 0:
        return None, None

    p = -net_loss / denom
    ev = (p * net_win) + ((1 - p) * net_loss)

    return p, ev


def monte_carlo_equity(win_prob, reward_ratio=0.02, risk_ratio=0.01, trades=200):
    equity_curve = []
    capital = 1.0

    for _ in range(trades):
        capital *= (1 + reward_ratio) if np.random.rand() < win_prob else (1 - risk_ratio)
        equity_curve.append(round(capital, 4))

    return equity_curve


PROMPT_BUILDER = (
    "You are a professional analyst. "
    "Use emojis. Use **bold** for key concepts. "
    "Use structured sections. Avoid text walls."
)

PROMPT_AUDITOR = (
    "Provide a decisive final professional conclusion. "
    "Use emojis. Use **bold**. Keep it clean and structured."
)


# -------------------------
# Main Endpoint
# -------------------------
@app.post("/debate", response_model=DebateResponse)
async def debate(
    req: DebateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    limiter = UsageLimiter(db)
    usage_info = limiter.check_and_consume(current_user)

    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question empty.")

    mode = detect_mode(question)
    transcript = []

    builder = await call_llm(
        GROQ_API_KEY_AI1,
        PROMPT_BUILDER,
        [{"role": "user", "content": question}],
        0.3,
    )

    transcript.append(DebateMessage(role="🧠 Builder", content=builder))

    deterministic = simulation_block = simulation_data = risk_alerts = None
    authority = "LLM"
    confidence = "Medium"

    if mode == "quant":
        p, ev = deterministic_engine(question)

        if p is not None:
            deterministic = (
                f"🎯 **Break-even** = {p:.4f} ({p*100:.2f}%) | "
                f"**EV** = {ev:.4f}"
            )

            equity_curve = monte_carlo_equity(p)
            simulation_data = equity_curve
            simulation_block = (
                f"📊 **Equity Curve Generated** with {len(equity_curve)} trades."
            )

            risk_alerts = (
                "🟢 **Stable Risk Profile**"
                if p > 0.4
                else "🔴 **High Risk Profile**"
            )

            authority = "Deterministic + LLM"
            confidence = "High"

    final_answer = await call_llm(
        GROQ_API_KEY_AI2,
        PROMPT_AUDITOR,
        [{"role": "user", "content": builder}],
        0.3,
    )

    db.add(
        DebateHistory(
            user_id=current_user.id,
            question=question,
            final_answer=final_answer,
            mode=mode,
        )
    )

    db.commit()

    logger.info(f"Debate executed | User: {current_user.id} | Mode: {mode}")

    return DebateResponse(
        mode=mode,
        transcript=transcript,
        deterministic=deterministic,
        simulation=simulation_block,
        simulation_data=simulation_data,
        risk_alerts=risk_alerts,
        final_answer=final_answer,
        authority=authority,
        confidence=confidence,
        usage=usage_info,
    )


# -------------------------
# History
# -------------------------
@app.get("/history")
def get_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return (
        db.query(DebateHistory)
        .filter(DebateHistory.user_id == current_user.id)
        .order_by(DebateHistory.created_at.desc())
        .all()
    )


# -------------------------
# Usage
# -------------------------
@app.get("/usage")
def get_usage(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from database.models import UsageTracking, PLAN_LIMITS

    tracking = db.query(UsageTracking).filter_by(
        user_id=current_user.id
    ).first()

    limits = PLAN_LIMITS.get(current_user.plan or "free", {})

    if not tracking:
        return {
            "plan": current_user.plan,
            "debates_today": 0,
            "debates_this_month": 0,
            "daily_limit": limits.get("daily_debates"),
            "monthly_limit": limits.get("monthly_debates"),
        }

    return {
        "plan": current_user.plan,
        "debates_today": tracking.debates_today,
        "daily_limit": limits.get("daily_debates"),
        "debates_this_month": tracking.debates_this_month,
        "monthly_limit": limits.get("monthly_debates"),
        "total_debates": tracking.total_debates,
    }


# -------------------------
# Health Check
# -------------------------
@app.get("/health")
async def health():
    return {"status": "NUROX V6.3 Running ✅"}