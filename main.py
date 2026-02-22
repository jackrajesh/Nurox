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
logging.basicConfig(level=logging.INFO)
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
# CORS (Stable Production Version)
# -------------------------
FRONTEND_URL = os.getenv("FRONTEND_URL")

if FRONTEND_URL:
    origins = [FRONTEND_URL]
else:
    origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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
async def call_llm(api_key, system_prompt, messages, temperature=0.1):
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
                "max_tokens": 1500,
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
        return None, None, None, None

    risk       = nums[0]
    reward     = nums[1]
    transaction = nums[2] if len(nums) >= 3 else 0
    slippage   = nums[3] if len(nums) >= 4 else 0

    net_win  = reward - transaction - slippage
    net_loss = -risk - transaction
    denom    = net_win - net_loss

    if denom == 0:
        return None, None, None, None

    p  = -net_loss / denom
    ev = (p * net_win) + ((1 - p) * net_loss)

    return p, ev, risk, reward


def monte_carlo_equity(win_prob, reward, risk, trades=200):
    """
    Simulate equity curve using ACTUAL risk/reward from the question.
    Normalise to percentage of capital per trade (assume 1% risk per trade baseline,
    scaled by the actual RR ratio so the chart reflects the real setup).
    """
    equity_curve = []
    capital      = 1.0

    # Normalise: treat risk as 1 unit, reward scales accordingly
    rr           = reward / risk if risk > 0 else 1.0
    risk_pct     = 0.01                  # risk 1% of capital per trade
    reward_pct   = risk_pct * rr         # reward scales with actual RR

    for _ in range(trades):
        if np.random.rand() < win_prob:
            capital *= (1 + reward_pct)
        else:
            capital *= (1 - risk_pct)
        equity_curve.append(round(capital, 4))

    return equity_curve


PROMPT_BUILDER = """You are NUROX — a world-class all-rounder intelligence system. You are an expert in every domain: mathematics, science, trading, finance, coding, history, logic, language, and more.

## STRICT RULES — NEVER BREAK THESE:

### ACCURACY
- Every answer must be 100% factually correct. Zero tolerance for wrong answers.
- For ANY math question: compute it step-by-step, show the working, give the exact final answer.
- NEVER say "approximately" or "around" for calculations that have exact answers.
- If you are unsure about something, say so clearly. Never hallucinate facts.

### MATH & CALCULATIONS
- Always show: Given → Formula → Step-by-step working → Final Answer
- Basic arithmetic must be computed exactly. 1+1=2, not "around 2".
- Express fractions, ratios, and percentages in their simplest correct form.
- Double-check every calculation before responding.

### TRADING & FINANCE (when relevant)
- Risk:Reward (RR) ratio is ALWAYS written as 1:X — Risk is ALWAYS 1, Reward is X.
  - Formula: X = Reward / Risk. Then express as 1:X.
  - risk $1, make $2 → 2/1 = 2 → RR = 1:2 ✅ NEVER write this as 2:1 ❌
  - risk $1, make $3 → 3/1 = 3 → RR = 1:3 ✅
  - risk $50, make $150 → 150/50 = 3 → RR = 1:3 ✅
- Break-even win rate = Risk / (Risk + Reward) × 100%
- EV = (Win% × Reward) − (Loss% × Risk)
- NEVER express RR as X:1 unless the user explicitly asks for reward-to-risk ratio.

### FORMAT
- Use **bold** ONLY for the final answer or the most critical insight.
- Use clear headers for multi-part answers.
- Be concise — no filler, no fluff, no unnecessary words.
- Use 1-2 emojis maximum. Do not spam emojis.
- Never start with "Great question!" or similar sycophantic phrases.

### CORRECTIONS
- If the user states something factually wrong, correct them directly and politely with proof.
- Show them why their assumption is wrong and give the right answer.
"""

PROMPT_AUDITOR = """You are NUROX Auditor — the final verification layer of the NUROX intelligence system.

Your job:
1. Read the ORIGINAL QUESTION and the BUILDER's analysis below.
2. Verify the answer is 100% correct. If the builder made ANY error, fix it.
3. Give a final, clean, definitive answer that directly addresses the original question.

## STRICT RULES:
- If the builder's math is wrong, recalculate and give the correct answer.
- If the builder was vague, be specific.
- If the builder was too long, summarize to the essential point.
- Always end with one clear **Final Answer:** in bold.
- No fluff. No "the builder said...". Just the verified truth.
- Maximum 200 words unless the question genuinely requires more detail.

## TRADING CONVENTION CHECK:
- RR ratio is ALWAYS 1:X format. If builder wrote it as X:1, correct it to 1:X.
- risk $1 make $2 = 1:2 RR. If builder said 2:1, override it with 1:2.
- Break-even win rate = Risk / (Risk + Reward) × 100%. Verify this if present.
"""


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

    # Builder — temperature 0.0 for maximum accuracy (no creativity on facts)
    builder = await call_llm(
        GROQ_API_KEY_AI1,
        PROMPT_BUILDER,
        [{"role": "user", "content": question}],
        0.0,  # Zero temperature = most deterministic, accurate output
    )

    transcript.append(DebateMessage(role="🧠 Builder", content=builder))

    deterministic = simulation_block = simulation_data = risk_alerts = None
    authority = "LLM"
    confidence = "High"  # Default High — auditor verifies everything

    if mode == "quant":
        p, ev, risk, reward = deterministic_engine(question)

        if p is not None:
            rr_ratio = reward / risk if risk > 0 else 0

            deterministic = (
                f"🎯 **Break-even Win Rate** = {p*100:.2f}% | "
                f"**Expected Value (EV)** = {ev:.4f} | "
                f"**RR Ratio** = 1:{rr_ratio:.2f}"
            )

            # Monte Carlo uses ACTUAL risk/reward from the question — dynamic chart
            equity_curve   = monte_carlo_equity(p, reward=reward, risk=risk)
            simulation_data  = equity_curve
            simulation_block = (
                f"📊 **Monte Carlo Simulation** — {len(equity_curve)} trades "
                f"at 1:{rr_ratio:.2f} RR with {p*100:.2f}% break-even win rate."
            )

            risk_alerts = (
                "🟢 **Stable Risk Profile**"
                if p > 0.4
                else "🔴 **High Risk Profile**"
            )

            authority = "Deterministic + LLM"

    # Auditor — sees BOTH the original question AND the builder's answer
    # This way it can fact-check and correct any mistakes
    auditor_input = (
        f"## Original Question:\n{question}\n\n"
        f"## Builder's Analysis:\n{builder}"
    )

    final_answer = await call_llm(
        GROQ_API_KEY_AI2,
        PROMPT_AUDITOR,
        [{"role": "user", "content": auditor_input}],
        0.0,  # Zero temperature = no hallucination in verification
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