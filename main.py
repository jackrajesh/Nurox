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
logger = logging.getLogger(__name__)

from config import GROQ_API_KEY_AI1, GROQ_API_KEY_AI2, MODEL_NAME
from database.connection import engine
from database.models import Base, User, DebateHistory
from auth.routes import router as auth_router, get_current_user, get_db
from services.usage_limiter import UsageLimiter
from admin.routes import router as admin_router


# ─────────────────────────────────────────
# App Init
# ─────────────────────────────────────────
app = FastAPI(title="NUROX V6.3 Intelligence Platform")

FRONTEND_URL = os.getenv("FRONTEND_URL")
origins = [FRONTEND_URL] if FRONTEND_URL else [
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

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)

app.include_router(auth_router)
app.include_router(admin_router)


# ─────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# LLM Caller
# ─────────────────────────────────────────
async def call_llm(api_key: str, system_prompt: str, messages: list, temperature: float = 0.0) -> str:
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
        raise HTTPException(status_code=500, detail="Invalid LLM response.")
    return data["choices"][0]["message"]["content"].strip()


# ─────────────────────────────────────────
# Mode Detection
# ─────────────────────────────────────────
QUANT_KEYWORDS = ["risk", "reward", "win rate", "winrate", "break even", "breakeven",
                  "transaction", "slippage", "rr ratio", "risk reward", "pip", "lot size",
                  "position size", "drawdown", "expectancy", "kelly"]

def detect_mode(question: str) -> str:
    q = question.lower()
    return "quant" if any(k in q for k in QUANT_KEYWORDS) else "general"


# ─────────────────────────────────────────
# Deterministic Engine
# Python computes all math — LLM only explains
# ─────────────────────────────────────────
def deterministic_engine(question: str) -> dict | None:
    nums = list(map(float, re.findall(r"\d+\.?\d*", question)))
    if len(nums) < 2:
        return None

    risk        = nums[0]
    reward      = nums[1]
    transaction = nums[2] if len(nums) >= 3 else 0.0
    slippage    = nums[3] if len(nums) >= 4 else 0.0

    if risk <= 0 or reward <= 0:
        return None

    # All formulas computed in Python — never touched by LLM
    net_win     = reward - transaction - slippage
    net_loss    = -(risk + transaction)
    denom       = net_win - net_loss

    if denom == 0:
        return None

    breakeven_prob  = -net_loss / denom              # minimum win rate to break even
    ev              = (breakeven_prob * net_win) + ((1 - breakeven_prob) * net_loss)
    rr_ratio        = reward / risk                  # e.g. 2.0 for 1:2
    rr_string       = f"1:{rr_ratio:.2f}"           # always 1:X format

    return {
        "risk":           risk,
        "reward":         reward,
        "transaction":    transaction,
        "slippage":       slippage,
        "net_win":        net_win,
        "net_loss":       net_loss,
        "breakeven_pct":  round(breakeven_prob * 100, 2),
        "ev":             round(ev, 4),
        "rr_ratio":       round(rr_ratio, 2),
        "rr_string":      rr_string,
        "is_stable":      breakeven_prob > 0.4,
    }


# ─────────────────────────────────────────
# Monte Carlo — dynamic per question
# ─────────────────────────────────────────
def monte_carlo_equity(quant: dict, trades: int = 200) -> list[float]:
    win_prob    = quant["breakeven_pct"] / 100
    rr          = quant["rr_ratio"]
    risk_pct    = 0.01                 # 1% capital risked per trade
    reward_pct  = risk_pct * rr        # scales with actual RR

    capital     = 1.0
    curve       = []
    for _ in range(trades):
        if np.random.rand() < win_prob:
            capital *= (1 + reward_pct)
        else:
            capital *= (1 - risk_pct)
        curve.append(round(capital, 4))
    return curve


# ─────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────

# BUILDER: First debater — deep analysis, no math (Python handles math)
def build_prompt_builder(mode: str, quant: dict | None) -> str:
    base = """You are NUROX Builder — the first AI in a two-AI debate system. Your job is deep analysis and reasoning.

## YOUR ROLE IN THE DEBATE:
- You are the ANALYST. You provide thorough reasoning, context, and explanation.
- A second AI (the Auditor) will review your work and either confirm or correct you.
- So be thorough, structured, and honest about any uncertainty.

## ABSOLUTE RULES:
1. Be 100% factually accurate. Never guess. Never hallucinate.
2. For math questions: show Given → Formula → Working → Answer clearly.
3. Be direct. No "Great question!", no filler, no waffle.
4. Use **bold** only for the final answer or critical insight.
5. Use headers for multi-part answers. Keep it clean.
6. Max 2 emojis. Not every line needs one.
7. If something is wrong in the question, correct it with proof.

## DOMAINS YOU COVER:
Mathematics, Physics, Chemistry, Biology, History, Geography, Economics,
Trading & Finance, Programming & Code, Logic & Reasoning, Language & Grammar,
Science & Technology, General Knowledge — everything.
"""

    if mode == "quant" and quant:
        # Inject Python-computed facts directly — LLM only explains, never recalculates
        q = quant
        math_facts = f"""
## ⚠️ VERIFIED MATH FACTS (computed by Python — DO NOT recalculate, DO NOT override):
- Risk         = {q['risk']}
- Reward       = {q['reward']}
- Transaction  = {q['transaction']}
- Slippage     = {q['slippage']}
- Net Win      = {q['net_win']}
- Net Loss     = {q['net_loss']}
- **RR Ratio   = {q['rr_string']}** ← ALWAYS in 1:X format, NEVER X:1
- **Break-even Win Rate = {q['breakeven_pct']}%**
- **Expected Value (EV) = {q['ev']}**
- Risk Profile = {"Stable ✅" if q['is_stable'] else "High Risk ⚠️"}

YOUR JOB: Explain these results clearly. Do NOT recalculate them. Do NOT second-guess them.
Explain what the RR ratio means, what the break-even win rate means for this trader,
and whether this is a good or bad setup with reasoning.

IMPORTANT: RR ratio is ALWAYS expressed as 1:X (Risk:Reward).
{q['rr_string']} means the trader risks 1 unit to make {q['rr_ratio']} units.
NEVER write it as {q['rr_ratio']}:1.
"""
        return base + math_facts

    return base + """
## FOR GENERAL QUESTIONS:
- Answer directly and completely.
- Show full working for any math.
- Cite reasoning for factual claims.
- Be the best expert in the room.
"""


# AUDITOR: Second debater — verifies, challenges, delivers final verdict
def build_prompt_auditor(mode: str, quant: dict | None) -> str:
    base = """You are NUROX Auditor — the second AI in a two-AI debate system. You are the final authority.

## YOUR ROLE IN THE DEBATE:
- You receive the original question AND the Builder's full analysis.
- Your job: verify, challenge, correct if needed, then deliver the definitive final answer.
- You are the last word. What you say is what NUROX outputs as truth.

## ABSOLUTE RULES:
1. Read the Builder's analysis carefully.
2. If the Builder is correct → confirm and summarize cleanly.
3. If the Builder made ANY error → correct it explicitly and explain why.
4. If the Builder was too vague → add precision.
5. If the Builder was too long → distill to the core truth.
6. Always end with a clear **Final Answer:** section in bold.
7. Be authoritative but fair. No "the builder said..." — just deliver truth.
8. No fluff. No sycophancy. Pure signal.
"""

    if mode == "quant" and quant:
        q = quant
        math_facts = f"""
## ⚠️ GROUND TRUTH (Python-verified — these are correct, non-negotiable):
- RR Ratio         = {q['rr_string']} ← this is correct, DO NOT change it
- Break-even Rate  = {q['breakeven_pct']}%
- Expected Value   = {q['ev']}
- Risk Profile     = {"Stable" if q['is_stable'] else "High Risk"}

If the Builder stated different numbers → they are WRONG. Override with the above.
If the Builder used X:1 format for RR → correct it to {q['rr_string']}.

YOUR FINAL ANSWER must include:
1. Confirmed correct values (from ground truth above)
2. What this means practically for the trader
3. Clear recommendation: is this a good setup or not, and why
"""
        return base + math_facts

    return base + """
## FOR GENERAL QUESTIONS:
- Verify the Builder's reasoning and facts.
- Correct any errors with explanation.
- Deliver one clean, definitive Final Answer.
- Keep it under 150 words unless complexity genuinely demands more.
"""


# ─────────────────────────────────────────
# Main Debate Endpoint
# ─────────────────────────────────────────
@app.post("/debate", response_model=DebateResponse)
async def debate(
    req: DebateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    limiter    = UsageLimiter(db)
    usage_info = limiter.check_and_consume(current_user)

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    mode       = detect_mode(question)
    transcript = []

    # ── Step 1: Run deterministic engine FIRST (Python math, never LLM)
    quant           = None
    deterministic   = None
    simulation_block= None
    simulation_data = None
    risk_alerts     = None
    authority       = "LLM"
    confidence      = "High"

    if mode == "quant":
        quant = deterministic_engine(question)
        if quant:
            deterministic = (
                f"**RR Ratio** = {quant['rr_string']} | "
                f"**Break-even Win Rate** = {quant['breakeven_pct']}% | "
                f"**Expected Value (EV)** = {quant['ev']}"
            )
            equity_curve    = monte_carlo_equity(quant)
            simulation_data = equity_curve
            simulation_block= (
                f"**Monte Carlo** — {len(equity_curve)} trades at "
                f"{quant['rr_string']} RR | Break-even: {quant['breakeven_pct']}%"
            )
            risk_alerts = (
                "🟢 **Stable Risk Profile**" if quant["is_stable"]
                else "🔴 **High Risk Profile**"
            )
            authority   = "Deterministic + LLM"

    # ── Step 2: Builder analyses (with verified math pre-injected)
    builder_prompt = build_prompt_builder(mode, quant)
    builder = await call_llm(
        GROQ_API_KEY_AI1,
        builder_prompt,
        [{"role": "user", "content": question}],
        temperature=0.0,
    )
    transcript.append(DebateMessage(role="🧠 Builder", content=builder))

    # ── Step 3: Auditor verifies (also gets verified math)
    auditor_prompt = build_prompt_auditor(mode, quant)
    auditor_input  = (
        f"## Original Question:\n{question}\n\n"
        f"## Builder's Analysis:\n{builder}"
    )
    final_answer = await call_llm(
        GROQ_API_KEY_AI2,
        auditor_prompt,
        [{"role": "user", "content": auditor_input}],
        temperature=0.0,
    )

    # ── Step 4: Save to history
    db.add(DebateHistory(
        user_id=current_user.id,
        question=question,
        final_answer=final_answer,
        mode=mode,
    ))
    db.commit()

    logger.info(f"Debate | User: {current_user.id} | Mode: {mode} | Q: {question[:60]}")

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


# ─────────────────────────────────────────
# History
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# Usage
# ─────────────────────────────────────────
@app.get("/usage")
def get_usage(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from database.models import UsageTracking, PLAN_LIMITS

    tracking = db.query(UsageTracking).filter_by(user_id=current_user.id).first()
    limits   = PLAN_LIMITS.get(current_user.plan or "free", {})

    if not tracking:
        return {
            "plan":              current_user.plan,
            "debates_today":     0,
            "debates_this_month":0,
            "daily_limit":       limits.get("daily_debates"),
            "monthly_limit":     limits.get("monthly_debates"),
        }

    return {
        "plan":               current_user.plan,
        "debates_today":      tracking.debates_today,
        "daily_limit":        limits.get("daily_debates"),
        "debates_this_month": tracking.debates_this_month,
        "monthly_limit":      limits.get("monthly_debates"),
        "total_debates":      tracking.total_debates,
    }


# ─────────────────────────────────────────
# Health
# ─────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "NUROX V6.3 Running ✅"}
