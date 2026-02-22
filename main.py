from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
import httpx, re, numpy as np, os, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import GROQ_API_KEY_AI1, GROQ_API_KEY_AI2, MODEL_NAME
from database.connection import engine
from database.models import Base, User, DebateHistory
from auth.routes import router as auth_router, get_current_user, get_db
from services.usage_limiter import UsageLimiter
from admin.routes import router as admin_router

# ─────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="NUROX V6.3 Intelligence Platform")

FRONTEND_URL = os.getenv("FRONTEND_URL")
origins = [FRONTEND_URL] if FRONTEND_URL else ["http://localhost:3000", "http://127.0.0.1:3000"]

app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)

app.include_router(auth_router)
app.include_router(admin_router)


# ─────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# LLM CALLER
# ─────────────────────────────────────────────────────────────
async def call_llm(api_key: str, system_prompt: str, messages: list, temperature: float = 0.0) -> str:
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "system", "content": system_prompt}] + messages,
                "temperature": temperature,
                "max_tokens": 1500,
            },
        )
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"LLM Error: {r.text}")
    data = r.json()
    if not data.get("choices"):
        raise HTTPException(status_code=500, detail="Invalid LLM response.")
    return data["choices"][0]["message"]["content"].strip()


# ─────────────────────────────────────────────────────────────
# MODE DETECTION
# ─────────────────────────────────────────────────────────────
QUANT_KEYWORDS = [
    "risk", "reward", "win rate", "winrate", "break even", "breakeven",
    "rr ratio", "risk reward", "r:r", "pip", "lot size", "position size",
    "drawdown", "expectancy", "kelly", "slippage", "transaction cost",
    "make", "profit", "loss", "trade", "trading"
]

def detect_mode(question: str) -> str:
    q = question.lower()
    # Only quant if it has trading context + numbers
    has_numbers = bool(re.search(r'\d', q))
    has_keyword = any(k in q for k in QUANT_KEYWORDS)
    return "quant" if (has_numbers and has_keyword) else "general"


# ─────────────────────────────────────────────────────────────
# PYTHON MATH ENGINE
# All trading calculations done here — LLM never touches numbers
# ─────────────────────────────────────────────────────────────
def compute_quant(question: str) -> dict | None:
    """
    Extract numbers and compute everything in Python.
    Returns a dict of pre-written CORRECT statements the LLM just quotes.
    """
    nums = list(map(float, re.findall(r"\d+\.?\d*", question)))
    if len(nums) < 2:
        return None

    risk        = nums[0]
    reward      = nums[1]
    transaction = nums[2] if len(nums) >= 3 else 0.0
    slippage    = nums[3] if len(nums) >= 4 else 0.0

    if risk <= 0 or reward <= 0:
        return None

    net_win  = reward - transaction - slippage
    net_loss = risk + transaction          # positive number = what you lose

    denom = net_win + net_loss             # net_win + net_loss (both positive now)
    if denom == 0:
        return None

    breakeven_prob  = net_loss / denom     # min win% to not lose money
    ev              = (breakeven_prob * net_win) - ((1 - breakeven_prob) * net_loss)

    # RR RATIO — always 1:X where X = reward/risk
    # e.g. risk 50, reward 100 → X = 100/50 = 2 → "1:2"
    rr_x        = reward / risk
    rr_string   = f"1:{rr_x:.2f}".rstrip('0').rstrip('.')  # "1:2" not "1:2.00"
    if '.' not in rr_string.split(':')[1]:
        pass  # already clean

    is_stable   = breakeven_prob > 0.4

    # Pre-write the correct sentence so LLM just copies it, never reformulates
    rr_sentence = (
        f"If you risk {risk:.0f} to make {reward:.0f}, "
        f"the Risk:Reward ratio is {rr_string} "
        f"(you risk 1 unit to make {rr_x:.2f} units)."
    )

    return {
        "risk":           risk,
        "reward":         reward,
        "transaction":    transaction,
        "slippage":       slippage,
        "net_win":        round(net_win, 4),
        "net_loss":       round(net_loss, 4),
        "rr_x":           round(rr_x, 4),
        "rr_string":      rr_string,
        "rr_sentence":    rr_sentence,
        "breakeven_pct":  round(breakeven_prob * 100, 2),
        "ev":             round(ev, 4),
        "is_stable":      is_stable,
        "risk_label":     "Stable" if is_stable else "High Risk",
    }


# ─────────────────────────────────────────────────────────────
# MONTE CARLO — dynamic per actual RR
# ─────────────────────────────────────────────────────────────
def monte_carlo(quant: dict, trades: int = 200) -> list[float]:
    win_prob   = quant["breakeven_pct"] / 100
    rr         = quant["rr_x"]
    risk_pct   = 0.01          # 1% capital per trade
    reward_pct = risk_pct * rr

    capital = 1.0
    curve   = []
    for _ in range(trades):
        capital = capital * (1 + reward_pct) if np.random.rand() < win_prob else capital * (1 - risk_pct)
        curve.append(round(capital, 4))
    return curve


# ─────────────────────────────────────────────────────────────
# PROMPTS
# Key principle: for quant questions, LLM never sees raw numbers.
# It only receives pre-written correct statements to explain.
# ─────────────────────────────────────────────────────────────

SYSTEM_BUILDER_GENERAL = """You are NUROX AI — Agent 1 in a two-AI debate system. You are a world-class expert across ALL domains: mathematics, science, finance, trading, coding, history, logic, language and everything else.

DEBATE ROLE: You are the ANALYST. You build the initial answer. A second AI (Auditor) will review you.

STRICT RULES:
- Answer with 100% accuracy. Never guess or hallucinate.
- For math: show Given → Formula → Step-by-step working → Final Answer.
- Be direct. No "Great question!" or filler phrases ever.
- Use **bold** only for the final answer or key insight.
- Max 2 emojis total. No emoji spam.
- If the user is factually wrong, correct them politely with proof.
- Cover ALL domains with expert-level depth.
"""

SYSTEM_AUDITOR_GENERAL = """You are NUROX AI — Agent 2 in a two-AI debate system. You are the AUDITOR and FINAL AUTHORITY.

DEBATE ROLE: You receive the original question and Agent 1's analysis. You verify, challenge, and deliver the final verdict.

STRICT RULES:
- If Agent 1 is correct → confirm and give a clean summary.
- If Agent 1 made ANY error → state the error and give the correct answer.
- If Agent 1 was vague → be precise.
- If Agent 1 was too long → distill to the essential truth.
- End with **Final Answer:** in bold — always.
- No fluff. Pure signal. Be authoritative.
- Max 150 words unless the topic genuinely needs more.
"""


def build_quant_builder_prompt(q: dict) -> str:
    """
    For quant: LLM gets ONLY pre-written correct statements.
    It NEVER sees raw numbers to calculate from.
    It only explains the meaning and implications.
    """
    return f"""You are NUROX AI — Agent 1 in a two-AI debate system. Expert trading analyst.

DEBATE ROLE: Explain and analyse the following VERIFIED trading setup to the user.

== VERIFIED RESULTS (already computed — just explain these, do not recalculate) ==
{q['rr_sentence']}
Break-even Win Rate: {q['breakeven_pct']}% (you need to win at least {q['breakeven_pct']}% of trades to not lose money)
Expected Value (EV): {q['ev']} ({"positive = profitable edge" if q['ev'] > 0 else "negative = losing edge long-term"})
Risk Profile: {q['risk_label']}

== YOUR JOB ==
1. State the verified results clearly and correctly.
2. Explain what each metric MEANS for this trader in plain English.
3. Analyse: is this a good or bad trading setup and why?
4. Give a practical trading recommendation.

RULES:
- Do NOT recalculate anything. The numbers above are final and correct.
- Do NOT change the RR ratio. It is {q['rr_string']} — never write it any other way.
- Use **bold** for key points only. Be concise. Max 2 emojis.
"""


def build_quant_auditor_prompt(q: dict) -> str:
    return f"""You are NUROX AI — Agent 2 in a two-AI debate system. You are the AUDITOR and FINAL AUTHORITY.

== GROUND TRUTH (Python-verified, 100% correct) ==
RR Ratio:          {q['rr_string']} ← this is the ONLY correct format
Break-even Rate:   {q['breakeven_pct']}%
Expected Value:    {q['ev']}
Risk Profile:      {q['risk_label']}

DEBATE ROLE:
- Read Agent 1's analysis and the original question.
- If Agent 1 stated the RR as anything other than {q['rr_string']}, it is WRONG — correct it.
- If Agent 1 got any numbers wrong, override with the ground truth above.
- If Agent 1 is correct, confirm and add any missing insight.
- End with a crisp **Final Answer:** covering RR ratio, break-even rate, EV, and recommendation.
- Be the definitive last word. Under 200 words.
"""


# ─────────────────────────────────────────────────────────────
# MAIN DEBATE ENDPOINT
# ─────────────────────────────────────────────────────────────
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

    mode     = detect_mode(question)
    quant    = compute_quant(question) if mode == "quant" else None
    transcript = []

    # ── Deterministic outputs (Python math — always correct)
    deterministic    = None
    simulation_block = None
    simulation_data  = None
    risk_alerts      = None
    authority        = "LLM"
    confidence       = "High"

    if quant:
        deterministic = (
            f"**RR Ratio** = {quant['rr_string']} | "
            f"**Break-even Win Rate** = {quant['breakeven_pct']}% | "
            f"**Expected Value (EV)** = {quant['ev']}"
        )
        curve           = monte_carlo(quant)
        simulation_data = curve
        simulation_block = (
            f"**Monte Carlo** — 200 trades at {quant['rr_string']} RR | "
            f"Break-even: {quant['breakeven_pct']}%"
        )
        risk_alerts = "🟢 **Stable Risk Profile**" if quant["is_stable"] else "🔴 **High Risk Profile**"
        authority   = "Deterministic + LLM"

    # ── AGENT 1: Builder
    builder_sys = build_quant_builder_prompt(quant) if quant else SYSTEM_BUILDER_GENERAL
    builder     = await call_llm(
        GROQ_API_KEY_AI1,
        builder_sys,
        [{"role": "user", "content": question}],
        temperature=0.0,
    )
    transcript.append(DebateMessage(role="🧠 Builder", content=builder))

    # ── AGENT 2: Auditor (sees question + builder response)
    auditor_sys   = build_quant_auditor_prompt(quant) if quant else SYSTEM_AUDITOR_GENERAL
    auditor_input = f"Original Question:\n{question}\n\nAgent 1 Analysis:\n{builder}"
    final_answer  = await call_llm(
        GROQ_API_KEY_AI2,
        auditor_sys,
        [{"role": "user", "content": auditor_input}],
        temperature=0.0,
    )

    # ── Save history
    db.add(DebateHistory(
        user_id=current_user.id,
        question=question,
        final_answer=final_answer,
        mode=mode,
    ))
    db.commit()

    logger.info(f"Debate | user={current_user.id} | mode={mode} | q={question[:60]}")

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


# ─────────────────────────────────────────────────────────────
# HISTORY
# ─────────────────────────────────────────────────────────────
@app.get("/history")
def get_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return (
        db.query(DebateHistory)
        .filter(DebateHistory.user_id == current_user.id)
        .order_by(DebateHistory.created_at.desc())
        .all()
    )


# ─────────────────────────────────────────────────────────────
# USAGE
# ─────────────────────────────────────────────────────────────
@app.get("/usage")
def get_usage(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    from database.models import UsageTracking, PLAN_LIMITS
    tracking = db.query(UsageTracking).filter_by(user_id=current_user.id).first()
    limits   = PLAN_LIMITS.get(current_user.plan or "free", {})

    if not tracking:
        return {
            "plan": current_user.plan,
            "debates_today": 0, "debates_this_month": 0,
            "daily_limit": limits.get("daily_debates"),
            "monthly_limit": limits.get("monthly_debates"),
        }
    return {
        "plan":               current_user.plan,
        "debates_today":      tracking.debates_today,
        "daily_limit":        limits.get("daily_debates"),
        "debates_this_month": tracking.debates_this_month,
        "monthly_limit":      limits.get("monthly_debates"),
        "total_debates":      tracking.total_debates,
    }


# ─────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "NUROX V6.3 Running ✅"}
