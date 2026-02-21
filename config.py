import os
from dotenv import load_dotenv

load_dotenv()

# ======================
# GROQ API KEYS
# ======================

GROQ_API_KEY_AI1 = os.getenv("GROQ_API_KEY_AI1")
GROQ_API_KEY_AI2 = os.getenv("GROQ_API_KEY_AI2")

MODEL_NAME = "llama-3.1-8b-instant"

# ======================
# JWT SECURITY
# ======================

SECRET_KEY = os.getenv("SECRET_KEY", "SUPER_SECRET_DEV_KEY")
ALGORITHM = "HS256"