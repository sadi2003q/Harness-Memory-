# ---------- Model ----------
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"   # change this to any HF model
MAX_NEW_TOKENS = 200

# ---------- Agent loop ----------
MAX_LOOP_STEPS = 3          # guardrail: agent stops after 3 steps
CHAT_HISTORY_LIMIT = 6      # how many past messages to remember

# ---------- Memory ----------
DATA_FOLDER = "memory_data"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_TOP_K = 3
SEMANTIC_MIN_SCORE = 0.3
EPISODIC_RECENT = 3

# ---------- Tracer limits ----------
MAX_RESPONSE_SECONDS = 30
MAX_TOKENS_PER_ANSWER = 1500