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

SEMANTIC_MAX_INSTANCES = 6        # keep up to 6 recent facts per topic (your "yellow -> teal" idea)
SEMANTIC_DUPLICATE_SCORE = 0.75   # how similar counts as "same topic"

HISTORICAL_TOP_K = 3
HISTORICAL_MIN_SCORE = 0.3

ABSTENTION_MARKERS = [
    "don't know", "do not know", "don't have", "no information",
    "not sure", "unable to determine", "cannot determine", "no access",
]

# ---------- Boolean Switch ----------
STORE_LIVE_CONVERSATION_IN_HISTORICAL = True   # if True, the live conversation is stored in historical memory


# ---------- Tracer limits ----------
MAX_RESPONSE_SECONDS = 30
MAX_TOKENS_PER_ANSWER = 1500