# MERIT — utility-credited memory for agent harnesses

MERIT asks a question existing agent-memory systems don't: **did this memory
entry actually change what the agent did, and did that help?** Every entry gets
a measured effect on task outcome, estimated from randomised withholding during
normal operation, and the store's promote / demote / forget decisions are made
from that estimate instead of from how often an entry gets retrieved.

## Layout

| Path | What it is |
| --- | --- |
| `merit/ledger.py` | Layer 1 — the inject ledger (SQLite) |
| `merit/runner.py` | the turn loop: retrieve, mask, inject, call, score, close |
| `merit/credit.py` | Layer 2 — utility estimation (regression; empirical-Bayes for large stores) |
| `merit/policy.py` | Layer 3 — utility-ranked retrieval (negative result, kept for the record) |
| `merit/lifecycle.py` | Layer 4 — promotion, demotion, forgetting |
| `merit/locomo.py`, `merit/stage5.py` | Stage 5 — the real-data test on LoCoMo |
| `merit/clients.py` | mock client + Groq client with a key pool and pause/resume on daily limits |
| `notebooks/layer1–4.ipynb` | development on the planted fixture |
| `notebooks/stage5.ipynb` | the pre-registered real-data test |
| `PREREGISTRATION.md` | hypotheses and decision rule, fixed before the confirmatory run |

## Setup

```bash
conda activate merit
pip install -r requirements.txt
cp .env.example .env      # then paste your Groq keys into .env
```

## Status

| Stage | Result |
| --- | --- |
| Layer 1 — instrument | passes on mock and live (floor 0.0, live 0.85) |
| Layer 2 — credit | recovers planted utility; zero false promotions at 200+ turns |
| Layer 3 — retrieval | **negative**: global utility is the wrong signal for per-query ranking |
| Layer 4 — lifecycle | utility beats recall frequency on promotion and forgetting (fixture) |
| Stage 5 — real data | pre-registered; see `PREREGISTRATION.md` |
