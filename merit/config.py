"""Layer 1 configuration.

Everything that changes the shape of the experiment lives here, so a run is
reproducible from (config, seed) alone.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json


@dataclass
class Config:
    # --- experiment identity -------------------------------------------------
    run_id: str = "dev"
    seed: int = 20261006

    # --- retrieval -----------------------------------------------------------
    top_k: int = 4                      # candidates considered per turn
    max_bundle_tokens: int = 400        # token budget for injected memory

    # --- masking (the mechanism) --------------------------------------------
    mask_p: float = 0.15                # P(drop) per eligible candidate
    protect_bundle: bool = False        # if True, never drop every candidate

    # --- model ---------------------------------------------------------------
    backend: str = "mock"               # "mock" | "groq"
    model: str = "openai/gpt-oss-20b"
    temperature: float = 0.0
    max_tokens: int = 640
    reasoning_effort: str | None = None  # "low" | "medium" | "high" for gpt-oss on Groq

    # --- use detection -------------------------------------------------------
    lexical_min_len: int = 4            # min token length for rare n-gram match
    semantic_enabled: bool = False      # needs sentence-transformers
    semantic_model: str = "all-MiniLM-L6-v2"
    semantic_threshold: float = 0.45

    # --- harness state (recorded, not yet varied) ----------------------------
    planner_active: bool = False
    verifier_active: bool = True
    policy: str = "relevance"           # set by the runner from the policy object

    # --- storage -------------------------------------------------------------
    db_path: str = "data/merit.db"

    # --- budget guard --------------------------------------------------------
    max_turns: int = 60
    shuffle_epochs: bool = False        # reshuffle task order each pass (seeded)
    max_total_tokens: int = 250_000     # hard stop across the whole run

    def harness_state(self) -> str:
        bits = []
        bits.append(f"planner={int(self.planner_active)}")
        bits.append(f"verifier={int(self.verifier_active)}")
        bits.append(f"policy={self.policy}")
        return ",".join(bits)

    def resolved_db_path(self) -> str:
        """Anchor db_path to the repo root, not the caller's cwd.

        Without this, a notebook in notebooks/ writes to notebooks/data/ while
        a script at the root writes to data/, and you end up with two databases.
        """
        from pathlib import Path
        p = Path(self.db_path)
        if p.is_absolute():
            return str(p)
        root = Path(__file__).resolve().parent.parent
        full = root / p
        full.parent.mkdir(parents=True, exist_ok=True)
        return str(full)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)
