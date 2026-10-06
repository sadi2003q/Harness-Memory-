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

    # --- use detection -------------------------------------------------------
    lexical_min_len: int = 4            # min token length for rare n-gram match
    semantic_enabled: bool = False      # needs sentence-transformers
    semantic_model: str = "all-MiniLM-L6-v2"
    semantic_threshold: float = 0.45

    # --- harness state (recorded, not yet varied) ----------------------------
    planner_active: bool = False
    verifier_active: bool = True

    # --- storage -------------------------------------------------------------
    db_path: str = "data/merit.db"

    # --- budget guard --------------------------------------------------------
    max_turns: int = 60
    max_total_tokens: int = 250_000     # hard stop across the whole run

    def harness_state(self) -> str:
        bits = []
        bits.append(f"planner={int(self.planner_active)}")
        bits.append(f"verifier={int(self.verifier_active)}")
        return ",".join(bits)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)
