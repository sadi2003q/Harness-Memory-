"""The Layer 1 turn loop.

Per turn:
  1. retrieve candidates
  2. mask each candidate independently with probability p   <- the mechanism
  3. assemble the survivors into a token-budgeted bundle
  4. OPEN one ledger row per candidate (masked ones included)
  5. call the model
  6. detect use, score the outcome
  7. CLOSE every row of the turn with the same outcome_score

Layer 1 produces no decisions and improves nothing. It produces the dataset.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from .clients import BudgetExhausted, Completion, build_client
from .config import Config
from .detect import build_detector
from .ledger import Ledger, new_id
from .policy import RelevancePolicy
from .store import MemoryStore, Scored, count_tokens
from .tasks import SYSTEM_PROMPT, Task, build_store, build_tasks


@dataclass
class TurnResult:
    turn_id: str
    task_id: str
    n_candidates: int
    n_injected: int
    outcome_score: float
    turn_tokens: int
    response: str
    error: Optional[str] = None


def render_bundle(injected: list[Scored]) -> str:
    if not injected:
        return ""
    lines = ["Project notes:"]
    for s in injected:
        lines.append(f"- {s.entry.text}")
    return "\n".join(lines)


def mask_candidates(candidates: list[Scored], rng: random.Random, p: float,
                    protect: bool = False) -> list[bool]:
    """Independent Bernoulli(p) drop per candidate. True == masked (withheld)."""
    masked = [rng.random() < p for _ in candidates]
    if protect and candidates and all(masked):
        masked[rng.randrange(len(masked))] = False
    return masked


class Layer1Runner:
    def __init__(self, cfg: Config, store: Optional[MemoryStore] = None,
                 tasks: Optional[list[Task]] = None,
                 ledger: Optional[Ledger] = None,
                 client=None, policy=None, system_prompt: Optional[str] = None,
                 resume: bool = False):
        self.cfg = cfg
        self.store = store or build_store()
        self.tasks = tasks or build_tasks()
        self.ledger = ledger or Ledger(cfg.resolved_db_path())
        self.policy = policy or RelevancePolicy(self.store)
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        cfg.policy = self.policy.name
        self.client = client if client is not None else build_client(cfg)
        self.detector = build_detector(cfg)
        self.rng = random.Random(cfg.seed)
        self.spent_tokens = 0
        self.stopped_early = False
        self.resume_from = 0
        if resume:
            self.ledger.start_run(cfg.run_id, cfg.to_json(), reset=False)
            dropped = self.ledger.drop_unsettled(cfg.run_id)
            self.resume_from, self.spent_tokens = self.ledger.settled_turns(cfg.run_id)
            if self.resume_from or dropped:
                print(f"[resume] {cfg.run_id}: {self.resume_from} turns already done"
                      + (f", {dropped} half-finished turn(s) discarded" if dropped else ""))
        else:
            self.ledger.start_run(cfg.run_id, cfg.to_json())

    # -- one turn ------------------------------------------------------------

    def run_turn(self, task: Task) -> TurnResult:
        cfg = self.cfg
        turn_id = new_id("trn")
        bundle_id = new_id("bnd")

        candidates = self.policy.candidates(task.prompt, cfg.top_k)
        masked = mask_candidates(candidates, self.rng, cfg.mask_p, cfg.protect_bundle)

        injected: list[Scored] = []
        positions: dict[str, int] = {}
        budget = cfg.max_bundle_tokens
        for cand, is_masked in zip(candidates, masked):
            if is_masked:
                continue
            if cand.entry.tokens > budget:
                continue
            positions[cand.entry.entry_id] = len(injected)
            injected.append(cand)
            budget -= cand.entry.tokens

        bundle = render_bundle(injected)
        bundle_tokens = count_tokens(bundle) if bundle else 0

        # ---- write 1: open ------------------------------------------------
        self.ledger.open_turn(
            turn_id=turn_id, run_id=cfg.run_id, task_id=task.task_id,
            task_family=task.family, bundle_id=bundle_id,
            n_candidates=len(candidates), n_injected=len(injected),
            bundle_tokens=bundle_tokens, harness_state=cfg.harness_state(),
            model=cfg.model, backend=cfg.backend, seed=cfg.seed,
        )
        self.ledger.open_rows([
            {
                "run_id": cfg.run_id, "turn_id": turn_id, "task_id": task.task_id,
                "task_family": task.family, "entry_id": c.entry.entry_id,
                "bundle_id": bundle_id, "masked": m, "mask_p": cfg.mask_p,
                "position": positions.get(c.entry.entry_id),
                "entry_tokens": c.entry.tokens, "retrieval_rank": c.rank,
                "retrieval_score": c.score, "harness_state": cfg.harness_state(),
                "model": cfg.model, "seed": cfg.seed,
            }
            for c, m in zip(candidates, masked)
        ])

        # ---- the call ------------------------------------------------------
        user = f"{bundle}\n\nQuestion: {task.prompt}" if bundle else f"Question: {task.prompt}"
        comp: Completion = self.client.complete(self.system_prompt, user)
        self.spent_tokens += comp.total_tokens

        # ---- detect + score -------------------------------------------------
        empty = (not comp.text) or (not comp.text.strip())
        err = comp.error or ("empty_completion" if empty else None)

        per_entry = {}
        if not empty:
            for s in injected:
                per_entry[s.entry.entry_id] = self.detector.score(
                    s.entry, comp.text
                ).as_dict()
        outcome = float("nan") if empty else task.score(comp.text)

        # ---- write 2: close -------------------------------------------------
        self.ledger.close_rows(
            turn_id, per_entry, outcome_score=outcome,
            turn_tokens=comp.total_tokens, key_id=comp.key_id, error=err,
        )
        self.ledger.close_turn(
            turn_id, outcome_score=outcome, prompt_tokens=comp.prompt_tokens,
            completion_tokens=comp.completion_tokens,
            turn_tokens=comp.total_tokens, key_id=comp.key_id,
            n_tool_calls=0, response_text=comp.text, error=err,
        )

        return TurnResult(turn_id, task.task_id, len(candidates), len(injected),
                          outcome, comp.total_tokens, comp.text, comp.error)

    # -- the run -------------------------------------------------------------

    def run(self, n_turns: Optional[int] = None, verbose: bool = True,
            on_progress=None, progress_every: int = 10) -> list[TurnResult]:
        """on_progress(runner, turns_done, final) is called every `progress_every`
        new turns and once at the end; it replaces the default progress line."""
        cfg = self.cfg
        n = n_turns or cfg.max_turns
        order = list(self.tasks)
        order_rng = random.Random(cfg.seed + 1)
        results: list[TurnResult] = []
        for i in range(n):
            if cfg.shuffle_epochs and i % len(order) == 0:
                order_rng.shuffle(order)
            if i < self.resume_from:
                # Replay the masking draws without calling the model, so a
                # resumed run continues exactly where the original would have.
                t = order[i % len(order)]
                mask_candidates(self.policy.candidates(t.prompt, cfg.top_k),
                                self.rng, cfg.mask_p, cfg.protect_bundle)
                continue
            if self.spent_tokens >= cfg.max_total_tokens:
                print(f"[budget] stopping at turn {i}: "
                      f"{self.spent_tokens} tokens spent (cap {cfg.max_total_tokens})")
                break
            task = order[i % len(order)]
            try:
                r = self.run_turn(task)
            except BudgetExhausted as exc:
                print(f"[budget] {exc}")
                self.stopped_early = True
                break
            results.append(r)
            self.policy.observe(self.ledger, cfg.run_id, i + 1)
            if on_progress is not None:
                if len(results) % max(1, progress_every) == 0:
                    on_progress(self, i + 1, False)
                continue
            if verbose and (i + 1) % 10 == 0:
                scored = [x.outcome_score for x in results
                          if x.outcome_score == x.outcome_score]
                n_empty = len(results) - len(scored)
                hit = sum(scored) / len(scored) if scored else float("nan")
                print(f"  turn {i+1:>3}  mean outcome {hit:.2f}  "
                      f"empty {n_empty}  tokens {self.spent_tokens:,}")
        self.ledger.finish_run(cfg.run_id)
        if on_progress is not None:
            on_progress(self, self.resume_from + len(results), True)
        elif verbose:
            stranded = self.ledger.open_row_count(cfg.run_id)
            n_empty = sum(1 for x in results
                          if x.outcome_score != x.outcome_score)
            print(f"done: {len(results)} turns, {self.spent_tokens:,} tokens, "
                  f"{stranded} unsettled rows, {n_empty} empty completions")
        return results

    def close(self) -> None:
        """Release the SQLite handle.

        Call this before building a second runner on the same db file in the
        same process -- two live connections on one WAL file can crash the
        interpreter at teardown.
        """
        self.ledger.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
