"""Layer 3 -- retrieval as a decision, not a similarity sort.

Input:  the utility table from Layer 2, and an incoming turn.
Output: the candidate set handed to Layer 1 for that turn.

Two policies behind one interface, so the runner and the ledger don't change:

  RelevancePolicy  -- top-k by BM25. The baseline every system uses.

  UtilityPolicy    -- pull a wider relevance pool, then rank it by
                      sampled utility x normalised relevance / tokens,
                      and inject only entries whose sampled utility is
                      positive. Posteriors are refit online from the
                      ledger as the run goes.

Why multiply by relevance at all. Layer 2's utility is averaged over every turn
an entry was a candidate on. A build-flag entry has high utility, but only on
build tasks. Ranked on utility alone it would be pushed into test tasks where it
does nothing. Relevance keeps the decision local to the current query without
needing task labels. A fully contextual estimate -- utility per query region --
is the obvious extension, and a known limitation of this version.

Why Thompson sampling rather than the posterior mean. Ranking on the mean is
greedy: an entry that looks mediocre after a few unlucky turns is never offered
again, so it never gets the data that would correct the estimate. Sampling from
the posterior offers uncertain entries occasionally, in proportion to how
plausible it is that they're good. Exploration falls out of the uncertainty.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .store import MemoryStore, Scored


@dataclass
class Posterior:
    mean: float
    sd: float
    n: int = 0          # turns the estimate rests on; 0 = prior only


class RelevancePolicy:
    name = "relevance"

    def __init__(self, store: MemoryStore):
        self.store = store

    def candidates(self, prompt: str, k: int) -> list[Scored]:
        return self.store.retrieve(prompt, top_k=k)

    def observe(self, ledger, run_id: str, turn_index: int) -> None:
        return None

    def describe(self) -> dict:
        return {"policy": self.name}


class UtilityPolicy:
    """Thompson-sampled, budget-aware retrieval over a relevance pool."""

    name = "utility"

    def __init__(self, store: MemoryStore, pool: int = 6,
                 prior_mean: float = 0.10, prior_sd: float = 0.35,
                 inject_threshold: float = 0.0,
                 refit_every: int = 25, min_trials: int = 5,
                 seed: int = 0, sample: bool = True,
                 mode: str = "thompson",
                 dead_weight: float = 0.10, dead_weight_min_n: int = 60,
                 floor: float = 0.02):
        """mode="thompson": sample from posteriors, explore via the draw.
        mode="greedy":   use posterior means; exploration comes only from Layer 1
                         masking. Unfitted entries get a neutral multiplier, so
                         relevance decides until evidence exists, and an entry
                         with enough evidence of ~zero effect is dropped.
        """
        self.mode = mode
        self.dead_weight = dead_weight
        self.dead_weight_min_n = dead_weight_min_n
        self.floor = floor
        self.store = store
        self.pool = pool
        self.prior_mean = prior_mean
        self.prior_sd = prior_sd
        self.inject_threshold = inject_threshold
        self.refit_every = refit_every
        self.min_trials = min_trials
        self.sample = sample
        self.rng = random.Random(seed)
        self.post: dict[str, Posterior] = {}
        self.history: list[dict] = []      # one row per refit, for plotting
        self._last_refit = 0

    # -- posteriors ------------------------------------------------------------

    def posterior(self, entry_id: str) -> Posterior:
        return self.post.get(entry_id, Posterior(self.prior_mean, self.prior_sd, 0))

    def load(self, utility_df) -> None:
        """Seed from a Layer 2 utility table (warm start)."""
        for _, r in utility_df.iterrows():
            mean = max(-1.0, min(1.0, float(r["utility"])))   # LPM can overshoot
            sd = max(float(r["se"]), 0.02)
            n = int(r.get("n_injected", 0)) + int(r.get("n_withheld", 0))
            self.post[r["entry_id"]] = Posterior(mean, sd, n)

    def refit(self, ledger_df) -> bool:
        from .credit import fit_utility
        try:
            u = fit_utility(ledger_df, min_trials=self.min_trials, verbose=False)
        except Exception:
            return False            # too little data yet; keep the prior
        self.load(u)
        return True

    # -- the decision ------------------------------------------------------------

    def _draw(self, p: Posterior) -> float:
        if not self.sample:
            return p.mean
        return self.rng.gauss(p.mean, p.sd)

    def candidates(self, prompt: str, k: int) -> list[Scored]:
        if self.mode == "greedy":
            return self._greedy(prompt, k)
        pool = self.store.retrieve(prompt, top_k=self.pool)
        if not pool:
            return []
        top = max(s.score for s in pool) or 1.0

        scored = []
        for s in pool:
            p = self.posterior(s.entry.entry_id)
            draw = self._draw(p)
            if draw <= self.inject_threshold:
                continue
            rel = s.score / top
            density = draw * rel / max(s.entry.tokens, 1)
            scored.append((density, draw, s))

        scored.sort(key=lambda x: -x[0])
        out = []
        for rank, (_dens, _draw, s) in enumerate(scored[:k]):
            out.append(Scored(entry=s.entry, score=s.score, rank=rank))
        return out

    def _greedy(self, prompt: str, k: int) -> list[Scored]:
        pool = self.store.retrieve(prompt, top_k=self.pool)
        if not pool:
            return []
        top = max(s.score for s in pool) or 1.0
        scored = []
        for s in pool:
            p = self.post.get(s.entry.entry_id)
            if p is None:
                mult = self.prior_mean                    # no evidence: neutral
            else:
                if p.n >= self.dead_weight_min_n and p.mean < self.dead_weight:
                    continue                              # proven dead weight
                mult = max(self.floor, min(1.0, p.mean))
            scored.append((mult * s.score / top, s))
        scored.sort(key=lambda x: -x[0])
        return [Scored(entry=s.entry, score=s.score, rank=r)
                for r, (_v, s) in enumerate(scored[:k])]

    # -- online learning ---------------------------------------------------------

    def observe(self, ledger, run_id: str, turn_index: int) -> None:
        if self.refit_every <= 0:
            return
        if turn_index - self._last_refit < self.refit_every:
            return
        self._last_refit = turn_index
        df = ledger.dataframe(run_id)
        ok = self.refit(df)
        snap = {"turn": turn_index, "refit_ok": ok}
        for eid, p in self.post.items():
            snap[eid] = round(p.mean, 3)
        self.history.append(snap)

    def describe(self) -> dict:
        return {"policy": self.name, "mode": self.mode, "pool": self.pool,
                "prior_mean": self.prior_mean, "prior_sd": self.prior_sd,
                "inject_threshold": self.inject_threshold,
                "refit_every": self.refit_every, "sample": self.sample}
