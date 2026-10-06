"""Layer 4 -- utility-gated lifecycle: promote, demote, forget.

Input:  Layer 2's utility table and the retrieval statistics from the ledger.
Output: decisions about the STORE, not about a query:
          - which entries sit in the always-injected core
          - which get demoted out of it
          - which get forgotten when the store is over budget

Why global utility is the right signal here, when Layer 3 showed it is the wrong
one for per-query retrieval: "should this entry be in the core / in the store at
all?" is a question about the entry. It has no query. Layer 3 failed because it
used a query-free number to make a per-query decision. Layer 4 uses it for the
decision it actually answers.

The score
---------
Layer 2's utility is conditional: the effect of injecting an entry *on turns
where it was a retrieval candidate*. A core entry is injected on EVERY turn, so
what matters is the expected gain per turn:

    expected_gain = utility x eligibility_rate
    eligibility_rate = turns the entry was a candidate / total turns

and, because core seats cost tokens on every turn, per token. The incumbent
rule (OpenClaw-style recall frequency) ranks on eligibility_rate alone. The two
rules see the same retrievability data; MERIT multiplies in measured effect.

The gate
--------
Nothing is promoted unless the lower end of its utility interval is above zero.
That is the line between "looks useful" and "measured to be useful".
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import pandas as pd

from .credit import fit_utility, recall_frequency
from .store import MemoryStore, Scored


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #

def core_scores(ledger: pd.DataFrame, store: MemoryStore,
                utility: Optional[pd.DataFrame] = None,
                min_trials: int = 5) -> pd.DataFrame:
    """One row per entry in the store, with everything both rules need."""
    if utility is None:
        utility = fit_utility(ledger, min_trials=min_trials, verbose=False)
    n_turns = ledger["turn_id"].nunique()
    rec = recall_frequency(ledger).set_index("entry_id")
    u = utility.set_index("entry_id")

    rows = []
    for eid, e in store.entries.items():
        retrievals = int(rec["retrievals"].get(eid, 0))
        elig = retrievals / n_turns if n_turns else 0.0
        if eid in u.index:
            util = float(u.loc[eid, "utility"])
            lo, hi = float(u.loc[eid, "ci_lo"]), float(u.loc[eid, "ci_hi"])
            n_withheld = int(u.loc[eid, "n_withheld"])
            measured = True
        else:
            util, lo, hi, measured = 0.0, float("nan"), float("nan"), False
            n_withheld = 0
        util_c = max(-1.0, min(1.0, util))          # LPM can overshoot 1.0
        gain = util_c * elig
        rows.append({
            "entry_id": eid,
            "tokens": e.tokens,
            "retrievals": retrievals,
            "eligibility_rate": round(elig, 3),
            "utility": round(util, 3),
            "ci_lo": round(lo, 3) if measured else None,
            "ci_hi": round(hi, 3) if measured else None,
            "measured": measured,
            "n_withheld": n_withheld,
            "promotable": bool(measured and lo > 0),
            "expected_gain": round(gain, 4),
            "gain_per_token": round(gain / max(e.tokens, 1), 5),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Decisions
# --------------------------------------------------------------------------- #

def promote(scores: pd.DataFrame, rule: str, size: int) -> list[str]:
    """Pick the always-injected core.

    rule="recall":  top `size` by retrievals. The incumbent.
    rule="utility": only gate-passing entries, ranked by expected gain per token.
                    May return fewer than `size` -- an empty seat beats a seat
                    filled with something measured to do nothing.
    """
    if rule == "recall":
        s = scores.sort_values(["retrievals", "entry_id"], ascending=[False, True])
        return s["entry_id"].head(size).tolist()
    if rule == "utility":
        s = scores[scores["promotable"]].sort_values(
            ["gain_per_token", "entry_id"], ascending=[False, True])
        return s["entry_id"].head(size).tolist()
    raise ValueError(f"unknown rule {rule!r}")


def demote(scores: pd.DataFrame, core: Iterable[str],
           min_withheld: int = 15) -> list[str]:
    """Entries to remove from the core.

    A seat is a lease, not a deed -- but an entry is only judged once it has
    enough WITHHELD trials to be judged. Withheld rows carry the counterfactual;
    total retrievals do not. Without this guard, every entry looks unproven
    early on (wide interval) and useful entries get evicted along with the
    dead weight.
    """
    s = scores.set_index("entry_id")
    out = []
    for eid in core:
        if eid not in s.index:
            continue
        r = s.loc[eid]
        if not r["measured"]:
            continue
        if r["n_withheld"] >= min_withheld and not r["promotable"]:
            out.append(eid)
    return out


def _session(entry_tags) -> int:
    for t in entry_tags or []:
        if str(t).startswith("session:"):
            return int(str(t).split(":")[1])
    return 0


def forget(scores: pd.DataFrame, rule: str, keep: int,
           store: Optional[MemoryStore] = None, seed: int = 0) -> list[str]:
    """Which entries survive when the store must shrink to `keep` entries.

    rule="recall":  keep the most-retrieved. What an LRU / frequency cache does.
    rule="utility": keep the highest expected gain. Unmeasured entries score 0
                    -- nothing has been shown about them -- and are evicted
                    before anything with a measured positive effect.
    """
    if rule == "recall":
        s = scores.sort_values(["retrievals", "entry_id"], ascending=[False, True])
    elif rule == "utility":
        s = scores.sort_values(["expected_gain", "retrievals", "entry_id"],
                               ascending=[False, False, True])
    elif rule == "random":
        s = scores.sample(frac=1.0, random_state=seed)
    elif rule == "recency":
        if store is None:
            raise ValueError("recency needs the store (for session tags)")
        s = scores.assign(_sess=scores["entry_id"].map(
                lambda e: _session(store.get(e).tags)))
        s = s.sort_values(["_sess", "entry_id"], ascending=[False, True])
    else:
        raise ValueError(f"unknown rule {rule!r}")
    return s["entry_id"].head(keep).tolist()


def subset_store(store: MemoryStore, keep: Sequence[str]) -> MemoryStore:
    return MemoryStore([store.get(e) for e in keep])


# --------------------------------------------------------------------------- #
# Policies that apply the decisions at retrieval time
# --------------------------------------------------------------------------- #

class CorePolicy:
    """Always inject the core; add `k_episodic` relevance hits from the rest.

    Core entries pass through Layer 1 masking like anything else, so they keep
    generating withheld trials -- which is what lets a core entry be demoted.
    """

    name = "core"

    def __init__(self, store: MemoryStore, core: Sequence[str], k_episodic: int = 1):
        self.store = store
        self.core = list(core)
        self.k_episodic = k_episodic

    def candidates(self, prompt: str, k: int) -> list[Scored]:
        out = []
        for rank, eid in enumerate(self.core):
            out.append(Scored(entry=self.store.get(eid), score=1.0, rank=rank))
        core = set(self.core)
        hits = [s for s in self.store.retrieve(prompt, top_k=len(self.store))
                if s.entry.entry_id not in core][: self.k_episodic]
        for s in hits:
            out.append(Scored(entry=s.entry, score=s.score, rank=len(out)))
        return out

    def observe(self, ledger, run_id: str, turn_index: int) -> None:
        return None

    def describe(self) -> dict:
        return {"policy": self.name, "core": self.core, "k_episodic": self.k_episodic}


class LifecyclePolicy(CorePolicy):
    """CorePolicy whose core is revised online from the run's own ledger.

    Every `refit_every` turns: refit utility, demote core entries that no longer
    clear the gate, promote gate-passing entries into free seats.
    """

    name = "lifecycle"

    def __init__(self, store: MemoryStore, core: Sequence[str], size: int,
                 k_episodic: int = 1, refit_every: int = 50,
                 min_trials: int = 5, demote_min_withheld: int = 15):
        super().__init__(store, core, k_episodic)
        self.size = size
        self.refit_every = refit_every
        self.min_trials = min_trials
        self.demote_min_withheld = demote_min_withheld
        self.history: list[dict] = [{"turn": 0, "core": list(core),
                                     "demoted": [], "promoted": []}]
        self._last = 0

    def observe(self, ledger, run_id: str, turn_index: int) -> None:
        if turn_index - self._last < self.refit_every:
            return
        self._last = turn_index
        df = ledger.dataframe(run_id)
        try:
            sc = core_scores(df, self.store, min_trials=self.min_trials)
        except Exception:
            return
        gone = demote(sc, self.core, self.demote_min_withheld)
        core = [e for e in self.core if e not in gone]
        added = []
        for eid in promote(sc, "utility", self.size):
            if len(core) >= self.size:
                break
            if eid not in core:
                core.append(eid)
                added.append(eid)
        self.core = core
        self.history.append({"turn": turn_index, "core": list(core),
                             "demoted": gone, "promoted": added})
