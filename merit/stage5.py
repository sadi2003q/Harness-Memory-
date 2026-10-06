"""Stage 5 -- the real-data test of MERIT's core claim.

Claim under test
----------------
When a memory store must shrink, keeping entries by MEASURED UTILITY preserves
more downstream answer quality on UNSEEN questions than keeping them by RECALL
FREQUENCY -- the signal incumbent systems use.

Protocol (per conversation)
---------------------------
  1. Split questions into a learning half and a held-out test half.
  2. LEARN  -- run the learning questions for several shuffled passes with
               masking on. This produces the ledger.
  3. DECIDE -- estimate utility (empirical Bayes) from the ledger only, then
               shrink the store to a fixed fraction under each rule.
  4. TEST   -- answer every held-out question once against each pruned store,
               masking off, same retriever, same k. Compare arms PAIRED by
               question.

Nothing about the test questions is visible at steps 2-3, except in the
"oracle" arm, which is labelled as a ceiling and never as a result.
"""
from __future__ import annotations

import contextlib
import io
import random
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .clients import build_client
from .config import Config
from .credit import fit_utility_eb
from .lifecycle import core_scores, forget, subset_store
from .locomo import QA_SYSTEM_PROMPT, Conversation, QAMock, QATask
from .runner import Layer1Runner

RULES = ("recall", "utility", "random", "recency")


class Paused(RuntimeError):
    """The daily token/request limit was reached mid-phase.

    Every finished turn is saved. Re-run the same cell after the limit resets
    and the phase continues from the exact turn where it stopped.
    """


def _client(cfg: Config, conv: Conversation, mock_competence: float):
    if cfg.backend == "mock":
        return QAMock(conv, competence=mock_competence, seed=cfg.seed)
    return build_client(cfg)


def _quiet(fn, verbose: bool):
    if verbose:
        return fn()
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #

def learn_phase(conv: Conversation, learn: list[QATask], *, backend: str,
                epochs: int = 4, mask_p: float = 0.30, top_k: int = 5,
                seed: int = 0, run_tag: str = "", db_path: str = "data/stage5.db",
                model: str = "openai/gpt-oss-20b", reasoning_effort: Optional[str] = "low",
                max_tokens: int = 512, max_total_tokens: int = 400_000,
                mock_competence: float = 0.85, verbose: bool = True,
                monitor_every: int = 0, blind: bool = True) -> pd.DataFrame:
    cfg = Config(run_id=f"s5_learn_c{conv.index}{run_tag}", backend=backend,
                 model=model, mask_p=mask_p, top_k=top_k,
                 max_turns=epochs * len(learn), shuffle_epochs=True, seed=seed,
                 max_tokens=max_tokens, reasoning_effort=reasoning_effort,
                 max_total_tokens=max_total_tokens, db_path=db_path,
                 max_bundle_tokens=10_000)
    client = _client(cfg, conv, mock_competence)

    def go():
        with Layer1Runner(cfg, store=conv.store, tasks=learn, client=client,
                          system_prompt=QA_SYSTEM_PROMPT, resume=True) as r:
            mon = (LiveMonitor(f"conv {conv.index} | learning", "learn", cfg.max_turns,
                               blind=blind) if monitor_every else None)
            r.run(verbose=verbose, on_progress=mon, progress_every=max(1, monitor_every))
            done, _ = r.ledger.settled_turns(cfg.run_id)
            if done < cfg.max_turns:
                raise Paused(f"{cfg.run_id}: {done}/{cfg.max_turns} turns done -- "
                             f"daily limit reached. Re-run this cell after the reset.")
            return r.ledger.dataframe(cfg.run_id)
    return go() if monitor_every else _quiet(go, verbose)


def decide(ledger: pd.DataFrame, conv: Conversation, keep_frac: float,
           test: list[QATask], seed: int = 0,
           rules: Sequence[str] = RULES) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Kept-entry lists for every arm. Uses the learning ledger only."""
    util = _quiet(lambda: fit_utility_eb(ledger, verbose=False), False)
    scores = core_scores(ledger, conv.store, utility=util)
    keep = max(1, int(round(keep_frac * len(conv.store))))
    arms = {r: forget(scores, r, keep, store=conv.store, seed=seed) for r in rules}
    arms["full"] = list(conv.store.entries)
    # Ceiling, not a result: keeps exactly what the TEST questions need, padded
    # with recall-ranked entries to the same size.
    need = list(conv.evidence_entries(test))
    pad = [e for e in forget(scores, "recall", len(conv.store)) if e not in set(need)]
    arms["oracle"] = (need + pad)[:max(keep, len(need))]
    return arms, scores


def test_arm(name: str, conv: Conversation, kept: list[str], test: list[QATask], *,
             backend: str, top_k: int = 5, seed: int = 0, run_tag: str = "",
             db_path: str = "data/stage5.db", model: str = "openai/gpt-oss-20b",
             reasoning_effort: Optional[str] = "low", max_tokens: int = 512,
             max_total_tokens: int = 200_000, mock_competence: float = 0.85,
             verbose: bool = False, monitor_every: int = 0,
             blind: bool = True) -> pd.DataFrame:
    cfg = Config(run_id=f"s5_test_c{conv.index}_{name}{run_tag}", backend=backend,
                 model=model, mask_p=0.0, top_k=top_k, max_turns=len(test),
                 seed=seed, max_tokens=max_tokens, reasoning_effort=reasoning_effort,
                 max_total_tokens=max_total_tokens, db_path=db_path,
                 max_bundle_tokens=10_000)
    client = _client(cfg, conv, mock_competence)
    store = subset_store(conv.store, kept)

    def go():
        with Layer1Runner(cfg, store=store, tasks=test, client=client,
                          system_prompt=QA_SYSTEM_PROMPT, resume=True) as r:
            mon = (LiveMonitor(f"conv {conv.index} | test arm '{name}'", "test",
                               cfg.max_turns, blind=blind, arm=name)
                   if monitor_every else None)
            r.run(verbose=verbose, on_progress=mon, progress_every=max(1, monitor_every))
            done, _ = r.ledger.settled_turns(cfg.run_id)
            if done < cfg.max_turns:
                raise Paused(f"{cfg.run_id}: {done}/{cfg.max_turns} turns done -- "
                             f"daily limit reached. Re-run this cell after the reset.")
            return r.ledger.turns_dataframe(cfg.run_id)
    t = go() if monitor_every else _quiet(go, verbose)
    t["arm"] = name
    t["conv"] = conv.index
    t["store_size"] = len(kept)
    return t


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

def arm_table(turns: pd.DataFrame) -> pd.DataFrame:
    g = turns.dropna(subset=["outcome_score"]).groupby("arm")
    out = pd.DataFrame({
        "questions": g.size(),
        "mean_f1": g["outcome_score"].mean().round(4),
        "exact_ish": g["outcome_score"].apply(lambda s: (s >= 0.5).mean()).round(3),
        "store_size": g["store_size"].first(),
        "tokens": g["turn_tokens"].sum(),
    })
    order = ["full", "oracle", "utility", "recall", "recency", "random"]
    return out.reindex([a for a in order if a in out.index] +
                       [a for a in out.index if a not in order])


def paired_bootstrap(turns: pd.DataFrame, a: str, b: str, n_boot: int = 10_000,
                     seed: int = 0) -> dict:
    """Mean F1(a) - F1(b) over the SAME questions, with a stratified bootstrap CI.

    Paired, because every arm answers the same questions: question difficulty
    cancels out, which is where most of the variance lives. Stratified by
    conversation, so pooling several conversations does not let one dominate.
    """
    t = turns.dropna(subset=["outcome_score"])
    wide = t.pivot_table(index=["conv", "task_id"], columns="arm",
                         values="outcome_score", aggfunc="mean")
    wide = wide[[a, b]].dropna()
    diff = (wide[a] - wide[b]).reset_index()
    diff.columns = ["conv", "task_id", "d"]
    rng = np.random.default_rng(seed)
    groups = [g["d"].to_numpy() for _, g in diff.groupby("conv")]
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = np.mean(np.concatenate(
            [g[rng.integers(0, len(g), len(g))] for g in groups]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"comparison": f"{a} - {b}", "questions": len(diff),
            "mean_diff": round(float(diff["d"].mean()), 4),
            "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
            "wins": int((diff["d"] > 0).sum()), "losses": int((diff["d"] < 0).sum()),
            "ties": int((diff["d"] == 0).sum()),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def ledger_quality(ledger: pd.DataFrame) -> dict:
    """How much did the learning phase actually measure?"""
    g = ledger.groupby("entry_id")["masked"].agg(["sum", "count"])
    return {
        "turns": ledger["turn_id"].nunique(),
        "entries_seen": len(g),
        "entries_with_withheld_trial": int((g["sum"] > 0).sum()),
        "entries_with_5plus_withheld": int((g["sum"] >= 5).sum()),
        "median_candidate_appearances": float(g["count"].median()),
        "empty_rate": round(float(ledger["outcome_score"].isna().mean()), 3),
    }


def estimate_tokens(n_turns: int, prompt_tokens_per_turn: float,
                    completion_tokens_per_turn: float = 180) -> int:
    """Rough budget. Calibrate completion tokens from the first few live turns."""
    return int(n_turns * (prompt_tokens_per_turn + completion_tokens_per_turn))


# --------------------------------------------------------------------------- #
# Resumable per-conversation driver
# --------------------------------------------------------------------------- #

def _done(db_path: str, run_id: str, expected_turns: int):
    """Return a completed run's turns, or None. Lets a multi-day run resume."""
    import sqlite3
    from pathlib import Path
    p = Path(Config(db_path=db_path).resolved_db_path())
    if not p.exists():
        return None
    con = sqlite3.connect(p)
    try:
        t = pd.read_sql("SELECT * FROM turns WHERE run_id=?", con, params=(run_id,))
        led = pd.read_sql("SELECT * FROM inject_ledger WHERE run_id=?", con, params=(run_id,))
    except Exception:
        return None
    finally:
        con.close()
    if len(t) >= expected_turns and (led["settled"] == 1).all():
        return t, led
    return None


def run_conversation(data, conv_index: int, *, track: str, backend: str,
                     settings: dict, arms: Sequence[str] = ("utility", "recall", "random"),
                     db_path: str = "data/stage5.db", resume: bool = True,
                     verbose: bool = True, monitor_every: int = 10,
                     blind: bool = True) -> dict:
    """One conversation, one track, end to end. Safe to re-run: finished phases are reused."""
    from .locomo import build_conversation, split_tasks, transfer_ceiling
    inc = track == "autocapture"
    conv = build_conversation(data, conv_index, include_turns=inc)
    learn, test = split_tasks(conv.tasks, seed=settings["split_seed"], frac=0.5)
    tag = f"_{track}"
    common = dict(backend=backend, seed=settings["seed"], db_path=db_path,
                  model=settings["model"], reasoning_effort=settings["reasoning_effort"],
                  max_tokens=settings["max_tokens"], top_k=settings["top_k"])

    learn_id = f"s5_learn_c{conv.index}{tag}"
    n_learn = settings["epochs"] * len(learn)
    hit = _done(db_path, learn_id, n_learn) if resume else None
    if hit is not None:
        ledger = hit[1]
        if verbose: print(f"[c{conv_index} {track}] learning phase reused ({n_learn} turns)")
    else:
        if verbose: print(f"[c{conv_index} {track}] learning: {n_learn} turns")
        ledger = learn_phase(conv, learn, epochs=settings["epochs"], mask_p=settings["mask_p"],
                             run_tag=tag, verbose=verbose, monitor_every=monitor_every,
                             blind=blind, **common)

    kept, _scores = decide(ledger, conv, settings["keep_frac"], test, seed=settings["seed"])
    out = []
    for a in arms:
        rid = f"s5_test_c{conv.index}_{a}{tag}"
        hit = _done(db_path, rid, len(test)) if resume else None
        if hit is not None:
            t = hit[0]; t["arm"] = a; t["conv"] = conv.index; t["store_size"] = len(kept[a])
            if verbose: print(f"[c{conv_index} {track}] arm {a} reused")
        else:
            if verbose: print(f"[c{conv_index} {track}] arm {a}: {len(test)} turns")
            t = test_arm(a, conv, kept[a], test, run_tag=tag, verbose=False,
                         monitor_every=monitor_every, blind=blind, **common)
        t["track"] = track
        out.append(t)
    turns = pd.concat(out, ignore_index=True)
    need = conv.evidence_entries(test)
    return {"conv": conv_index, "track": track, "turns": turns,
            "ledger_quality": ledger_quality(ledger),
            "transfer": transfer_ceiling(conv, learn, test),
            "evidence_kept": {a: f"{len(set(kept[a]) & need)}/{len(need)}" for a in arms},
            "store_entries": len(conv.store)}


# --------------------------------------------------------------------------- #
# Progress monitoring -- blind by default
# --------------------------------------------------------------------------- #

def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None and "IPKernelApp" in get_ipython().config
    except Exception:
        return False


class LiveMonitor:
    """A dashboard that refreshes every few turns while a phase runs.

    Always shown (cannot bias the result):
      progress, ETA, tokens, empty answers, key status,
      learning-phase F1 (one arm only -- no comparison),
      estimator readiness (is it finding differences between entries?).

    Shown only when blind=False:
      per-arm F1 on the held-out questions -- the H1 comparison itself.
      Looking at it before every primary conversation is finished invites
      optional stopping, which is what the pre-registration rules out.
    """

    def __init__(self, label: str, phase: str, total: int, *, blind: bool = True,
                 window: int = 20, fit_every: int = 60, arm: Optional[str] = None):
        import time
        self.label, self.phase, self.total = label, phase, total
        self.blind, self.window, self.fit_every = blind, window, fit_every
        self.arm = arm
        self.t0 = time.time()
        self.start_done = None
        self._last_fit = 0
        self._ready = "not yet estimated"
        self.notebook = _in_notebook()

    # -- metrics -------------------------------------------------------------
    def _metrics(self, runner) -> dict:
        import time
        rid = runner.cfg.run_id
        con = runner.ledger.con
        t = pd.read_sql("SELECT outcome_score, turn_tokens, error FROM turns "
                        "WHERE run_id=? AND settled_at IS NOT NULL ORDER BY rowid",
                        con, params=(rid,))
        done = len(t)
        if self.start_done is None:
            self.start_done = getattr(runner, "resume_from", 0)   # done before this session
        elapsed = max(time.time() - self.t0, 1e-6)
        rate = max(done - self.start_done, 0) / elapsed * 60          # turns / min
        remaining = self.total - done
        eta_min = remaining / rate if rate > 0 else float("nan")
        f1 = t["outcome_score"].dropna()
        last = f1.iloc[-self.window:]
        prev = f1.iloc[-2 * self.window:-self.window]
        m = {
            "done": done, "remaining": remaining, "rate": rate, "eta_min": eta_min,
            "tokens": int(t["turn_tokens"].fillna(0).sum()),
            "tok_per_turn": float(t["turn_tokens"].mean()) if done else float("nan"),
            "empties": int(t["error"].fillna("").eq("empty_completion").sum()),
            "f1_all": float(f1.mean()) if len(f1) else float("nan"),
            "f1_last": float(last.mean()) if len(last) else float("nan"),
            "f1_prev": float(prev.mean()) if len(prev) else float("nan"),
            "f1_noise": float(2 * (f1.iloc[-2 * self.window:].std(ddof=1) or 0)
                              * (2 / self.window) ** 0.5) if len(prev) >= 2 else float("nan"),
        }
        pool = getattr(getattr(runner, "client", None), "pool", None)
        if pool is not None:
            m["keys"] = f"{sum(not s.exhausted for s in pool.states)}/{len(pool.states)} keys live"
        if self.phase == "learn" and done - self._last_fit >= self.fit_every:
            self._last_fit = done
            led = pd.read_sql("SELECT * FROM inject_ledger WHERE run_id=? AND settled=1",
                              con, params=(rid,))
            g = led.groupby("entry_id")["masked"].sum()
            try:
                eb = fit_utility_eb(led, verbose=False)
                spread = eb.attrs["tau2"] ** 0.5
                self._ready = (f"{int((g >= 1).sum())} entries with a withheld trial, "
                               f"{int((g >= 5).sum())} with 5+; "
                               f"between-entry spread {spread:.3f}; "
                               f"{int(eb['promotable'].sum())} entries clearly helpful so far")
            except Exception:
                self._ready = f"{int((g >= 1).sum())} entries with a withheld trial (too early to estimate)"
        m["ready"] = self._ready
        return m

    # -- render ----------------------------------------------------------------
    def __call__(self, runner, turns_done: int, final: bool) -> None:
        m = self._metrics(runner)
        pct = 100 * m["done"] / max(self.total, 1)
        bar = "#" * int(pct // 5) + "." * (20 - int(pct // 5))
        trend = ""
        if m["f1_prev"] == m["f1_prev"] and m["f1_noise"] == m["f1_noise"]:
            d = m["f1_last"] - m["f1_prev"]
            if abs(d) <= m["f1_noise"]:
                trend = f"  (vs previous {self.window}: {m['f1_prev']:.3f}, within noise)"
            else:
                trend = (f"  (vs previous {self.window}: {m['f1_prev']:.3f}, "
                         f"{'UP' if d > 0 else 'DOWN'} beyond noise ±{m['f1_noise']:.3f})")
        lines = [f"{self.label}  [{bar}] {m['done']}/{self.total} ({pct:.0f}%)"
                 + ("  DONE" if final and m["remaining"] <= 0 else ""),
                 f"  speed {m['rate']:.1f} turns/min   ETA {m['eta_min']:.0f} min   "
                 f"tokens {m['tokens']:,} ({m['tok_per_turn']:.0f}/turn)   "
                 f"empty answers {m['empties']}" + (f"   {m['keys']}" if "keys" in m else "")]
        if self.phase == "learn":
            lines.append(f"  learning F1: overall {m['f1_all']:.3f}, last {self.window} turns "
                         f"{m['f1_last']:.3f}{trend}   <- should stay roughly flat; "
                         f"near 0 means something is broken")
            lines.append(f"  estimator: {m['ready']}")
        else:
            if self.blind:
                lines.append(f"  arm '{self.arm}': F1 hidden (blind mode) -- revealed in "
                             f"section 6 once every primary conversation is finished")
            else:
                lines.append(f"  arm '{self.arm}': F1 so far {m['f1_all']:.3f}   "
                             f"(UNBLINDED -- record this in the Deviations section)")
        if m["done"] and m["f1_all"] == m["f1_all"] and m["f1_all"] < 0.02 and m["done"] >= 20:
            lines.append("  WARNING: F1 is near zero -- stop and check the answers")
        if m["done"] >= 20 and m["empties"] / m["done"] > 0.05:
            lines.append("  WARNING: more than 5% empty answers -- consider max_tokens")
        text = "\n".join(lines)
        if self.notebook:
            from IPython.display import clear_output
            clear_output(wait=True)
        print(text, flush=True)


def status(db_path: str, data, settings: dict, plan: Sequence[tuple],
           arms: Sequence[str] = ("utility", "recall", "random"),
           blind: bool = True) -> pd.DataFrame:
    """Where is the whole experiment? Safe to call any time, from any notebook,
    even while a run is going in another window (SQLite allows concurrent reads).
    """
    import sqlite3
    from pathlib import Path
    from .locomo import build_conversation, split_tasks
    p = Path(Config(db_path=db_path).resolved_db_path())
    con = sqlite3.connect(p) if p.exists() else None
    rows = []
    for track, c in plan:
        conv = build_conversation(data, c, include_turns=(track == "autocapture"))
        learn, test = split_tasks(conv.tasks, seed=settings["split_seed"])
        phases = [("learn", f"s5_learn_c{c}_{track}", settings["epochs"] * len(learn), None)]
        phases += [("test", f"s5_test_c{c}_{a}_{track}", len(test), a) for a in arms]
        for ph, rid, total, arm in phases:
            done = toks = empt = 0
            f1 = float("nan")
            if con is not None:
                r = con.execute("SELECT COUNT(*), COALESCE(SUM(turn_tokens),0), "
                                "SUM(CASE WHEN error='empty_completion' THEN 1 ELSE 0 END), "
                                "AVG(outcome_score) FROM turns WHERE run_id=? "
                                "AND settled_at IS NOT NULL", (rid,)).fetchone()
                done, toks, empt, f1 = r[0], r[1], r[2] or 0, r[3]
            show_f1 = ph == "learn" or not blind
            rows.append({"track": track, "conv": c, "phase": ph, "arm": arm or "",
                         "done": done, "total": total,
                         "pct": round(100 * done / total, 1) if total else 0.0,
                         "tokens": toks, "empty": empt,
                         "f1": (round(f1, 3) if (f1 is not None and show_f1) else
                                ("hidden" if ph == "test" and done else None))})
    if con is not None:
        con.close()
    df = pd.DataFrame(rows)
    left = int((df["total"] - df["done"]).clip(lower=0).sum())
    tpt = (df["tokens"].sum() / df["done"].sum()) if df["done"].sum() else float("nan")
    est = f"~{int(left * tpt):,} tokens" if tpt == tpt else "token estimate after the first turns"
    print(f"overall: {int(df['done'].sum())}/{int(df['total'].sum())} turns done, "
          f"{int(df['tokens'].sum()):,} tokens used, ~{left} turns left ({est})")
    return df
