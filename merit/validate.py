"""Phase 0 checks: is the instrument working?

Three questions, in order of severity.

  1. Ledger integrity. Does every opened row get closed? Does every turn carry
     the same outcome across its rows? Are masked rows present at the expected
     rate?
  2. Detector sanity. Does the behavioural signal fire on 'useful' entries and
     stay quiet on 'distractor' ones?
  3. Signal check. Does a naive difference in means separate 'useful' from
     'distractor'? This is a preview of Layer 2, not Layer 2 itself -- no
     controls, no confidence intervals. If it is flat here, stop.
"""
from __future__ import annotations

import math

import pandas as pd

from .store import MemoryStore


def integrity(df: pd.DataFrame, expected_p: float) -> pd.DataFrame:
    checks = []

    unsettled = int((df["settled"] == 0).sum())
    checks.append(("all rows settled", unsettled == 0, f"{unsettled} open"))

    per_turn = df.groupby("turn_id")["outcome_score"].nunique()
    bad = int((per_turn > 1).sum())
    checks.append(("one outcome per turn", bad == 0, f"{bad} turns disagree"))

    masked_rate = float(df["masked"].mean())
    lo, hi = expected_p - 0.08, expected_p + 0.08
    checks.append(("mask rate near p", lo <= masked_rate <= hi,
                   f"{masked_rate:.3f} vs p={expected_p}"))

    n_masked = int(df["masked"].sum())
    checks.append(("masked rows exist", n_masked > 0, f"{n_masked} rows"))

    leak = int(((df["masked"] == 1) & (df["used_behavioural"] == 1)).sum())
    checks.append(("no use on masked rows", leak == 0, f"{leak} leaked"))

    if "error" in df.columns:
        empties = int(df["error"].fillna("").eq("empty_completion").sum())
        rate = empties / max(len(df), 1)
        checks.append(("few empty completions", rate < 0.05,
                       f"{empties} rows ({rate:.1%})"))

    return pd.DataFrame(
        [{"check": c, "pass": bool(ok), "detail": d} for c, ok, d in checks]
    )


def detector_report(df: pd.DataFrame, store: MemoryStore) -> pd.DataFrame:
    planted = {eid: e.planted for eid, e in store.entries.items()}
    d = df[df["masked"] == 0].copy()
    d["planted"] = d["entry_id"].map(planted)
    return (
        d.groupby("planted")
        .agg(injections=("inject_id", "count"),
             behavioural_rate=("used_behavioural", "mean"),
             lexical_mean=("used_lexical", "mean"),
             semantic_mean=("used_semantic", "mean"))
        .reset_index()
    )


def naive_effect(df: pd.DataFrame, store: MemoryStore) -> pd.DataFrame:
    """Difference in mean outcome, present vs withheld, stratified by task family.

    Stratification matters: retrieval mixes families into one candidate list, so
    an entry gets masked on turns where it was never needed. Pooling those in
    washes real effects to zero. Layer 2 does this properly with a regression;
    this is the cheap version for the Phase 0 check.
    """
    planted = {eid: e.planted for eid, e in store.entries.items()}
    d = df.dropna(subset=["outcome_score"])
    rows = []
    for eid, g in d.groupby("entry_id"):
        parts, wts, n_p, n_a = [], [], 0, 0
        for _fam, fg in g.groupby("task_family"):
            present = fg[fg["masked"] == 0]["outcome_score"]
            absent = fg[fg["masked"] == 1]["outcome_score"]
            if len(present) == 0 or len(absent) == 0:
                continue
            parts.append(present.mean() - absent.mean())
            wts.append(len(present) + len(absent))
            n_p += len(present)
            n_a += len(absent)
        if parts:
            eff = sum(p * w for p, w in zip(parts, wts)) / sum(wts)
            se = (max(parts) - min(parts)) / 2 if len(parts) > 1 else 0.0
        else:
            eff, se = float("nan"), float("nan")
        rows.append({
            "entry_id": eid, "planted": planted.get(eid, "?"),
            "n_present": n_p, "n_absent": n_a,
            "n_strata": len(parts),
            "naive_effect": round(eff, 3) if eff == eff else None,
            "spread": round(se, 3) if se == se else None,
        })
    return (pd.DataFrame(rows)
            .sort_values("naive_effect", ascending=False, na_position="last")
            .reset_index(drop=True))

def verdict(effects: pd.DataFrame) -> str:
    useful = effects[effects["planted"] == "useful"]["naive_effect"].dropna()
    distractor = effects[effects["planted"] == "distractor"]["naive_effect"].dropna()
    if useful.empty or distractor.empty:
        return ("INCONCLUSIVE: a planted class never got both a present and an "
                "absent row. Run more turns.")
    if useful.min() > distractor.max():
        return ("PASS: every useful entry separates from every distractor. "
                "The instrument can see the effect. Proceed to Layer 2.")
    if useful.mean() - distractor.mean() > 0.15:
        return ("WEAK PASS: useful entries lead on average but the classes "
                "overlap. More turns, or a lower-competence mock, before Layer 2.")
    return ("FAIL: useful and distractor entries look the same. Fix the "
            "instrument -- check the detector and the outcome scorer -- before "
            "building anything on top.")
