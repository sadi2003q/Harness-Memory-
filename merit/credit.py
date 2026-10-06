"""Layer 2 -- credit assignment.

Input:  the inject_ledger table Layer 1 wrote.
Output: per-entry utility, with a confidence interval and a trial count.

The model
---------
One observation per TURN, not per row:

    y_t = a + sum_i beta_i * z_it + sum_i g_i * elig_it + controls_t + e_t

  y_t     outcome of turn t
  elig_it 1 if entry i was a retrieval candidate on turn t
  z_it    1 if entry i was a candidate AND survived masking

Why both columns. Randomisation only happened among candidates: if an entry was
never retrieved, no coin was flipped for it, and a plain z would conflate
"withheld" with "never offered". Conditioning on elig restores the experiment,
so beta_i is the average effect of injecting entry i *given that it was
retrievable* -- which is exactly the quantity Layer 3 needs to rank with.

Controls: task family, candidate-set size, bundle token count, harness state.
Task family is the one that matters most -- retrieval mixes families into a
single candidate list, so an entry accumulates "absent" observations on turns
where it was irrelevant. Pooling those dilutes real effects toward zero. That
is the confound the stratified Phase 0 check exposed; here it is handled
properly.

Estimator: linear probability model with HC3 robust standard errors. The
outcome is binary, but we want an average treatment effect on the probability
scale, and OLS gives that directly. Logit is available for a robustness check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_CONTROLS = ("task_family", "n_candidates", "bundle_tokens", "harness_state")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_ledger(source: str, run_id: Optional[str] = None) -> pd.DataFrame:
    """Load inject_ledger rows from a SQLite file or a CSV."""
    from pathlib import Path
    if not source.endswith(".csv") and not Path(source).exists():
        raise FileNotFoundError(f"no database at {Path(source).resolve()}")
    if source.endswith(".csv"):
        df = pd.read_csv(source)
    else:
        import sqlite3
        con = sqlite3.connect(source)
        try:
            q = "SELECT * FROM inject_ledger"
            params: tuple = ()
            if run_id:
                q += " WHERE run_id=?"
                params = (run_id,)
            df = pd.read_sql(q, con, params=params)
        finally:
            con.close()
    if run_id and "run_id" in df.columns:
        df = df[df["run_id"] == run_id]
    if df.empty:
        raise ValueError(f"run {run_id!r} has no rows in {source} -- wrong file or run_id?")
    return df.reset_index(drop=True)


def load_turns(source: str, run_id: Optional[str] = None) -> pd.DataFrame:
    if source.endswith(".csv"):
        df = pd.read_csv(source)
    else:
        import sqlite3
        con = sqlite3.connect(source)
        try:
            q = "SELECT * FROM turns"
            params: tuple = ()
            if run_id:
                q += " WHERE run_id=?"
                params = (run_id,)
            df = pd.read_sql(q, con, params=params)
        finally:
            con.close()
    if run_id and "run_id" in df.columns:
        df = df[df["run_id"] == run_id]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Design matrix
# --------------------------------------------------------------------------- #

@dataclass
class Design:
    X: pd.DataFrame
    y: pd.Series
    entries: list[str]
    n_turns: int

    def summary(self) -> str:
        return (f"{self.n_turns} turns, {len(self.entries)} entries, "
                f"{self.X.shape[1]} design columns")


def build_design(ledger: pd.DataFrame,
                 controls: Sequence[str] = DEFAULT_CONTROLS,
                 min_trials: int = 5) -> Design:
    """Pivot the per-row ledger into a per-turn design matrix.

    min_trials drops entries that were never withheld (or never injected) often
    enough to be identified. Estimating them anyway produces a number with no
    experiment behind it.
    """
    df = ledger.dropna(subset=["outcome_score"]).copy()
    if df.empty:
        raise ValueError("no settled rows with a numeric outcome")

    df["injected"] = (df["masked"] == 0).astype(int)

    counts = df.groupby("entry_id")["injected"].agg(["sum", "count"])
    counts["withheld"] = counts["count"] - counts["sum"]
    keep = counts[(counts["sum"] >= min_trials)
                  & (counts["withheld"] >= min_trials)].index.tolist()
    dropped = sorted(set(counts.index) - set(keep))
    if dropped:
        print(f"[design] dropping {len(dropped)} under-sampled entries: "
              f"{', '.join(dropped)}")
    if not keep:
        raise ValueError("no entry has enough present/absent trials; run more turns")

    elig = (df.pivot_table(index="turn_id", columns="entry_id",
                           values="injected", aggfunc="size", fill_value=0)
              .reindex(columns=keep, fill_value=0)
              .clip(upper=1))
    inj = (df.pivot_table(index="turn_id", columns="entry_id",
                          values="injected", aggfunc="max", fill_value=0)
             .reindex(columns=keep, fill_value=0))

    elig.columns = [f"elig__{c}" for c in elig.columns]
    inj.columns = [f"z__{c}" for c in inj.columns]

    turn_cols = ["turn_id", "outcome_score", *[c for c in controls if c in df.columns]]
    meta = df[turn_cols].drop_duplicates("turn_id").set_index("turn_id")

    X = pd.concat([inj, elig, meta.drop(columns=["outcome_score"])], axis=1)

    cat = [c for c in X.columns
           if not pd.api.types.is_numeric_dtype(X[c])]
    if cat:
        X = pd.get_dummies(X, columns=cat, drop_first=True, dtype=float)

    # constant columns carry no information and make the fit singular
    nunique = X.nunique()
    X = X.loc[:, nunique > 1]

    # Dummy trap: if every turn has the same number of candidates (always true
    # at k=1), the eligibility columns sum to a constant and are collinear with
    # the intercept. Drop the most common one as the reference category.
    elig_cols = [c for c in X.columns if c.startswith("elig__")]
    if elig_cols and X[elig_cols].sum(axis=1).nunique() == 1:
        ref = X[elig_cols].sum().idxmax()
        X = X.drop(columns=[ref])

    X = X.sort_index()
    y = meta["outcome_score"].reindex(X.index).astype(float)
    return Design(X=X.astype(float), y=y, entries=keep, n_turns=len(y))


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #

def fit_utility(ledger: pd.DataFrame,
                controls: Sequence[str] = DEFAULT_CONTROLS,
                min_trials: int = 5,
                alpha: float = 0.05,
                verbose: bool = True) -> pd.DataFrame:
    """Estimate per-entry utility. One row per entry, with a CI and trial counts."""
    import statsmodels.api as sm

    d = build_design(ledger, controls=controls, min_trials=min_trials)
    if verbose:
        print(f"[fit] {d.summary()}")

    X = sm.add_constant(d.X, has_constant="add")
    model = sm.OLS(d.y, X).fit(cov_type="HC3")

    df = ledger.dropna(subset=["outcome_score"]).copy()
    df["injected"] = (df["masked"] == 0).astype(int)
    counts = df.groupby("entry_id")["injected"].agg(n_injected="sum", n_candidate="count")
    counts["n_withheld"] = counts["n_candidate"] - counts["n_injected"]

    ci = model.conf_int(alpha=alpha)
    rows = []
    for e in d.entries:
        col = f"z__{e}"
        if col not in model.params.index:
            continue
        beta = float(model.params[col])
        se = float(model.bse[col])
        lo, hi = float(ci.loc[col, 0]), float(ci.loc[col, 1])
        rows.append({
            "entry_id": e,
            "utility": round(beta, 4),
            "se": round(se, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "p_value": round(float(model.pvalues[col]), 5),
            "n_injected": int(counts.loc[e, "n_injected"]),
            "n_withheld": int(counts.loc[e, "n_withheld"]),
            "promotable": bool(lo > 0),       # Layer 4 promote gate
            "demotable": bool(hi < 0),        # Layer 4 demote gate
        })

    out = pd.DataFrame(rows).sort_values("utility", ascending=False)
    out = out.reset_index(drop=True)
    out.attrs["r_squared"] = float(model.rsquared)
    out.attrs["n_turns"] = d.n_turns
    out.attrs["model"] = model
    if verbose:
        print(f"[fit] R^2 = {model.rsquared:.3f}")
    return out


def fit_logit(ledger: pd.DataFrame,
              controls: Sequence[str] = DEFAULT_CONTROLS,
              min_trials: int = 5) -> pd.DataFrame:
    """Robustness check: same design, logistic link, average marginal effects."""
    import statsmodels.api as sm

    d = build_design(ledger, controls=controls, min_trials=min_trials)
    X = sm.add_constant(d.X, has_constant="add")
    model = sm.Logit(d.y, X).fit(disp=False)
    margeff = model.get_margeff(at="overall")
    summ = margeff.summary_frame()
    rows = []
    for e in d.entries:
        col = f"z__{e}"
        if col not in summ.index:
            continue
        rows.append({
            "entry_id": e,
            "utility_logit_ame": round(float(summ.loc[col, "dy/dx"]), 4),
            "se": round(float(summ.loc[col, "Std. Err."]), 4),
        })
    return pd.DataFrame(rows).sort_values("utility_logit_ame", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# The baseline MERIT is arguing against
# --------------------------------------------------------------------------- #

def recall_frequency(ledger: pd.DataFrame) -> pd.DataFrame:
    """The incumbent promotion signal: how often an entry gets retrieved.

    This is what OpenClaw-style consolidation ranks candidates by -- retrieval
    relevance, recall frequency, query diversity, recurrence. Every one of those
    measures retrievability, not effect.
    """
    df = ledger.copy()
    g = df.groupby("entry_id")
    out = pd.DataFrame({
        "retrievals": g.size(),
        "query_diversity": g["task_id"].nunique(),
        "mean_rank": g["retrieval_rank"].mean().round(2),
        "mean_score": g["retrieval_score"].mean().round(3),
    })
    out["recall_rank"] = out["retrievals"].rank(ascending=False, method="min").astype(int)
    return out.reset_index().sort_values("retrievals", ascending=False).reset_index(drop=True)


def compare_rankings(utility: pd.DataFrame, recall: pd.DataFrame,
                     planted: Optional[dict] = None) -> pd.DataFrame:
    """Side by side: what each promotion rule would pick.

    The headline table. Where an entry sits high on recall and low on utility,
    the incumbent rule promotes something that does nothing.
    """
    u = utility.copy()
    u["utility_rank"] = u["utility"].rank(ascending=False, method="min").astype(int)
    m = u.merge(recall[["entry_id", "retrievals", "recall_rank"]],
                on="entry_id", how="left")
    if planted:
        m["planted"] = m["entry_id"].map(planted)
    m["rank_shift"] = m["recall_rank"] - m["utility_rank"]
    cols = ["entry_id"]
    if planted:
        cols.append("planted")
    cols += ["utility", "ci_lo", "ci_hi", "promotable", "retrievals",
             "recall_rank", "utility_rank", "rank_shift"]
    return m[cols].sort_values("utility", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Did it work?
# --------------------------------------------------------------------------- #

def recovery_report(utility: pd.DataFrame, store) -> pd.DataFrame:
    planted = {eid: e.planted for eid, e in store.entries.items()}
    u = utility.copy()
    u["planted"] = u["entry_id"].map(planted)
    return (u.groupby("planted")
             .agg(entries=("entry_id", "count"),
                  mean_utility=("utility", "mean"),
                  min_utility=("utility", "min"),
                  max_utility=("utility", "max"),
                  n_promotable=("promotable", "sum"))
             .round(3).reset_index())


def verdict(utility: pd.DataFrame, store) -> str:
    planted = {eid: e.planted for eid, e in store.entries.items()}
    u = utility.copy()
    u["planted"] = u["entry_id"].map(planted)
    useful = u[u.planted == "useful"]
    other = u[u.planted.isin(["distractor", "irrelevant"])]
    if useful.empty or other.empty:
        return "INCONCLUSIVE: a planted class survived no entries into the fit."

    promoted_useful = int(useful["promotable"].sum())
    promoted_other = int(other["promotable"].sum())

    if promoted_useful == len(useful) and promoted_other == 0:
        return (f"PASS: all {len(useful)} useful entries are promotable "
                f"(CI excludes zero) and no distractor or irrelevant entry is. "
                f"Utility-gated promotion works. Proceed to Layer 3.")
    if promoted_useful >= 1 and promoted_other == 0:
        return (f"PARTIAL: {promoted_useful}/{len(useful)} useful entries clear "
                f"the promotion gate, no false positives. The estimator is "
                f"correct but under-powered -- run more turns.")
    if promoted_other > 0:
        return (f"FAIL: {promoted_other} distractor/irrelevant entries are "
                f"marked promotable. A false positive here means Layer 4 would "
                f"promote dead weight. Check the controls before proceeding.")
    return ("FAIL: no useful entry clears the promotion gate. Either the sample "
            "is too small or the design is mis-specified.")


def power_curve(ledger: pd.DataFrame, store,
                sizes: Sequence[int] = (100, 200, 300, 400, 500, 600),
                min_trials: int = 5) -> pd.DataFrame:
    """How many turns before the promotion gate is trustworthy?

    Refit on growing prefixes of the run. The row where every useful entry is
    promotable and no distractor is, is the sample size to report.
    """
    planted = {eid: e.planted for eid, e in store.entries.items()}
    turn_order = ledger.drop_duplicates("turn_id")["turn_id"].tolist()
    rows = []
    for n in sizes:
        sub = ledger[ledger["turn_id"].isin(turn_order[:n])]
        try:
            u = fit_utility(sub, min_trials=min_trials, verbose=False)
        except Exception as exc:  # under-powered prefixes legitimately fail
            rows.append({"turns": n, "fitted": False, "note": str(exc)[:60]})
            continue
        u["planted"] = u["entry_id"].map(planted)
        useful = u[u.planted == "useful"]
        other = u[u.planted.isin(["distractor", "irrelevant"])]
        rows.append({
            "turns": n,
            "fitted": True,
            "entries_fitted": len(u),
            "useful_promotable": f"{int(useful['promotable'].sum())}/{len(useful)}",
            "false_promotions": int(other["promotable"].sum()),
            "mean_useful": round(float(useful["utility"].mean()), 3),
            "mean_other": round(float(other["utility"].mean()), 3),
            "mean_ci_width": round(float((u["ci_hi"] - u["ci_lo"]).mean()), 3),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Scalable estimator for real data: per-entry contrast + empirical-Bayes shrink
# --------------------------------------------------------------------------- #

def fit_utility_eb(ledger: pd.DataFrame, z: float = 1.96,
                   verbose: bool = True) -> pd.DataFrame:
    """Per-entry utility for stores too large for the joint regression.

    The joint model needs two columns per entry, so with hundreds of entries and
    a few hundred turns it has more parameters than observations. This estimator
    scales to any store size.

    For each entry, among the turns it was a candidate on:
        d_i = mean(outcome | injected) - mean(outcome | withheld)
    Masking is independent per entry, so d_i is an unbiased effect estimate;
    co-injected entries add variance, not bias.

    Entries with few withheld trials have huge standard errors, so each d_i is
    shrunk toward the across-entry mean by empirical Bayes (method of moments):
        d_i ~ N(mu, tau^2),  posterior mean = (tau^2 d_i + se_i^2 mu)/(tau^2 + se_i^2)
    An entry is promotable only if its posterior interval excludes zero.
    Entries never both injected and withheld are not estimated at all.
    """
    import numpy as np
    df = ledger.dropna(subset=["outcome_score"])
    rows = []
    for eid, g in df.groupby("entry_id"):
        y1 = g.loc[g["masked"] == 0, "outcome_score"].astype(float)
        y0 = g.loc[g["masked"] == 1, "outcome_score"].astype(float)
        n1, n0 = len(y1), len(y0)
        if n1 == 0 or n0 == 0:
            continue
        v1 = y1.var(ddof=1) if n1 > 1 else 0.25
        v0 = y0.var(ddof=1) if n0 > 1 else 0.25
        se = float(np.sqrt(max(v1, 1e-4) / n1 + max(v0, 1e-4) / n0))
        rows.append({"entry_id": eid, "raw": float(y1.mean() - y0.mean()),
                     "raw_se": se, "n_injected": n1, "n_withheld": n0})
    if not rows:
        raise ValueError("no entry was both injected and withheld; run more turns")
    t = pd.DataFrame(rows)

    w = 1.0 / t["raw_se"] ** 2
    mu = float((w * t["raw"]).sum() / w.sum())
    q = float((w * (t["raw"] - mu) ** 2).sum())
    k = len(t)
    c = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0

    if tau2 == 0.0:
        post, psd = np.full(k, mu), np.zeros(k) + 1e-6
    else:
        s2 = t["raw_se"] ** 2
        post = (tau2 * t["raw"] + s2 * mu) / (tau2 + s2)
        psd = np.sqrt(tau2 * s2 / (tau2 + s2))
    t["utility"] = np.round(post, 4)
    t["se"] = np.round(psd, 4)
    t["ci_lo"] = np.round(post - z * psd, 4)
    t["ci_hi"] = np.round(post + z * psd, 4)
    t["promotable"] = t["ci_lo"] > 0
    t["demotable"] = t["ci_hi"] < 0
    t = t.sort_values("utility", ascending=False).reset_index(drop=True)
    t.attrs.update({"mu": mu, "tau2": tau2, "n_entries": k})
    if verbose:
        print(f"[eb] {k} entries estimated, prior mean {mu:.3f}, "
              f"between-entry sd {tau2 ** 0.5:.3f}")
    return t
