"""Layer 3 evaluation: does utility-ranked retrieval beat relevance retrieval?

Two experiments, both on the planted fixture, both A/B against the same seed:

  E1  One memory slot (k=1). Relevance puts the needed entry first on only 17 of
      24 tasks; on the other 7 a distractor or wrong-family entry outranks it.
      Question: does the utility policy learn, from a cold start, to fix those?

  E2  Four slots (k=4), the Layer 1 setting. Relevance always injects four.
      The utility policy injects only entries whose sampled utility is
      positive. Question: same outcome at fewer injected tokens?

Both arms keep Layer 1's masking on, at the same rate, so the comparison is
fair and the utility arm can keep learning.
"""
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from .config import Config
from .runner import Layer1Runner
from .store import MemoryStore
from .tasks import Task, build_store, build_tasks


def misranked_tasks(store: MemoryStore, tasks: list[Task]) -> set[str]:
    """Tasks where relevance top-1 is NOT the entry carrying the needed flag."""
    bad = set()
    for t in tasks:
        top = store.retrieve(t.prompt, top_k=1)
        if not top or top[0].entry.artifact != t.required_artifact:
            bad.add(t.task_id)
    return bad


def run_arm(name: str, cfg: Config, policy_factory: Optional[Callable] = None,
            client=None, store: Optional[MemoryStore] = None,
            tasks: Optional[list[Task]] = None, verbose: bool = False):
    store = store or build_store()
    tasks = tasks or build_tasks()
    policy = policy_factory(store) if policy_factory else None
    with Layer1Runner(cfg, store=store, tasks=tasks, client=client,
                      policy=policy) as run:
        run.run(verbose=verbose)
        turns = run.ledger.turns_dataframe(cfg.run_id)
    turns = turns.reset_index(drop=True)
    turns["arm"] = name
    turns["turn_index"] = range(1, len(turns) + 1)
    return turns, policy


def summarize(turns: pd.DataFrame, misranked: set[str]) -> pd.DataFrame:
    rows = []
    for arm, g in turns.groupby("arm", sort=False):
        g = g.dropna(subset=["outcome_score"])
        hard = g[g["task_id"].isin(misranked)]
        easy = g[~g["task_id"].isin(misranked)]
        tok = g["bundle_tokens"].mean()
        rows.append({
            "arm": arm,
            "turns": len(g),
            "outcome": round(g["outcome_score"].mean(), 3),
            "outcome_misranked": round(hard["outcome_score"].mean(), 3) if len(hard) else None,
            "outcome_rest": round(easy["outcome_score"].mean(), 3) if len(easy) else None,
            "entries_injected": round(g["n_injected"].mean(), 2),
            "memory_tokens": round(tok, 1),
            "outcome_per_100_mem_tokens": round(100 * g["outcome_score"].mean() / tok, 3) if tok else None,
        })
    return pd.DataFrame(rows)


def learning_curve(turns: pd.DataFrame, misranked: set[str],
                   window: int = 40) -> pd.DataFrame:
    """Rolling outcome on the misranked tasks, per arm. The learning figure."""
    out = []
    for arm, g in turns.groupby("arm", sort=False):
        g = g.dropna(subset=["outcome_score"]).copy()
        g = g[g["task_id"].isin(misranked)]
        g["block"] = (g["turn_index"] - 1) // window
        b = g.groupby("block")["outcome_score"].mean().reset_index()
        b["turns_up_to"] = (b["block"] + 1) * window
        b["arm"] = arm
        out.append(b[["arm", "turns_up_to", "outcome_score"]])
    wide = (pd.concat(out).pivot(index="turns_up_to", columns="arm",
                                 values="outcome_score").round(3))
    return wide.reset_index()
