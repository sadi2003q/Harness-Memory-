"""The Inject Ledger.

One row per (turn x candidate entry). Written twice: opened BEFORE the model
call, closed AFTER the turn settles.

The critical property: rows are written for MASKED candidates too. A row with
masked=1 records that an entry was retrieved, was eligible, and was
deliberately withheld -- then carries the same outcome_score as every other
row in that turn. Those rows are half of Layer 2's regression. Drop them and
there is no counterfactual and nothing to estimate.
"""
from __future__ import annotations

import sqlite3
from turtle import reset
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    config_json   TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    note          TEXT
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id         TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    task_id         TEXT NOT NULL,
    task_family     TEXT NOT NULL,
    bundle_id       TEXT NOT NULL,
    n_candidates    INTEGER NOT NULL,
    n_injected      INTEGER NOT NULL,
    bundle_tokens   INTEGER NOT NULL,
    harness_state   TEXT NOT NULL,
    model           TEXT NOT NULL,
    backend         TEXT NOT NULL,
    seed            INTEGER NOT NULL,
    started_at      TEXT NOT NULL,
    settled_at      TEXT,
    outcome_score   REAL,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    turn_tokens     INTEGER,
    key_id          TEXT,
    n_tool_calls    INTEGER,
    response_text   TEXT,
    error           TEXT
);

CREATE TABLE IF NOT EXISTS inject_ledger (
    inject_id        TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL,
    turn_id          TEXT NOT NULL,
    task_id          TEXT NOT NULL,
    task_family      TEXT NOT NULL,
    entry_id         TEXT NOT NULL,
    bundle_id        TEXT NOT NULL,

    -- write 1: before the call
    masked           INTEGER NOT NULL,
    mask_p           REAL    NOT NULL,
    position         INTEGER,
    entry_tokens     INTEGER NOT NULL,
    retrieval_rank   INTEGER NOT NULL,
    retrieval_score  REAL    NOT NULL,
    harness_state    TEXT    NOT NULL,
    model            TEXT    NOT NULL,
    seed             INTEGER NOT NULL,
    injected_at      TEXT    NOT NULL,

    -- write 2: after it settles
    settled          INTEGER NOT NULL DEFAULT 0,
    used_lexical     REAL,
    used_semantic    REAL,
    used_behavioural INTEGER,
    outcome_score    REAL,
    turn_tokens      INTEGER,
    key_id           TEXT,
    settled_at       TEXT,
    error            TEXT
);

CREATE INDEX IF NOT EXISTS ix_ledger_entry ON inject_ledger(entry_id);
CREATE INDEX IF NOT EXISTS ix_ledger_turn  ON inject_ledger(turn_id);
CREATE INDEX IF NOT EXISTS ix_ledger_run   ON inject_ledger(run_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Ledger:
    def __init__(self, db_path: str = "data/merit.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(db_path)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(SCHEMA)
        self.con.commit()

    # -- lifecycle -----------------------------------------------------------

    def start_run(self, run_id: str, config_json: str, note: str = "",
                  reset: bool = True) -> None:
        if reset:
            self.con.execute("DELETE FROM inject_ledger WHERE run_id=?", (run_id,))
            self.con.execute("DELETE FROM turns WHERE run_id=?", (run_id,))
        self.con.execute(
            "INSERT OR REPLACE INTO runs(run_id, config_json, started_at, note) "
            "VALUES (?,?,?,?)",
            (run_id, config_json, _now(), note),
        )
        self.con.commit()

    def finish_run(self, run_id: str) -> None:
        self.con.execute(
            "UPDATE runs SET finished_at=? WHERE run_id=?", (_now(), run_id)
        )
        self.con.commit()

    # -- write 1 -------------------------------------------------------------

    def open_turn(self, **kw) -> None:
        cols = ",".join(kw)
        marks = ",".join("?" * len(kw))
        self.con.execute(
            f"INSERT INTO turns({cols}, started_at) VALUES ({marks}, ?)",
            (*kw.values(), _now()),
        )
        self.con.commit()

    def open_rows(self, rows: Iterable[dict]) -> list[str]:
        """Write one open row per candidate. Returns the inject_ids, in order."""
        rows = list(rows)
        if not rows:
            return []
        ids = []
        payload = []
        now = _now()
        for r in rows:
            iid = new_id("inj")
            ids.append(iid)
            payload.append(
                (
                    iid, r["run_id"], r["turn_id"], r["task_id"], r["task_family"],
                    r["entry_id"], r["bundle_id"], int(r["masked"]), float(r["mask_p"]),
                    r.get("position"), int(r["entry_tokens"]), int(r["retrieval_rank"]),
                    float(r["retrieval_score"]), r["harness_state"], r["model"],
                    int(r["seed"]), now,
                )
            )
        self.con.executemany(
            "INSERT INTO inject_ledger("
            "inject_id, run_id, turn_id, task_id, task_family, entry_id, bundle_id,"
            "masked, mask_p, position, entry_tokens, retrieval_rank, retrieval_score,"
            "harness_state, model, seed, injected_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            payload,
        )
        self.con.commit()
        return ids

    # -- write 2 -------------------------------------------------------------

    def close_turn(self, turn_id: str, **kw) -> None:
        sets = ",".join(f"{k}=?" for k in kw)
        self.con.execute(
            f"UPDATE turns SET {sets}, settled_at=? WHERE turn_id=?",
            (*kw.values(), _now(), turn_id),
        )
        self.con.commit()

    def close_rows(self, turn_id: str, per_entry: dict[str, dict],
                   outcome_score: float, turn_tokens: int,
                   key_id: Optional[str], error: Optional[str] = None) -> int:
        """Close every row of a turn.

        per_entry maps entry_id -> {used_lexical, used_semantic, used_behavioural}.
        Masked entries get use signals of 0 by construction; they still receive
        the turn's outcome_score, which is exactly what makes them informative.
        """
        cur = self.con.execute(
            "SELECT inject_id, entry_id, masked FROM inject_ledger WHERE turn_id=?",
            (turn_id,),
        )
        updates = []
        now = _now()
        for row in cur.fetchall():
            sig = per_entry.get(row["entry_id"], {}) if not row["masked"] else {}
            updates.append(
                (
                    1,
                    float(sig.get("used_lexical", 0.0)),
                    float(sig.get("used_semantic", 0.0)),
                    int(sig.get("used_behavioural", 0)),
                    float(outcome_score),
                    int(turn_tokens),
                    key_id,
                    now,
                    error,
                    row["inject_id"],
                )
            )
        self.con.executemany(
            "UPDATE inject_ledger SET settled=?, used_lexical=?, used_semantic=?,"
            "used_behavioural=?, outcome_score=?, turn_tokens=?, key_id=?,"
            "settled_at=?, error=? WHERE inject_id=?",
            updates,
        )
        self.con.commit()
        return len(updates)

    # -- reads ---------------------------------------------------------------

    def open_row_count(self, run_id: str) -> int:
        """Rows opened but never closed -- turns that crashed mid-flight."""
        cur = self.con.execute(
            "SELECT COUNT(*) FROM inject_ledger WHERE run_id=? AND settled=0",
            (run_id,),
        )
        return cur.fetchone()[0]

    def dataframe(self, run_id: Optional[str] = None):
        import pandas as pd
        q = "SELECT * FROM inject_ledger"
        params: tuple = ()
        if run_id:
            q += " WHERE run_id=?"
            params = (run_id,)
        return pd.read_sql(q, self.con, params=params)

    def turns_dataframe(self, run_id: Optional[str] = None):
        import pandas as pd
        q = "SELECT * FROM turns"
        params: tuple = ()
        if run_id:
            q += " WHERE run_id=?"
            params = (run_id,)
        return pd.read_sql(q, self.con, params=params)

    def close(self) -> None:
        self.con.close()


@contextmanager
def ledger(db_path: str = "data/merit.db"):
    lg = Ledger(db_path)
    try:
        yield lg
    finally:
        lg.close()
