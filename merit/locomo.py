"""Stage 5 data: LoCoMo, a public long-term conversational memory benchmark.

Why LoCoMo
----------
- It is what Mem0, Zep and harness-mem already report on, so reviewers know it.
- Every observation carries the dialogue ID it came from, and every question
  lists its evidence IDs. So we know which entries a question NEEDS -- used here
  only for the oracle arm and for analysis, never by retrieval, masking, the
  model, or the estimator.
- It was not designed by us, so it cannot have been designed to make MERIT win.

What it is NOT good at
----------------------
Reuse. An evidence observation serves ~1.7 questions on average. MERIT learns an
entry's value from past use, so on a held-out split only ~31-41% of test
questions depend on evidence any learning-phase question touched. LoCoMo is a
hard test for MERIT's premise, and the results must be read in that light.

Mapping
-------
  memory entry  = one LoCoMo observation, prefixed with its session date
  task          = one question (category 5 / adversarial excluded, as is standard)
  outcome       = token F1 against the gold answer (LoCoMo's standard metric)
  task family   = LoCoMo question category
"""
from __future__ import annotations

import json
import random
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .clients import Completion
from .store import Entry, MemoryStore

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

QA_SYSTEM_PROMPT = (
    "You answer questions about a long conversation between two people, using "
    "the memory notes provided. Reply with the shortest possible answer: a name, "
    "date, number or short phrase. No explanation, no full sentence. If the notes "
    "do not contain the answer, give your single best guess."
)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_locomo(path: Optional[str] = None) -> list[dict]:
    """Load locomo10.json, downloading it into data/ on first use."""
    if path is None:
        root = Path(__file__).resolve().parent.parent
        path = str(root / "data" / "locomo10.json")
    p = Path(path)
    if not p.exists():
        import urllib.request
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"[locomo] downloading to {p}")
        urllib.request.urlretrieve(LOCOMO_URL, p)
    return json.loads(p.read_text())


def _session_dates(conv: dict) -> dict[int, str]:
    out = {}
    for k, v in conv["conversation"].items():
        m = re.fullmatch(r"session_(\d+)_date_time", k)
        if m:
            # "1:56 pm on 8 May, 2023" -> "8 May, 2023"
            out[int(m.group(1))] = v.split(" on ")[-1].strip()
    return out


# --------------------------------------------------------------------------- #
# Scoring -- LoCoMo / SQuAD-style token F1
# --------------------------------------------------------------------------- #

_PUNCT = set(string.punctuation)


def normalize(s: str) -> str:
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #

@dataclass
class QATask:
    task_id: str
    family: str
    prompt: str
    gold: str
    evidence: list[str]                     # analysis/oracle only
    required_artifact: Optional[str] = None  # API compatibility with the fixture

    def score(self, output: str) -> float:
        return round(token_f1(output, self.gold), 4)


@dataclass
class Conversation:
    index: int
    sample_id: str
    store: MemoryStore
    tasks: list[QATask]
    dia_to_entries: dict[str, list[str]] = field(default_factory=dict)

    def evidence_entries(self, tasks: list[QATask]) -> set[str]:
        out = set()
        for t in tasks:
            for d in t.evidence:
                out.update(self.dia_to_entries.get(d, []))
        return out


def build_conversation(data: list[dict], index: int,
                       include_turns: bool = False) -> Conversation:
    """include_turns=False: store = curated observations only (clean store).
    include_turns=True:  store = observations + every raw dialogue turn, i.e.
                         what naive auto-capture keeps. Evidence turns carry
                         answers; most other turns are chit-chat that shares
                         names and topics with questions but answers nothing.
    """
    c = data[index]
    dates = _session_dates(c)
    entries: list[Entry] = []
    dia_to_entries: dict[str, list[str]] = {}
    n = 0
    for skey, per_speaker in c["observation"].items():
        sess = int(re.search(r"session_(\d+)", skey).group(1))
        date = dates.get(sess, "")
        for _speaker, items in per_speaker.items():
            for it in items:
                text, dia = it[0], it[1]
                eid = f"c{index}_o{n:04d}"
                n += 1
                body = f"[{date}] {text}" if date else text
                entries.append(Entry(eid, body, planted="none", artifact=None,
                                     tags=[f"session:{sess}"]))
                for d in (dia if isinstance(dia, list) else [dia]):
                    dia_to_entries.setdefault(d, []).append(eid)

    if include_turns:
        conv = c["conversation"]
        for k, turns in conv.items():
            m = re.fullmatch(r"session_(\d+)", k)
            if not m or not isinstance(turns, list):
                continue
            sess = int(m.group(1))
            date = dates.get(sess, "")
            for tr in turns:
                text = tr.get("text") or ""
                if tr.get("blip_caption"):
                    text = f"{text} [shares a photo: {tr['blip_caption']}]"
                if not text.strip():
                    continue
                eid = f"c{index}_t{n:04d}"
                n += 1
                body = f"[{date}] {tr.get('speaker', '')}: {text}".strip()
                entries.append(Entry(eid, body, planted="turn", artifact=None,
                                     tags=[f"session:{sess}", "kind:turn"]))
                dia_to_entries.setdefault(tr.get("dia_id"), []).append(eid)

    tasks: list[QATask] = []
    for qi, q in enumerate(c["qa"]):
        if q.get("category") == 5:
            continue
        ev = q.get("evidence") or []
        if not ev or not all(d in dia_to_entries for d in ev):
            continue          # unanswerable from the store; excluded, as stated
        tasks.append(QATask(
            task_id=f"c{index}_q{qi:03d}",
            family=f"cat{q['category']}",
            prompt=q["question"],
            gold=str(q["answer"]),
            evidence=list(ev),
        ))
    return Conversation(index, c.get("sample_id", str(index)),
                        MemoryStore(entries), tasks, dia_to_entries)


def eligible_counts(data: list[dict]) -> list[tuple[int, int, int]]:
    """(conversation index, eligible questions, observations) for every conversation."""
    out = []
    for i in range(len(data)):
        cv = build_conversation(data, i)
        out.append((i, len(cv.tasks), len(cv.store)))
    return out


def split_tasks(tasks: list[QATask], seed: int, frac: float = 0.5):
    """Random question-level split into a learning half and a held-out test half."""
    tt = list(tasks)
    random.Random(seed).shuffle(tt)
    k = int(len(tt) * frac)
    return tt[:k], tt[k:]


def transfer_ceiling(conv: Conversation, learn: list[QATask],
                     test: list[QATask]) -> dict:
    """Share of test questions whose evidence was touched during learning.

    MERIT learns value from use. A test question whose evidence no learning
    question ever used is one where MERIT has, by construction, nothing to go on.
    """
    seen = {d for t in learn for d in t.evidence}
    anyseen = sum(any(d in seen for d in t.evidence) for t in test)
    allseen = sum(all(d in seen for d in t.evidence) for t in test)
    return {"test_questions": len(test),
            "any_evidence_seen": round(anyseen / len(test), 3),
            "all_evidence_seen": round(allseen / len(test), 3)}


# --------------------------------------------------------------------------- #
# Mock for zero-token development
# --------------------------------------------------------------------------- #

class QAMock:
    """Answers with the gold answer iff an evidence observation reached the prompt.

    For debugging the Stage 5 pipeline without spending tokens. It is a
    simulator, and a generous one: real models will be noisier, will sometimes
    answer from priors, and will sometimes miss evidence that is present.
    """

    def __init__(self, conv: Conversation, competence: float = 0.85, seed: int = 0):
        self.competence = competence
        self.seed = seed
        self.by_question = {}
        for t in conv.tasks:
            texts = [conv.store.get(e).text for d in t.evidence
                     for e in conv.dia_to_entries.get(d, [])]
            self.by_question[t.prompt] = (t.gold, texts)

    def complete(self, system: str, user: str, **_) -> Completion:
        _, _, question = user.partition("Question:")
        gold, texts = self.by_question.get(question.strip(), ("", []))
        present = any(tx in user for tx in texts)
        # Stateless noise: the same prompt always gets the same draw, so a run
        # that is paused and resumed behaves exactly like one that never stopped.
        import hashlib
        h = int(hashlib.md5(f"{self.seed}|{user}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if present and h < self.competence:
            out = gold
        else:
            out = "not mentioned"
        return Completion(out, max(1, len(user) // 4), max(1, len(out) // 4), key_id="mock")
