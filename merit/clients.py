"""Model clients.

Two backends behind one interface:

  MockClient  -- deterministic, zero network, zero tokens. Build and debug the
                 whole of Layer 1 against this. It simulates an agent that uses
                 an injected artifact when one is present, which is enough to
                 exercise masking, detection and the ledger end to end.

  GroqClient  -- real calls across a rotating pool of API keys.

Rotation rule that matters: rotate on rate-limit and quota errors ONLY. A
malformed request or a content error means your code is wrong; retrying it
across four keys just burns four keys.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    key_id: Optional[str] = None
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BudgetExhausted(RuntimeError):
    """Every key is spent, or the run-level token cap was reached."""


# --------------------------------------------------------------------------- #
# Mock
# --------------------------------------------------------------------------- #

ARTIFACT_RE = re.compile(r"--[a-z][a-z0-9-]{2,}(?:=[a-z0-9._-]+)?")


class MockClient:
    """Deterministic stand-in for an agent.

    Behaviour: if an actionable option appears anywhere in the prompt (only a
    'useful' memory entry carries one), the mock uses it. Otherwise it emits a
    plausible wrong answer. That makes the ground truth of 'did the memory
    help' exactly knowable, which is what validates the detector.

    `competence` adds controlled noise: with probability (1 - competence) the
    mock fumbles even when it had the artifact. Set it below 1.0 to confirm the
    regression still recovers the effect under noise.
    """

    def __init__(self, competence: float = 1.0, seed: int = 0):
        self.competence = competence
        self.seed = seed
        self._calls = 0

    @staticmethod
    def _words(text: str) -> set[str]:
        # 5-character stems, so "build"/"building" and "machine"/"machines"
        # match. Exact whole-word matching made unrelated notes tie on generic
        # words like "dialyx", and the mock then picked whichever note came
        # first -- a failure no real model would make.
        return {w[:5] for w in re.findall(r"[a-z]{4,}", text.lower())}

    def complete(self, system: str, user: str, **_) -> Completion:
        self._calls += 1
        prompt = f"{system}\n{user}"

        # Split the notes from the question so the mock behaves like a model
        # that picks the note matching the task, not the first flag it sees.
        notes_part, _, question = user.partition("Question:")
        q_words = self._words(question)

        best_flag, best_overlap = None, -1
        for line in notes_part.splitlines():
            flags = ARTIFACT_RE.findall(line)
            if not flags:
                continue
            overlap = len(self._words(line) & q_words)
            if overlap > best_overlap:
                best_flag, best_overlap = flags[0], overlap

        # deterministic pseudo-noise, no global RNG state
        h = (hash((self.seed, self._calls)) % 10_000) / 10_000.0
        if best_flag and best_overlap > 0 and h < self.competence:
            body = f"Run it with {best_flag} applied."
        else:
            body = "Run it with the default settings."
        return Completion(
            text=body,
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max(1, len(body) // 4),
            key_id="mock",
        )


# --------------------------------------------------------------------------- #
# Groq
# --------------------------------------------------------------------------- #

@dataclass
class KeyState:
    key_id: str
    api_key: str
    spent_tokens: int = 0
    calls: int = 0
    exhausted: bool = False
    cooldown_until: float = 0.0


class GroqKeyPool:
    """Round-robin pool. Advances on quota/rate errors, never on client errors."""

    RETRYABLE = ("rate_limit", "429", "quota", "insufficient", "over capacity",
                 "503", "502", "overloaded")

    def __init__(self, keys: Optional[list[str]] = None, cooldown_s: float = 20.0):
        keys = keys or self._from_env()
        if not keys:
            raise RuntimeError(
                "No Groq keys found. Put GROQ_API_KEY_1..N in .env and "
                "load it with dotenv.load_dotenv()."
            )
        self.states = [KeyState(f"k{i+1}", k) for i, k in enumerate(keys)]
        self.cooldown_s = cooldown_s
        self._i = 0

    @staticmethod
    def _from_env() -> list[str]:
        keys = []
        single = os.getenv("GROQ_API_KEY")
        if single:
            keys.append(single)
        i = 1
        while True:
            k = os.getenv(f"GROQ_API_KEY_{i}")
            if not k:
                break
            keys.append(k)
            i += 1
        # de-duplicate, preserve order
        seen, out = set(), []
        for k in keys:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    def is_retryable(self, msg: str) -> bool:
        m = msg.lower()
        return any(tok in m for tok in self.RETRYABLE)

    def current(self) -> KeyState:
        """Next usable key. Waits out per-minute limits; raises only when every
        key has hit its DAILY limit -- the run then pauses and resumes later."""
        while True:
            live = [s for s in self.states if not s.exhausted]
            if not live:
                raise BudgetExhausted("daily limit reached on every key -- "
                                      "re-run the same cell after the reset; "
                                      "finished turns are kept")
            now = time.time()
            for _ in range(len(self.states)):
                st = self.states[self._i % len(self.states)]
                if not st.exhausted and st.cooldown_until <= now:
                    return st
                self._i += 1
            wait = min(s.cooldown_until for s in live) - now
            time.sleep(max(0.5, min(wait, 90)))

    @staticmethod
    def _wait_seconds(msg: str) -> float:
        m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)?(ms|s)?", msg)
        if not m:
            return 0.0
        h, mi, sec, unit = m.groups()
        t = int(h or 0) * 3600 + int(mi or 0) * 60
        if sec:
            t += float(sec) / (1000 if unit == "ms" else 1)
        return t

    @staticmethod
    def is_daily(msg: str) -> bool:
        m = msg.lower()
        return ("per day" in m or "(tpd)" in m or "(rpd)" in m
                or "quota" in m or "insufficient" in m)

    def advance(self, st: KeyState, msg: str = "", exhausted: bool = False) -> None:
        if exhausted or self.is_daily(msg):
            st.exhausted = True
        else:
            st.cooldown_until = time.time() + max(self._wait_seconds(msg), 2.0) + 0.5
        self._i += 1

    def report(self) -> list[dict]:
        return [
            {"key_id": s.key_id, "calls": s.calls, "spent_tokens": s.spent_tokens,
             "exhausted": s.exhausted}
            for s in self.states
        ]

    @property
    def total_spent(self) -> int:
        return sum(s.spent_tokens for s in self.states)


class GroqClient:
    def __init__(self, model: str, pool: Optional[GroqKeyPool] = None,
                 temperature: float = 0.0, max_tokens: int = 256,
                 max_rotations: int = 200, reasoning_effort: Optional[str] = None):
        from groq import Groq  # imported lazily so mock mode needs no groq
        self._Groq = Groq
        self.model = model
        self.pool = pool or GroqKeyPool()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_rotations = max_rotations
        self.reasoning_effort = reasoning_effort
        self._clients: dict[str, object] = {}

    def _client_for(self, st: KeyState):
        if st.key_id not in self._clients:
            self._clients[st.key_id] = self._Groq(api_key=st.api_key)
        return self._clients[st.key_id]

    def complete(self, system: str, user: str, **_) -> Completion:
        last_err = None
        for _ in range(self.max_rotations):
            st = self.pool.current()
            client = self._client_for(st)
            try:
                kw = dict(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                if self.reasoning_effort:
                    kw["reasoning_effort"] = self.reasoning_effort
                resp = client.chat.completions.create(**kw)
                text = resp.choices[0].message.content or ""
                u = resp.usage
                pt = getattr(u, "prompt_tokens", 0) or 0
                ct = getattr(u, "completion_tokens", 0) or 0
                st.calls += 1
                st.spent_tokens += pt + ct
                return Completion(text, pt, ct, key_id=st.key_id)
            except Exception as exc:  # noqa: BLE001 - we classify below
                msg = str(exc)
                last_err = msg
                if self.reasoning_effort and "reasoning_effort" in msg:
                    print("[groq] reasoning_effort not accepted; continuing without it")
                    self.reasoning_effort = None
                    continue
                if self.pool.is_retryable(msg):
                    self.pool.advance(st, msg=msg)
                    continue
                # not a quota problem -- your request is wrong. Stop.
                return Completion("", 0, 0, key_id=st.key_id, error=msg)
        raise BudgetExhausted(f"rotated through every key; last error: {last_err}")


def build_client(cfg) -> object:
    if cfg.backend == "mock":
        return MockClient(seed=cfg.seed)
    if cfg.backend == "groq":
        return GroqClient(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            reasoning_effort=getattr(cfg, "reasoning_effort", None),
        )
    raise ValueError(f"unknown backend: {cfg.backend!r}")
