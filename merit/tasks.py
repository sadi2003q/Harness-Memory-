"""The planted-memory fixture.

This is the validation set for the instrument, not the final benchmark. Every
task has a checkable ground truth, so Layer 1's smoke test needs no verifier
model at all -- which is where most of the token saving comes from.

The design: fictional project conventions that a model cannot possibly guess.
Each task can only be answered correctly if a specific memory entry reached the
prompt. That gives us three known classes of entry:

  useful      -- contains the flag the task needs. Effect should be strongly +.
  distractor  -- same topic, retrieves well, contains nothing actionable.
                 Effect should be ~0. This is the entry a recall-frequency
                 promoter would wrongly promote.
  irrelevant  -- off topic. Effect ~0, and it should rarely be retrieved.

If the regression cannot separate 'useful' from 'distractor', the instrument is
broken and nothing downstream will work. That is the whole point of Phase 0.
"""
from __future__ import annotations

from dataclasses import dataclass

from .store import Entry, MemoryStore


@dataclass
class Task:
    task_id: str
    family: str
    prompt: str
    required_artifact: str   # ground truth: output must contain this to score 1

    def score(self, output: str) -> float:
        return float(self.required_artifact.lower() in output.lower())


# --------------------------------------------------------------------------- #
# Memory entries
# --------------------------------------------------------------------------- #

def build_store() -> MemoryStore:
    entries = [
        # --- useful: carry the artifact the task needs ----------------------
        Entry("e_test_flag", "The dialyx test suite hangs on CI unless you pass "
                        "--runner-profile=ci7 to the runner.",
              planted="useful", artifact="--runner-profile=ci7", tags=["test"]),
        Entry("e_build_flag", "Building dialyx on arm machines needs "
                              "--arch-profile=a9 or the wheel is unusable.",
              planted="useful", artifact="--arch-profile=a9", tags=["build"]),
        Entry("e_deploy_flag", "Deploys to the staging cluster must carry "
                               "--rollout-profile=d3 or in-flight requests are dropped.",
              planted="useful", artifact="--rollout-profile=d3", tags=["deploy"]),
        # --- distractors: same topic, high retrieval score, nothing to act on
        Entry("e_test_noise", "The dialyx test suite is slow and the runner "
                              "prints a lot of output while it works.",
              planted="distractor", artifact=None, tags=["test"]),
        Entry("e_build_noise", "Building dialyx takes a while on arm machines "
                               "and the wheel ends up fairly large.",
              planted="distractor", artifact=None, tags=["build"]),
        Entry("e_deploy_noise", "Deploys to the staging cluster are announced in "
                                "the team channel before they start.",
              planted="distractor", artifact=None, tags=["deploy"]),

        # --- irrelevant ------------------------------------------------------
        Entry("e_misc_1", "The quarterly planning review moved to Thursdays.",
              planted="irrelevant", artifact=None, tags=["misc"]),
        Entry("e_misc_2", "Office coffee machine reservations open each Monday.",
              planted="irrelevant", artifact=None, tags=["misc"]),
        Entry("e_misc_3", "The design system uses an eight point spacing scale.",
              planted="irrelevant", artifact=None, tags=["misc"]),
        Entry("e_misc_4", "Invoices are submitted through the finance portal.",
              planted="irrelevant", artifact=None, tags=["misc"]),
    ]
    return MemoryStore(entries)


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #

_TEST_PROMPTS = [
    "How do I run the dialyx test suite on CI?",
    "The dialyx test runner hangs on CI. What command should I use?",
    "Give me the exact command to run dialyx tests in the CI container.",
    "What do I need to add so the dialyx suite finishes on CI?",
    "CI never finishes the dialyx tests. How do I invoke the runner?",
    "What is the correct way to start the dialyx test suite in CI?",
    "Our dialyx CI job stalls at the test step. What is the fix?",
    "Which option does the dialyx test runner need under CI?",
]

_BUILD_PROMPTS = [
    "How do I build dialyx on an arm machine?",
    "The dialyx wheel from my arm build does not work. What should I pass?",
    "Give me the dialyx build command for arm.",
    "What option does a dialyx arm build need?",
    "Building dialyx on arm produces a broken wheel. How do I build it?",
    "What is the correct dialyx build invocation on arm hardware?",
    "My arm build of dialyx is unusable. Which flag am I missing?",
    "How should dialyx be compiled on arm?",
]

_DEPLOY_PROMPTS = [
    "How do I deploy dialyx to the staging cluster?",
    "Staging deploys are dropping requests. What should the deploy command be?",
    "Give me the dialyx staging deploy command.",
    "What option does a staging deploy need to avoid dropped requests?",
    "How do I push dialyx to staging safely?",
    "Our staging deploy kills in-flight requests. What is the fix?",
    "Which flag belongs on the dialyx staging deploy?",
    "What is the correct way to deploy dialyx to staging?",
]



def build_tasks() -> list[Task]:
    out: list[Task] = []
    for i, p in enumerate(_TEST_PROMPTS):
        out.append(Task(f"t_test_{i:02d}", "test", p, "--runner-profile=ci7"))
    for i, p in enumerate(_BUILD_PROMPTS):
        out.append(Task(f"t_build_{i:02d}", "build", p, "--arch-profile=a9"))
    for i, p in enumerate(_DEPLOY_PROMPTS):
        out.append(Task(f"t_deploy_{i:02d}", "deploy", p, "--rollout-profile=d3"))
    return out


SYSTEM_PROMPT = (
    "You are a terse engineering assistant for the dialyx project. "
    "Answer in one short sentence with the exact command or option. "
    "If project notes are provided, follow them exactly. "
    "Never invent an option you were not told about."
)
