# MERIT — Stage 5 pre-registration

**Commit this file and the code to git BEFORE running any confirmatory turn.**
Record the commit hash below. Nothing in this file may change after that commit;
any later deviation goes in a dated "Deviations" section at the end, with a reason.

- Commit hash at registration: `________________`
- Date registered: `________________`

---

## 1. Claim under test

When a memory store must shrink, keeping entries by **measured utility**
preserves more answer quality on **held-out** questions than keeping them by
**recall frequency** — the retrievability signal incumbent memory systems rank on.

## 2. Development history (disclosed in full)

All of this happened before registration, on **conversation 7 only**, using a
simulated model (`QAMock`), with zero real model calls:

1. On the **clean store** (curated observations only), utility-based forgetting
   did *not* beat recall frequency: −0.040 F1, CI [−0.133, +0.040]. Diagnosis:
   LoCoMo has little reuse — an evidence observation serves ~1.7 questions —
   and almost no entries that retrieve well but carry nothing.
2. The **auto-capture store** track (observations + every raw dialogue turn,
   i.e. what naive auto-capture keeps) was adopted **after** observing (1).
   On it, utility led recall by +0.094 F1, CI [+0.021, +0.167].
   An earlier version of the simulator, with stateful noise, gave −0.080 and
   +0.062 for the same two comparisons. The simulator's estimates therefore
   move by several points with its noise model and are treated as directional
   only.
3. A per-entry empirical-Bayes estimator replaced the joint regression, because
   the regression has more parameters than observations at this store size. It
   was validated on the planted fixture before use.

Conversation 7 is therefore the **development** conversation and is excluded
from every confirmatory analysis.

## 3. Hypotheses

| ID | Track | Comparison | Prediction |
| --- | --- | --- | --- |
| **H1 (primary)** | auto-capture | utility − recall | > 0 |
| H2 (secondary) | auto-capture | utility − random | > 0 |
| B1 (boundary) | clean | utility − recall | **no directional prediction**; development suggested < 0. Reported whatever it shows. |

## 4. Data and selection rules (fixed in advance)

- Dataset: LoCoMo, `locomo10.json` from `snap-research/locomo`.
- Questions: category 5 (adversarial) excluded; a question is eligible only if
  every evidence ID maps to an entry in that track's store.
- **Primary conversations (H1, H2):** the 4 conversations with the most eligible
  auto-capture questions, excluding conversation 7 → **3, 4, 9, 8**.
- **Boundary conversations (B1):** the 2 conversations with the most eligible
  clean-store questions, excluding conversation 7 → **3, 2**.
- Split: random 50/50 question split, seed `20261006`, per conversation.

## 5. Fixed settings

| Setting | Value |
| --- | --- |
| Model | `openai/gpt-oss-20b` via Groq, `reasoning_effort="low"`, temperature 0, max_tokens 512 |
| Retrieval | BM25, top-k = 5 |
| Learning | 4 shuffled passes over the learning questions, masking p = 0.30 |
| Estimator | `fit_utility_eb` (per-entry contrast, empirical-Bayes shrinkage) |
| Decision | `forget(rule, keep)`, keep = 30% of the store, rules as implemented at the registered commit |
| Test | each held-out question answered once per arm, masking off, same retriever and k |
| Arms | `utility`, `recall`, `random`. (The no-forgetting `full` arm appears in no hypothesis and is omitted to save budget.) |
| Outcome | LoCoMo token F1 vs gold answer; empty completions excluded |
| Seed | `20261006` throughout |

## 6. Analysis and decision rule

- Statistic: mean paired difference in F1, over the **same** held-out questions,
  pooled across the primary conversations.
- Interval: 95% stratified paired bootstrap (strata = conversation),
  10,000 resamples, seed `20261006`.

| Result for H1 | Verdict |
| --- | --- |
| CI entirely above 0 | **Supported** |
| CI includes 0 | **Inconclusive** — report effect size and CI; do not claim support |
| CI entirely below 0 | **Refuted** |

**Monitoring during the run is blind.** Process health (progress, tokens, empty
answers, key status), learning-phase F1, and estimator readiness are watched
live. Held-out arm F1 — the H1 comparison — is hidden (`BLIND = True`) until
every primary conversation has finished. Unblinding earlier is recorded as a
deviation, and the run is never stopped, extended or altered because of an
interim arm result.

No additional conversations, arms, keep fractions or estimator changes may be
added and then used to change the H1 verdict. Anything run afterwards is
exploratory and labelled as such.

## 7. Power, stated honestly

Pooled across the four primary conversations there are ~343 held-out questions.
From the development run, the per-question SD of the paired difference is
~0.36, giving a 95% CI half-width of roughly ±0.04 F1. The simulated
development effect would be detectable, but the simulator is generous and its
estimates are rough; a real-model effect is likely smaller. **Inconclusive is a
realistic outcome**, and it is an acceptable one to report.

## 8. Budget

~3,300 turns: primary 2,385 (learning 1,356 + test 3 × 343) and boundary 924.
Roughly 1.2–1.7M tokens depending on reasoning length; calibrated by the smoke
test before registration.

Groq's free tier limits `gpt-oss-20b` to 200K tokens/day, 1,000 requests/day
and 8K tokens/minute, **per organization, not per key**. Keys under one account
share one budget. The run therefore spans several days. It pauses cleanly when
the daily limit is reached and resumes from the exact turn where it stopped,
with identical masking decisions; this was verified against an uninterrupted
run before registration.

---

## Deviations

_(dated entries only, each with a reason)_
