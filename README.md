# MERIT — Layer 1: the Inject Ledger

Layer 1 improves nothing and decides nothing. It produces the dataset Layer 2
regresses on, and it answers one question before you spend a month on the rest:

> Is the effect of an individual memory entry detectable at all?

## Install

```bash
conda activate merit
pip install -r requirements.txt
```

`sentence-transformers` is optional and commented out — only needed if you set
`Config.semantic_enabled = True`.

## Run

```bash
jupyter lab notebooks/layer1.ipynb
```

Sections 1–6 use the mock backend and cost **zero Groq tokens**. Section 7 is
the live smoke test; run it once, after the mock checks pass.

Or from the command line:

```python
from merit import Config, Layer1Runner, build_store
from merit import validate as V

cfg = Config(run_id="mock", backend="mock", max_turns=240)
run = Layer1Runner(cfg)
run.run()
df = run.ledger.dataframe("mock")
print(V.verdict(V.naive_effect(df, build_store())))
run.close()
```

## What each module does

| File | Role |
| --- | --- |
| `config.py` | Every knob. A run is reproducible from `(Config, seed)` |
| `ledger.py` | SQLite schema, write 1 (open) and write 2 (close) |
| `store.py` | Memory entries + BM25 retrieval. Deliberately plain |
| `clients.py` | `MockClient` (free) and `GroqClient` (key pool, rotation) |
| `detect.py` | Three use signals: lexical, semantic, behavioural |
| `tasks.py` | The planted-memory fixture with ground-truth scoring |
| `runner.py` | The turn loop |
| `validate.py` | The Phase 0 checks and the pass/fail verdict |

## The one thing to understand

**Masked rows are written too.** When the masker withholds a candidate, that
candidate still gets a ledger row — with `masked=1`, no use signals, and the
same `outcome_score` as every other row in the turn.

Those rows are the counterfactual. They record a turn where the entry was
retrieved, was eligible, and was deliberately absent. Without them there is
nothing to compare against and Layer 2 has no effect to estimate. If you ever
find yourself "optimising" by skipping those writes, you have deleted the
experiment.

## The planted fixture

Three classes of entry, and the detector never sees the labels:

- **useful** — carries the flag the task needs. Effect should be strongly positive.
- **distractor** — same topic, retrieves just as well, nothing actionable.
  Effect should be ~0. *This is the entry a recall-frequency promoter would
  wrongly promote, which is the thing MERIT exists to fix.*
- **irrelevant** — off topic. Rarely retrieved.

Tasks use fictional project conventions (`--no-sandbox`, `--target-arch-native`,
`--drain-first`) that no model can guess, so outcome scoring is exact string
ground truth — no verifier model, no tokens.

## Reading the Phase 0 output

**Integrity** must be all green. An unsettled row means a turn crashed between
write 1 and write 2; a turn whose rows disagree on outcome means the close path
is buggy.

**Detector report** — `behavioural_rate` should be clearly positive for
`useful` and ~0 elsewhere. Against the mock, `lexical_mean` stays low because
mock replies are one line; against a real model it carries more weight.

**Naive effect** — ignore the `irrelevant` rows. They are rarely retrieved, so
`n_absent` is often 1–4 and their numbers are pure noise. Judge the verdict on
`useful` vs `distractor` only.

## Known limits, on purpose

- The naive effect in `validate.py` has **no controls** — not for task family,
  bundle size, or co-injection. That is Layer 2's job. This is a smoke test.
- Single-turn attribution. A memory that prevents a mistake three turns later
  gets no credit here.
- The mock is a mock. It models an agent that picks the topically closest note
  and applies it, with tunable competence. It is for debugging the plumbing,
  not for producing results.

## Gotcha

Two live `Ledger` connections on the same WAL file in one process can crash the
interpreter at teardown. Call `run.close()` before building another runner on
the same `db_path`, or use it as a context manager:

```python
with Layer1Runner(cfg) as run:
    run.run()
```

## Next

Layer 2 reads `inject_ledger` and fits

```
outcome ~ a + sum_i beta_i * z_i + controls
```

where `z_i` is 1 when entry *i* survived masking. `beta_i` is the entry's
utility. Nothing in this package needs to change for that.
