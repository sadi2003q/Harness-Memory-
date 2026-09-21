# Small Language Model AI Agent

A beginner-friendly AI agent with three parts: a **brain** (a small language model from Hugging Face), **tools**, and **memory**. Built to run on Kaggle or Google Colab.

## Project structure

```
project/
├── config.py            # all settings in one place
├── brain.py             # loads the model + tokenizer, generates replies
├── agent.py             # the main loop: memory -> think -> tool -> answer
├── tracer.py            # logs each run and flags problems
├── main.ipynb           # where you use the agent (no function definitions)
├── tools/               # one file per tool
│   ├── __init__.py      # registers all tools in ALL_TOOLS
│   ├── calculator_tool.py
│   ├── time_tool.py
│   └── meeting_tool.py
└── memory/
    ├── __init__.py      # empty
    ├── procedural_memory.py   # how to act (skills.md)
    ├── semantic_memory.py     # facts, searched by meaning
    ├── episodic_memory.py     # log of past events
    └── memory_manager.py      # one door to all three memories
```

## How it works

1. You ask a question with `agent.ask("...")`.
2. `MemoryManager` collects instructions, related facts and recent events.
3. The agent builds the context (system prompt + memory + chat history + question).
4. Loop (max `MAX_LOOP_STEPS`): the model either replies `TOOL: name / INPUT: ...` or `ANSWER: ...`.
5. The final answer is returned. The event is saved to memory and the run is recorded by the tracer.

## Setup

```python
!pip install -q -U transformers accelerate sentence-transformers
```

Then run the cells in `main.ipynb`. It asks for your Hugging Face token. Default model: `Qwen/Qwen2.5-1.5B-Instruct` (change it in `config.py`).

## Memory types

| Type | File | Stores |
|---|---|---|
| Procedural | `skills.md` | Instructions on how to act |
| Semantic | `facts.json` | Facts + embeddings, searched by similarity |
| Episodic | `episodes.json` | Question, answer, tools used, tokens |

All memory files are saved in `memory_data/` (created automatically).

`memory.consolidate(brain)` summarises recent events into one fact and saves it to semantic memory.

## Tracer

After each question the tracer records time, tokens, tools used and errors. It flags problems (`tool_error`, `too_slow`, `too_many_tokens`, `empty_answer`, `loop_limit_reached`) and prints fix ideas. View with `tracer.show()`.

## Add a new tool

1. Create `tools/my_tool.py` with `NAME`, `DESCRIPTION` and a `run(text)` function that returns a string.
2. Import it in `tools/__init__.py` and add it to the list.

## Settings

Change the model, loop limit, memory sizes and tracer limits in `config.py`.

## Known limits

- Small models sometimes break the `TOOL:` / `ANSWER:` format, so check the tracer output.
- The Fix step of the tracer is manual advice for now.
- Semantic memory only appends facts (no update or delete yet).