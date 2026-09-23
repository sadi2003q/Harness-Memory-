# beam_worker.py
# Put this file in your PROJECT_DIR (e.g. /kaggle/working/Harness-Memory-/)
# alongside brain.py, agent.py, tracer.py, tools.py, config.py, memory/.
#
# Why this file needs to exist at all:
# multiprocessing's "spawn" start method starts each child as a brand-new
# Python process. To know what function to run, the child re-imports the
# module the target function came from and looks it up by name. If the
# function was defined inline in a notebook cell, that module is just the
# Jupyter kernel's __main__ -- not a real file the child can re-import --
# so the lookup fails with:
#   AttributeError: Can't get attribute 'run_some_dialogues' on <module '__main__'...>
# Moving the function into a real .py file and importing it fixes this.

import os
import time
import signal
import shutil
import ast
import pandas as pd


QUESTION_TIMEOUT_SECONDS = 90
SETUP_TIMEOUT_SECONDS = 300


class TimeoutError_(Exception):
    pass


def _raise_timeout(signum, frame):
    raise TimeoutError_("timed out")


def run_with_timeout(fn, seconds, *args, **kwargs):
    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(seconds)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def run_some_dialogues(dialogue_ids, device, gpu_label, project_dir,
                        all_histories_path, all_questions_path,
                        how_many_questions_each, hf_token, results_file):

    import sys
    sys.path.insert(0, project_dir)

    from brain import Brain
    from agent import Agent
    from tracer import Tracer
    from tools import ALL_TOOLS
    from memory.memory_manager import MemoryManager
    from sentence_transformers import SentenceTransformer
    import config

    all_histories = pd.read_csv(all_histories_path)
    all_questions = pd.read_csv(all_questions_path)

    brain_instance = Brain(hf_token, device=device)

    def generate_fn(prompt):
        return brain_instance.think([{"role": "user", "content": prompt}])["text"]

    shared_embedder = SentenceTransformer(config.EMBEDDING_MODEL)

    def judge_abstention(agent_answer):
        answer_lower = agent_answer.lower()
        for marker in config.ABSTENTION_MARKERS:
            if marker in answer_lower:
                return True
        return False

    def judge_with_rubric(agent_answer, rubric_text):
        try:
            rubric_list = ast.literal_eval(rubric_text)
        except Exception:
            rubric_list = [rubric_text]
        answer_lower = agent_answer.lower()
        matches = 0
        for point in rubric_list:
            if str(point).lower()[:20] in answer_lower:
                matches = matches + 1
        needed = max(1, len(rubric_list) // 2)
        return matches >= needed

    def judge(question_type, agent_answer, rubric_text):
        if question_type == "abstention":
            return judge_abstention(agent_answer)
        return judge_with_rubric(agent_answer, rubric_text)

    base_folder = config.DATA_FOLDER

    for conversation_id in dialogue_ids:
        print("\n=== [" + gpu_label + "] Conversation:", conversation_id, "===", flush=True)

        conv_folder = base_folder + "/conv_" + str(conversation_id)
        if os.path.exists(conv_folder):
            shutil.rmtree(conv_folder)
        os.makedirs(conv_folder, exist_ok=True)

        memory = MemoryManager(embedder=shared_embedder)
        memory.historical.file_path = conv_folder + "/history.json"
        memory.semantic.file_path = conv_folder + "/facts.json"
        memory.episodic.file_path = conv_folder + "/episodes.json"

        tracer = Tracer()
        agent = Agent(brain_instance, memory, ALL_TOOLS, tracer)

        history_row = all_histories[all_histories["conversation_id"] == conversation_id].iloc[0]
        full_text = history_row["conversation"]

        setup_start = time.time()
        try:
            run_with_timeout(
                memory.add_to_historical, SETUP_TIMEOUT_SECONDS,
                full_text, generate_fn=generate_fn, auto_extract=True,
            )
        except TimeoutError_:
            print(f"  [{gpu_label}] SETUP TIMED OUT on conversation {conversation_id}, skipping it.", flush=True)
            continue
        except Exception as e:
            print(f"  [{gpu_label}] SETUP FAILED on conversation {conversation_id}: {e}, skipping it.", flush=True)
            continue
        setup_seconds = round(time.time() - setup_start, 2)
        print(f"  [{gpu_label}] setup took {setup_seconds}s", flush=True)

        this_conversation_questions = all_questions[all_questions["conversation_id"] == conversation_id]
        this_conversation_questions = this_conversation_questions.head(how_many_questions_each)

        for row_number in range(len(this_conversation_questions)):
            row = this_conversation_questions.iloc[row_number]
            question_text = row["question"]
            question_type = row["question_type"]
            rubric_text = row["rubric"]

            start_time = time.time()
            try:
                agent_answer = run_with_timeout(agent.ask, QUESTION_TIMEOUT_SECONDS, question_text)
            except TimeoutError_:
                agent_answer = "[TIMED OUT]"
                print(f"  [{gpu_label}][{question_type}] TIMED OUT after {QUESTION_TIMEOUT_SECONDS}s", flush=True)
            except Exception as e:
                agent_answer = f"[ERROR: {e}]"
                print(f"  [{gpu_label}][{question_type}] ERROR: {e}", flush=True)
            time_taken = time.time() - start_time

            is_correct = judge(question_type, agent_answer, rubric_text) if not agent_answer.startswith("[") else False
            print("  [" + gpu_label + "][" + question_type + "] correct =", is_correct, flush=True)

            one_result = pd.DataFrame([{
                "question_id": row["question_id"],
                "conversation_id": conversation_id,
                "q_type": question_type,
                "question": question_text,
                "gold_answer": row["gold_answer"],
                "agent_answer": agent_answer,
                "correct": is_correct,
                "seconds": round(time_taken, 2),
            }])

            file_already_exists = os.path.exists(results_file)
            one_result.to_csv(results_file, index=False, mode="a", header=not file_already_exists)

    print(f"[{gpu_label}] finished all assigned conversations.", flush=True)