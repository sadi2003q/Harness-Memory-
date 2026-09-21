import config


class Tracer:
    """Trace -> Evaluate -> Diagnose (Fix is done by you, using the advice)."""

    def __init__(self):
        self.traces = []

    def record(self, question, answer, tools_used, seconds, tokens, tool_error):
        trace = {
            "question": question,
            "answer": answer,
            "tools_used": tools_used,
            "seconds": round(seconds, 2),
            "tokens": tokens,
            "tool_error": tool_error,
        }
        trace["problems"] = self.evaluate(trace)
        trace["diagnosis"] = self.diagnose(trace["problems"])
        self.traces.append(trace)
        return trace

    def evaluate(self, trace):
        problems = []
        if trace["tool_error"]:
            problems.append("tool_error")
        if trace["tokens"] > config.MAX_TOKENS_PER_ANSWER:
            problems.append("too_many_tokens")
        if trace["seconds"] > config.MAX_RESPONSE_SECONDS:
            problems.append("too_slow")
        if len(trace["answer"]) < 2:
            problems.append("empty_answer")
        if trace["answer"].startswith("Sorry, I could not finish"):
            problems.append("loop_limit_reached")
        return problems

    def diagnose(self, problems):
        advice = {
            "tool_error": "A tool failed. Check the tool's input format in its DESCRIPTION.",
            "too_many_tokens": "Context is too big. Shorten memory or chat history.",
            "too_slow": "Model is slow. Use a smaller model or fewer loop steps.",
            "empty_answer": "Model gave no answer. Check the system prompt format.",
            "loop_limit_reached": "Model never gave ANSWER:. Make the prompt format clearer.",
        }
        return [advice[problem] for problem in problems]

    def show(self):
        for number, trace in enumerate(self.traces, start=1):
            print("--- Trace", number, "---")
            print("Question :", trace["question"])
            print("Answer   :", trace["answer"])
            print("Tools    :", trace["tools_used"])
            print("Time     :", trace["seconds"], "sec | Tokens:", trace["tokens"])
            if len(trace["problems"]) == 0:
                print("Status   : OK")
            else:
                print("Problems :", trace["problems"])
                print("Fix ideas:", trace["diagnosis"])