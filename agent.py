import time
import config

SYSTEM_PROMPT = """You are a helpful assistant.

You can use these tools:
{tools}

To use a tool, reply in EXACTLY this format and nothing else:
TOOL: tool_name
INPUT: what to give the tool

When you know the final answer, reply in this format:
ANSWER: your answer

{memory}"""


class Agent:
    def __init__(self, brain, memory, tools, tracer):
        self.brain = brain
        self.memory = memory
        self.tools = tools
        self.tracer = tracer
        self.chat_history = []

    def make_system_prompt(self, memory_text):
        tools_text = ""
        for name in self.tools:
            tools_text += "- " + name + ": " + self.tools[name]["description"] + "\n"
        return SYSTEM_PROMPT.format(tools=tools_text, memory=memory_text)

    def read_reply(self, reply):
        # Returns ("tool", tool_name, tool_input) or ("answer", answer_text, "")
        tool_name = ""
        tool_input = ""

        for line in reply.split("\n"):
            line = line.strip()
            if line.startswith("TOOL:"):
                tool_name = line.replace("TOOL:", "").strip()
            elif line.startswith("INPUT:"):
                tool_input = line.replace("INPUT:", "").strip()

        if tool_name != "":
            return "tool", tool_name, tool_input

        if "ANSWER:" in reply:
            answer = reply.split("ANSWER:", 1)[1].strip()
            return "answer", answer, ""

        return "answer", reply.strip(), ""

    def run_tool(self, name, tool_input):
        if name not in self.tools:
            return "ERROR: tool '" + name + "' does not exist"
        try:
            return self.tools[name]["run"](tool_input)
        except Exception as error:
            return "ERROR: " + str(error)

    def _looks_like_abstention(self, answer):
        # Checks if the answer sounds like "I don't know" / "I can't find that"
        lower = answer.lower()
        return any(marker in lower for marker in config.ABSTENTION_MARKERS)

    def _run_loop(self, question, memory_text):
        """One full think -> maybe use tool -> think again cycle.
        Returns (final_answer, tools_used, total_tokens, tool_error)."""
        messages = [{"role": "system", "content": self.make_system_prompt(memory_text)}]
        messages = messages + self.chat_history
        messages.append({"role": "user", "content": question})

        tools_used = []
        total_tokens = 0
        final_answer = ""
        tool_error = False

        for step in range(config.MAX_LOOP_STEPS):
            result = self.brain.think(messages)
            total_tokens += result["input_tokens"] + result["output_tokens"]

            kind, value, tool_input = self.read_reply(result["text"])

            if kind == "answer":
                final_answer = value
                break

            tool_result = self.run_tool(value, tool_input)
            tools_used.append(value)
            if tool_result.startswith("ERROR"):
                tool_error = True

            messages.append({"role": "assistant", "content": result["text"]})
            messages.append({
                "role": "user",
                "content": "TOOL RESULT: " + tool_result + "\nNow reply with ANSWER: or use another tool.",
            })

        # Guardrail: the loop ended without an answer
        if final_answer == "":
            final_answer = "Sorry, I could not finish in time."

        return final_answer, tools_used, total_tokens, tool_error

    def ask(self, question, allow_historical_fallback=True):
        start_time = time.time()

        # 1) Normal pass: procedural + semantic + episodic memory only (cheap, fast)
        memory_text = self.memory.get_memory_text(question, use_historical=False)
        final_answer, tools_used, total_tokens, tool_error = self._run_loop(question, memory_text)

        # 2) Only if that answer looks like "I don't know", try ONE more time
        #    with the Historical memory included (slower, only used when needed)
        used_historical = False
        if allow_historical_fallback and self._looks_like_abstention(final_answer):
            memory_text = self.memory.get_memory_text(question, use_historical=True)
            fallback_answer, fb_tools, fb_tokens, fb_error = self._run_loop(question, memory_text)

            total_tokens += fb_tokens
            tools_used += fb_tools
            tool_error = tool_error or fb_error
            final_answer = fallback_answer
            used_historical = True

        # 3) Save chat history, memory, and trace
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": final_answer})
        self.chat_history = self.chat_history[-config.CHAT_HISTORY_LIMIT:]

        seconds = time.time() - start_time
        self.memory.save_event(question, final_answer, tools_used, total_tokens)
        self.tracer.record(question, final_answer, tools_used, seconds, total_tokens, tool_error)

        return final_answer