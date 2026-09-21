import config
from memory.procedural_memory import ProceduralMemory
from memory.semantic_memory import SemanticMemory
from memory.episodic_memory import EpisodicMemory


class MemoryManager:
    """One door to all three memories. Your memory research goes here."""

    def __init__(self):
        self.procedural = ProceduralMemory()
        self.semantic = SemanticMemory()
        self.episodic = EpisodicMemory()

    def get_memory_text(self, question):
        # Collects memory that will be placed into the agent's context
        text = "HOW TO ACT:\n" + self.procedural.get_instructions() + "\n"

        facts = self.semantic.search(question, config.SEMANTIC_TOP_K)
        if len(facts) > 0:
            text += "KNOWN FACTS:\n"
            for fact in facts:
                text += "- " + fact + "\n"

        recent = self.episodic.get_recent(config.EPISODIC_RECENT)
        if len(recent) > 0:
            text += "RECENT EVENTS:\n"
            for event in recent:
                text += "- User asked: " + event["question"] + " | Answer: " + event["answer"] + "\n"

        return text

    def save_event(self, question, answer, tools_used, tokens):
        self.episodic.add_event(question, answer, tools_used, tokens)

    def consolidate(self, brain):
        # Episodic -> LLM summary -> Semantic (the bottom part of your diagram)
        recent = self.episodic.get_recent(5)
        if len(recent) == 0:
            print("Nothing to summarise yet.")
            return None

        events_text = ""
        for event in recent:
            events_text += "User asked: " + event["question"] + " | Answer: " + event["answer"] + "\n"

        messages = [
            {"role": "system", "content": "Summarise these events into ONE short factual sentence."},
            {"role": "user", "content": events_text},
        ]
        summary = brain.think(messages)["text"]
        self.semantic.add_fact(summary)
        return summary