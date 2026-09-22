import config
from memory.procedural_memory import ProceduralMemory
from memory.semantic_memory import SemanticMemory
from memory.episodic_memory import EpisodicMemory
from memory.historical_memory import HistoricalMemory
from utilities.fact_extractor import extract_facts_long


class MemoryManager:
    """One door to all four memories."""

    def __init__(self):
        self.procedural = ProceduralMemory()
        self.semantic = SemanticMemory()
        self.episodic = EpisodicMemory()
        self.historical = HistoricalMemory()

    def get_memory_text(self, question, use_historical=False):
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

        if use_historical:
            hist = self.historical.search(question, config.HISTORICAL_TOP_K)
            if len(hist) > 0:
                text += "RELEVANT PAST CONTEXT:\n"
                for chunk in hist:
                    text += "- " + chunk + "\n"

        return text

    def save_event(self, question, answer, tools_used, tokens):
        self.episodic.add_event(question, answer, tools_used, tokens)
        if config.STORE_LIVE_CONVERSATION_IN_HISTORICAL:
            turn_text = "User: " + question + "\nAssistant: " + answer
            self.historical.add_chunk(turn_text)

    def add_to_historical(self, text, generate_fn=None, auto_extract=True):
        """Intentionally add anything: a full conversation, a lecture, a paragraph.
        If generate_fn is given, facts are pulled out and added to semantic memory too."""
        self.historical.add_text(text)

        if auto_extract and generate_fn is not None:
            facts = extract_facts_long(text, generate_fn)
            for fact in facts:
                self.semantic.add_fact(fact)

    def consolidate(self, brain):
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