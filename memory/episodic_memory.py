import os
import json
from datetime import datetime
import config


class EpisodicMemory:
    """WHAT happened: a log of past events."""

    def __init__(self):
        os.makedirs(config.DATA_FOLDER, exist_ok=True)
        self.file_path = os.path.join(config.DATA_FOLDER, "episodes.json")
        self.events = []

        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                self.events = json.load(file)

    def add_event(self, question, answer, tools_used, tokens):
        event = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "question": question,
            "answer": answer,
            "tools_used": tools_used,
            "tokens": tokens,
        }
        self.events.append(event)
        self.save()

    def get_recent(self, count):
        return self.events[-count:]

    def save(self):
        with open(self.file_path, "w") as file:
            json.dump(self.events, file, indent=2)