import os
import config

DEFAULT_TEXT = """- Be short and clear.
- Use a tool for maths, date/time, or saving a meeting.
- If you do not know something, say you do not know.
"""


class ProceduralMemory:
    """HOW to act: skills and instructions (like skill.md)."""

    def __init__(self):
        os.makedirs(config.DATA_FOLDER, exist_ok=True)
        self.file_path = os.path.join(config.DATA_FOLDER, "skills.md")

        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as file:
                file.write(DEFAULT_TEXT)

    def get_instructions(self):
        with open(self.file_path, "r") as file:
            return file.read()