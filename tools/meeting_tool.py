import os
import json
import config

NAME = "schedule_meeting"
DESCRIPTION = "Saves a meeting. Input format: title | time  (example: Team sync | Monday 10am)"
FILE_PATH = os.path.join(config.DATA_FOLDER, "meetings.json")


def run(text):
    if "|" not in text:
        return "ERROR: use the format  title | time"

    parts = text.split("|")
    title = parts[0].strip()
    meeting_time = parts[1].strip()

    os.makedirs(config.DATA_FOLDER, exist_ok=True)

    meetings = []
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "r") as file:
            meetings = json.load(file)

    meetings.append({"title": title, "time": meeting_time})

    with open(FILE_PATH, "w") as file:
        json.dump(meetings, file, indent=2)

    return "Meeting saved: " + title + " at " + meeting_time