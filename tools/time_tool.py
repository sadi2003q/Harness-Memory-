from datetime import datetime

NAME = "current_time"
DESCRIPTION = "Gives today's date and time. Input can be empty."


def run(text):
    now = datetime.now()
    return now.strftime("%A, %d %B %Y, %H:%M")