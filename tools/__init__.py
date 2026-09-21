# To add a new tool: create a file, then add it to the list below.
from tools import calculator_tool, time_tool, meeting_tool

ALL_TOOLS = {}

for tool_file in [calculator_tool, time_tool, meeting_tool]:
    ALL_TOOLS[tool_file.NAME] = {
        "run": tool_file.run,
        "description": tool_file.DESCRIPTION,
    }