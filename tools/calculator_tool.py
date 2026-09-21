NAME = "calculator"
DESCRIPTION = "Solves maths. Input example: 25 * 4"


def run(text):
    allowed = "0123456789+-*/(). "
    for character in text:
        if character not in allowed:
            return "ERROR: only numbers and + - * / ( ) are allowed"

    if "**" in text:
        return "ERROR: power is not allowed"

    try:
        answer = eval(text)
        return str(answer)
    except Exception as error:
        return "ERROR: " + str(error)