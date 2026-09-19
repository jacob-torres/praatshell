"""Number-to-speech formatting.

Units are spelled out because a screen reader says "0.184 s" as "zero point one
eight four s".
"""


def parse_time(text):
    """Accept '0.12', '120ms', '120 ms'. Returns seconds."""
    t = text.strip().lower().replace(" ", "")
    if t.endswith("ms"):
        return float(t[:-2]) / 1000.0
    if t.endswith("s"):
        t = t[:-1]
    return float(t)


def secs(t):
    return f"{t:.3f} seconds"


def ms(dt):
    return f"{dt * 1000:.0f} milliseconds"


def duration(dt):
    """Milliseconds below a second, seconds above."""
    return ms(dt) if dt < 1.0 else f"{dt:.3f} seconds"


def hz(f):
    if f is None:
        return "not measurable"
    if f >= 1000:
        return f"{f / 1000:.2f} kilohertz"
    return f"{f:.0f} hertz"


def hz_plain(f):
    return "not measurable" if f is None else f"{f:.0f} hertz"


def db(x):
    return "not measurable" if x is None else f"{x:.1f} decibels"


def pct(x):
    return f"{x * 100:.1f} percent"


def amp(x):
    return f"{x:.3f} of full scale"


def article(phrase):
    return ("an " if phrase[:1].lower() in "aeiou" else "a ") + phrase


def join(items):
    """'a, b and c' - reads better aloud than 'a, b, c'."""
    items = list(items)
    if not items:
        return "none"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]
