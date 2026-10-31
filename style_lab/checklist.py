"""The style checklist: rule-based checks on an output, as pure functions of (input, output).

Each check reports pass/fail and the evidence it found, so a learner can see why.
Never imports the model.
"""
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    evidence: str


# ---- Professional email ---------------------------------------------------------------

SALUTATION = re.compile(r"^(dear|hi|hello|hey|greetings|good (morning|afternoon|evening))\b", re.IGNORECASE)
SIGN_OFF = re.compile(
    r"^(best( regards| wishes)?|kind regards|warm regards|regards|sincerely|yours sincerely|"
    r"thanks|thank you|many thanks|cheers|all the best)\s*,?$",
    re.IGNORECASE,
)
POINT_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
NAME = "{name}"


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _labelled_line(lines: list[str], label: str) -> str | None:
    return next((line for line in lines if line.lower().lstrip("*# ").startswith(label.lower())), None)


def _has_label(output: str, label: str) -> Check:
    line = _labelled_line(_lines(output), label)
    name = f"Has {label} line" if label == "Subject:" else f"Has {label} section"
    if line:
        return Check(name, True, f'Found: "{line}"')
    return Check(name, False, f"No line starts with `{label}`")


def _greeting(output: str) -> Check:
    lines = _lines(output)
    subject = _labelled_line(lines, "Subject:")
    body = lines[lines.index(subject) + 1:] if subject else lines
    if not body:
        return Check("Greeting", False, "The email has no body")
    opening = body[0]
    salutation = re.sub(r"^\W*body:\s*", "", opening, flags=re.IGNORECASE)
    if not salutation and len(body) > 1:  # "Body:" alone on its line
        opening = salutation = body[1]
    if not SALUTATION.match(salutation):
        return Check("Greeting", False, f'Body opens without a salutation: "{opening}"')
    uses_name = "uses {name}" if NAME in salutation else "does not use {name}"
    return Check("Greeting", True, f'"{opening}" ({uses_name})')


def _sign_off(output: str) -> Check:
    lines = _lines(output)
    name = "Sign-off with {name}"
    if not lines:
        return Check(name, False, "The output is empty")
    last = lines[-1]
    if last == NAME and len(lines) > 1 and SIGN_OFF.match(lines[-2]):
        return Check(name, True, f'Ends with "{lines[-2]}" then "{NAME}"')
    one_line = re.match(r"^(.*?),?\s*\{name\}$", last)
    if one_line and SIGN_OFF.match(one_line.group(1).strip()):
        return Check(name, True, f'Ends with "{last}"')
    tail = " / ".join(lines[-2:])
    return Check(name, False, f'Expected a sign-off line then "{NAME}"; last lines are: "{tail}"')


def content_points(notes: str) -> list[str]:
    """The points an email must keep: bullet or numbered lines, else every non-empty line."""
    lines = _lines(notes)
    bullets = [POINT_MARKER.sub("", line) for line in lines if POINT_MARKER.match(line)]
    return bullets or lines


def normalise(text: str) -> str:
    """Lowercase, punctuation turned into spaces, whitespace collapsed."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _content_kept(notes: str, output: str) -> Check:
    points = [normalise(p) for p in content_points(notes)]
    points = [p for p in points if p]
    if not points:
        return Check("Content kept", True, "The input has no points to keep")
    haystack = f" {normalise(output)} "
    missing = [p for p in points if f" {p} " not in haystack]
    kept = len(points) - len(missing)
    if missing:
        listed = "; ".join(f'"{p}"' for p in missing)
        return Check("Content kept", False, f"{kept}/{len(points)} points kept. Missing: {listed}")
    return Check("Content kept", True, f"{kept}/{len(points)} points kept")


def email_checklist(notes: str, output: str) -> list[Check]:
    return [
        _has_label(output, "Subject:"),
        _has_label(output, "Body:"),
        _greeting(output),
        _sign_off(output),
        _content_kept(notes, output),
    ]


# ---- Social media ---------------------------------------------------------------------

_EMOJI_CHAR = (
    "\U0001F000-\U0001FAFF"  # pictographs, emoticons, transport, symbols & pictographs ext.
    "☀-➿"  # misc symbols, dingbats
    "⬀-⯿"  # stars, arrows
    "⌀-⏿"  # technical (⌛, ⏰)
)
# One emoji = a pictograph plus any modifiers, variation selectors or ZWJ-joined parts.
EMOJI = re.compile(
    f"[{_EMOJI_CHAR}](?:️|[\U0001F3FB-\U0001F3FF]|‍[{_EMOJI_CHAR}])*"
)
HASHTAG = re.compile(r"(?<![\w#])#[^\W\d]\w*")


def social_media_checklist(text: str, output: str) -> list[Check]:
    emoji = EMOJI.findall(output)
    hashtags = HASHTAG.findall(output)
    return [
        Check("Emoji", bool(emoji), f"{len(emoji)} emoji" + (f": {' '.join(emoji)}" if emoji else "")),
        Check("Hashtags", bool(hashtags), f"{len(hashtags)} hashtags" + (f": {' '.join(hashtags)}" if hashtags else "")),
    ]
