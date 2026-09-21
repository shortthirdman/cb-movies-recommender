"""Natural-language intent parsing for the recommender chat page.

No language model is involved. This module turns a free-text message into a
small typed request that a recommendation backend can act on. It holds no
Streamlit imports and no knowledge of the catalogue, so every function here is
an ordinary pure function that can be tested directly.

One design decision needs explaining. Title extraction returns *candidates*
rather than a single string, because stripping a conversational prefix is
guesswork that regular expressions cannot finish:

    "movies like Toy Story"        -> the title is Toy Story
    "Like Water for Chocolate"     -> the title is the whole string

Both begin with a phrase that looks like a connector. The difference is not
grammatical, it is bibliographic: one of them is a film and the other is not,
and only the catalogue knows which. So the parser proposes an ordered list and
the caller tries each against the catalogue until one resolves. The same
mechanism covers "Ocean's 11", where the digits are part of the title rather
than a requested result count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

__all__ = ["DEFAULT_TOP_K", "MAX_TOP_K", "Intent", "Kind", "parse"]

# Bounds match the FastAPI schema in app/api.py (Field(ge=1, le=50)) so a
# request built here cannot be rejected by the service backend.
DEFAULT_TOP_K: Final = 5
MIN_TOP_K: Final = 1
MAX_TOP_K: Final = 50


class Kind(StrEnum):
    """What the user appears to be asking for."""

    RECOMMEND = "recommend"
    HELP = "help"
    GREETING = "greeting"


@dataclass(frozen=True, slots=True)
class Intent:
    """A parsed message.

    ``candidates`` is ordered by decreasing confidence. Callers should try each
    against the catalogue and use the first that resolves.
    """

    kind: Kind
    candidates: tuple[str, ...] = ()
    top_k: int = DEFAULT_TOP_K

    @property
    def best(self) -> str:
        """The most likely title, or an empty string when there is none."""
        return self.candidates[0] if self.candidates else ""


# --------------------------------------------------------------------------- #
# Result count
# --------------------------------------------------------------------------- #

_NUMBER_WORDS: Final[dict[str, int]] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20,
}

_NUM: Final = rf"\d{{1,2}}|{'|'.join(_NUMBER_WORDS)}"

# A bare number is not a count. "Ocean's 11" and "2012" are titles, and a count
# only counts when something in the sentence says it is one: either a counting
# noun after it, or a quantifying verb before it.
_COUNT_NOUNS: Final = (
    r"(?:movies?|films?|titles?|recommendations?|results?|suggestions?|picks?|options?)"
)
_COUNT_AFTER_RE: Final = re.compile(
    rf"\b({_NUM})\s+(?:more\s+|other\s+|different\s+|similar\s+|good\s+)*{_COUNT_NOUNS}\b",
    re.IGNORECASE,
)
_COUNT_BEFORE_RE: Final = re.compile(
    rf"\b(?:top|first|best)\s+({_NUM})\b", re.IGNORECASE
)
_COUNT_VERB_RE: Final = re.compile(
    rf"\b(?:show|give|list|send|get|find)\s+me\s+({_NUM})\b", re.IGNORECASE
)

_COUNT_PATTERNS: Final = (_COUNT_AFTER_RE, _COUNT_BEFORE_RE, _COUNT_VERB_RE)


def _to_int(token: str) -> int | None:
    token = token.strip().casefold()
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)


def _extract_count(text: str) -> tuple[str, int]:
    """Pull a requested result count out of ``text``.

    Returns the text with the matched count phrase blanked out, plus the count.
    Only the matched span is removed: other numbers stay put, because they are
    usually part of a title.
    """
    for pattern in _COUNT_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        value = _to_int(match.group(1))
        if value is None:
            continue
        cleaned = f"{text[: match.start()]} {text[match.end() :]}"
        return cleaned, max(MIN_TOP_K, min(MAX_TOP_K, value))
    return text, DEFAULT_TOP_K


# --------------------------------------------------------------------------- #
# Title
# --------------------------------------------------------------------------- #

# Phrases that mark the message as a request rather than a bare title. Stripping
# one is safe: no film in the catalogue starts with "recommend me".
_CUE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^(?:hi|hey|hello|yo)\b[\s,!.]*",
        r"^(?:ok|okay|so|well|and)\b[\s,]+",
        r"^(?:please|kindly)\s+",
        r"^(?:can|could|would|will)\s+you\s+(?:please\s+)?",
        r"^i(?:'d|\s+would)?\s+(?:like|want|need)\s+(?:to\s+(?:see|watch|get)\s+)?",
        r"^(?:what|which)\s+(?:should|can|could|do)\s+i\s+watch\b\s*",
        r"^(?:recommend|suggest|find|show|give|list|get|fetch)\s+(?:me\s+)?",
        r"^(?:some|a\s+few|a\s+couple\s+of|any|more|other)\s+",
        r"^(?:good|great|nice|similar)\s+",
        r"^(?:movies?|films?|titles?|something|anything|stuff)\b\s*",
        r"^(?:that\s+(?:are|is)|which\s+(?:are|is))\s+",
        r"^(?:based\s+on|inspired\s+by)\s+",
        r"^(?:if\s+i\s+)?(?:liked|loved|enjoyed|watched)\s+",
        r"^i\s+(?:just\s+)?(?:liked|loved|enjoyed|watched)\s+",
    )
)

# Connectors are only stripped once a cue has been consumed. Applied blindly
# they would decapitate "Like Water for Chocolate" and "To Kill a Mockingbird".
_CONNECTOR_RE: Final = re.compile(
    r"^(?:similar\s+to|comparable\s+to|close\s+to|in\s+the\s+vein\s+of|"
    r"along\s+the\s+lines\s+of|like|to|after|for)\s+",
    re.IGNORECASE,
)

# Each quote style closes only on its own kind. A single character class for all
# of them would end the span at the apostrophe in "I'll Be Seeing You".
# Written as escapes rather than literal curly quotes: the literals are visually
# indistinguishable from the straight forms in most editors, which is how the
# apostrophe bug got in.
_LDQ: Final = "\u201c"  # left double quotation mark
_RDQ: Final = "\u201d"  # right double quotation mark
_LSQ: Final = "\u2018"  # left single quotation mark
_RSQ: Final = "\u2019"  # right single quotation mark
_QUOTE_CHARS: Final = f"\"'{_LDQ}{_RDQ}{_LSQ}{_RSQ}"

_QUOTED_RE: Final = re.compile(
    r"\"([^\"]{2,})\""                          # "title"
    rf"|{_LDQ}([^{_RDQ}]{{2,}}){_RDQ}"          # curly double
    rf"|{_LSQ}([^{_RSQ}]{{2,}}){_RSQ}"          # curly single
    r"|'([^']{2,})'"                           # 'title', last so apostrophes lose
)
_TRAILING_RE: Final = re.compile(
    r"(?:[\s,]+(?:please|thanks|thank\s+you|pls|ty))?[\s?.!,]*$", re.IGNORECASE
)
# Includes the en and em dashes users paste when quoting a title.
_EN_DASH: Final = "\u2013"
_EM_DASH: Final = "\u2014"
_LEADING_JUNK_RE: Final = re.compile(rf"^[\s,:;\-{_EN_DASH}{_EM_DASH}]+")


def _tidy(text: str) -> str:
    """Collapse whitespace and shave conversational punctuation off both ends."""
    text = " ".join(text.split())
    text = _LEADING_JUNK_RE.sub("", text)
    return _TRAILING_RE.sub("", text).strip()


def _strip_guided(text: str) -> tuple[str, bool]:
    """Strip request boilerplate, treating connectors as connectors only in context.

    A connector is removed only once a cue has already been consumed. Applied
    unconditionally it would decapitate "Like Water for Chocolate"; applied
    after "movies", the word "like" in "movies like Toy Story" can only be a
    connector.

    The second return value reports whether the evidence was *strong*. A single
    cue word proves nothing, because "Get Out", "Show Boat" and "Something
    Wild" are all films whose first word doubles as a request verb or noun. A
    multi-word cue ("show me", "based on") or a connector is far harder to
    produce by accident, so only those outrank the text as typed.
    """
    seen_cue = False
    strong = False
    changed = True
    while changed and text:
        changed = False
        for pattern in _CUE_PATTERNS:
            if (candidate := pattern.sub("", text, count=1)) != text:
                consumed = text[: len(text) - len(candidate)].strip()
                strong = strong or " " in consumed
                text, changed, seen_cue = candidate.lstrip(), True, True
                break
        if changed:
            continue
        if seen_cue and (candidate := _CONNECTOR_RE.sub("", text, count=1)) != text:
            text, changed, strong = candidate.lstrip(), True, True
    return text, strong


def _strip_loose(text: str) -> str:
    """Strip a leading connector unconditionally.

    Only ever used as a last-resort candidate, never promoted ahead of the text
    as typed, so that "like Toy Story" can still reach *Toy Story* without
    putting "Water for Chocolate" ahead of the film that is actually called
    "Like Water for Chocolate".
    """
    while (candidate := _CONNECTOR_RE.sub("", text, count=1)) != text:
        text = candidate.lstrip()
    return text


def _candidates(text: str, *, counted: bool) -> tuple[str, ...]:
    """Ordered title guesses for ``text``, most likely first."""
    text = _tidy(text)
    if not text:
        return ()

    # An explicitly quoted span is unambiguous and outranks everything else.
    ordered: list[str] = []
    if (quoted := _QUOTED_RE.search(text)) is not None:
        inner = next((g for g in quoted.groups() if g is not None), "")
        ordered.append(_tidy(inner))

    bare = _tidy(text.strip(_QUOTE_CHARS))
    guided, strong = _strip_guided(text)
    guided = _tidy(guided)
    loose = _tidy(_strip_loose(bare))

    # An explicit result count ("3 movies like ...") is itself proof of a request.
    lead = [guided, bare] if strong or counted else [bare, guided]
    ordered.extend([*lead, loose])

    seen: set[str] = set()
    unique: list[str] = []
    for value in ordered:
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            unique.append(value)
    return tuple(unique)


# --------------------------------------------------------------------------- #
# Conversational intents
# --------------------------------------------------------------------------- #

_GREETING_RE: Final = re.compile(
    r"^(?:hi|hey|hello|yo|good\s+(?:morning|afternoon|evening)|greetings)\b[\s,!.]*$",
    re.IGNORECASE,
)
_HELP_RE: Final = re.compile(
    r"^(?:help|\?|what\s+can\s+you\s+do|how\s+(?:do|does)\s+(?:i|this|it)\s+work|"
    r"how\s+to\s+use|usage|commands?)\b[\s?.!]*$",
    re.IGNORECASE,
)


def parse(message: str) -> Intent:
    """Parse a chat message into an :class:`Intent`.

    >>> parse("recommend 3 movies like Toy Story").candidates[0]
    'Toy Story'
    >>> parse("recommend 3 movies like Toy Story").top_k
    3
    >>> parse("Ocean's 11").top_k
    5
    >>> parse("Like Water for Chocolate").candidates[0]
    'Like Water for Chocolate'
    """
    text = " ".join(message.split())
    if not text:
        return Intent(Kind.HELP)
    if _GREETING_RE.match(text):
        return Intent(Kind.GREETING)
    if _HELP_RE.match(text):
        return Intent(Kind.HELP)

    remainder, top_k = _extract_count(text)
    candidates = _candidates(remainder, counted=top_k != DEFAULT_TOP_K)
    if not candidates:
        return Intent(Kind.HELP, top_k=top_k)
    return Intent(Kind.RECOMMEND, candidates=candidates, top_k=top_k)
