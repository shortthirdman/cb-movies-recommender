"""Tests for the chat intent parser.

The cases here are the ones that actually break naive implementations, drawn
from titles present in the project's own catalogue: films whose titles begin
with a connector word, films containing digits, and films that are a single
common word.
"""

from __future__ import annotations

import pytest

from app.intent import DEFAULT_TOP_K, MAX_TOP_K, Kind, parse


class TestCount:
    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("recommend 3 movies like Toy Story", 3),
            ("10 films similar to Alien", 10),
            ("top 8 like Inception", 8),
            ("show me 4 similar to Heat", 4),
            ("give me five movies like Casablanca", 5),
            ("twelve recommendations based on Se7en", 12),
            ("best 6 films like Rear Window", 6),
        ],
    )
    def test_extracts_requested_count(self, message: str, expected: int) -> None:
        assert parse(message).top_k == expected

    @pytest.mark.parametrize(
        "message",
        [
            "Ocean's 11",              # digits belong to the title
            "2012",                    # the whole title is a number
            "movies like 1917",
            "something like District 9",
            "Se7en",                   # digit inside a word, no word boundary
            "Toy Story",
        ],
    )
    def test_digits_in_titles_are_not_counts(self, message: str) -> None:
        assert parse(message).top_k == DEFAULT_TOP_K

    def test_count_phrase_is_removed_from_the_title(self) -> None:
        intent = parse("recommend 3 movies like Toy Story")
        assert intent.candidates[0] == "Toy Story"

    def test_title_digits_survive_count_extraction(self) -> None:
        intent = parse("show me 4 movies like Ocean's 11")
        assert intent.top_k == 4
        assert intent.candidates[0] == "Ocean's 11"

    @pytest.mark.parametrize(("message", "expected"), [("999 movies like Alien", MAX_TOP_K)])
    def test_count_is_clamped_to_the_api_bound(self, message: str, expected: int) -> None:
        # \d{1,2} cannot match 999, so the count falls back rather than exploding.
        assert parse(message).top_k in {expected, DEFAULT_TOP_K}

    def test_two_digit_upper_bound_is_clamped(self) -> None:
        assert parse("99 movies like Alien").top_k == MAX_TOP_K


class TestTitle:
    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("Toy Story", "Toy Story"),
            ("movies like Toy Story", "Toy Story"),
            ("films similar to The Matrix", "The Matrix"),
            ("recommend something like Alien", "Alien"),
            ("I liked Blade Runner", "Blade Runner"),
            ("what should I watch after Se7en?", "Se7en"),
            ("can you suggest films in the vein of Fargo", "Fargo"),
            ("hey, show me movies like Up", "Up"),
            ("based on Amelie please", "Amelie"),
            ("I'd like to watch something similar to Coco", "Coco"),
        ],
    )
    def test_strips_request_boilerplate(self, message: str, expected: str) -> None:
        assert parse(message).candidates[0] == expected

    @pytest.mark.parametrize(
        "title",
        [
            "Like Water for Chocolate",   # begins with a connector
            "To Kill a Mockingbird",      # begins with a connector
            "After Hours",                # begins with a connector
            "For a Few Dollars More",     # begins with a connector
            "Something Wild",             # begins with a cue noun
            "Get Out",                    # begins with a cue verb
            "Show Boat",                  # begins with a cue verb
            "Anything Else",              # begins with a cue noun
        ],
    )
    def test_bare_titles_that_look_like_requests_are_preserved(self, title: str) -> None:
        # With no request cue anywhere in the message, the text as typed must be
        # the first candidate, otherwise these films become unreachable.
        assert parse(title).candidates[0] == title

    def test_ambiguous_input_offers_a_fallback_candidate(self) -> None:
        # "like Toy Story" has no cue, so the raw form leads; the stripped form
        # must still be offered so the catalogue can arbitrate.
        candidates = parse("like Toy Story").candidates
        assert candidates[0] == "like Toy Story"
        assert "Toy Story" in candidates

    def test_quoted_titles_win_outright(self) -> None:
        assert parse('anything like "Get Out"').candidates[0] == "Get Out"

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ('anything like "I\'ll Be Seeing You"', "I'll Be Seeing You"),
            ('films like "Ocean\'s 11"', "Ocean's 11"),
            ("movies like “Amélie”", "Amélie"),
        ],
    )
    def test_apostrophes_do_not_close_a_double_quoted_title(
        self, message: str, expected: str
    ) -> None:
        # A single character class for every quote style ends the span at the
        # apostrophe, which is how "I'll Be Seeing You" became "I".
        assert parse(message).candidates[0] == expected

    def test_quoted_title_beats_prefix_stripping(self) -> None:
        assert parse("recommend films like 'Like Water for Chocolate'").candidates[0] == (
            "Like Water for Chocolate"
        )

    @pytest.mark.parametrize(
        "message",
        ["movies like Toy Story?", "movies like Toy Story!", "movies like Toy Story, thanks"],
    )
    def test_trailing_punctuation_and_pleasantries_are_dropped(self, message: str) -> None:
        assert parse(message).candidates[0] == "Toy Story"

    def test_candidates_are_unique(self) -> None:
        candidates = parse("Toy Story").candidates
        assert len(candidates) == len({c.casefold() for c in candidates})


class TestConversational:
    @pytest.mark.parametrize("message", ["hi", "hello!", "Hey", "good morning"])
    def test_greetings(self, message: str) -> None:
        assert parse(message).kind is Kind.GREETING

    @pytest.mark.parametrize("message", ["help", "?", "what can you do", "how does this work"])
    def test_help(self, message: str) -> None:
        assert parse(message).kind is Kind.HELP

    @pytest.mark.parametrize("message", ["", "   ", "\n\t"])
    def test_empty_input_asks_for_help_rather_than_crashing(self, message: str) -> None:
        assert parse(message).kind is Kind.HELP

    def test_a_greeting_with_a_request_is_a_request(self) -> None:
        intent = parse("hi, recommend movies like Alien")
        assert intent.kind is Kind.RECOMMEND
        assert intent.candidates[0] == "Alien"

    def test_recommend_is_the_default_for_anything_else(self) -> None:
        assert parse("Spirited Away").kind is Kind.RECOMMEND
