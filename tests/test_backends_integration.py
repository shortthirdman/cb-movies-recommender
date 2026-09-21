"""Integration tests against the real model bundle and merged dataset.

These are the tests that decide whether the app works. The unit tests in
``test_intent.py`` prove the parser does what it was designed to do; these
prove the design was right, by taking real catalogue titles, wrapping them in
the phrasings a user would actually type, and checking the film comes back.

Skipped when the artefacts are absent so the suite still runs on a checkout
without them.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from app.backends import (
    UNKNOWN_COUNTRY,
    LocalBackend,
    TitleNotFoundError,
    build_country_index,
    load_recommender,
    suggest_first_match,
)
from app.intent import parse

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "baseline_v0" / "baseline_v0.joblib"
MERGED = ROOT / "datasets" / "processed" / "movies_merged.csv"

pytestmark = pytest.mark.skipif(
    not MODEL.is_file() or not MERGED.is_file(),
    reason="model bundle or merged dataset not present",
)

# Phrasings a user might plausibly type, each with the title substituted in.
PHRASINGS = (
    "{t}",
    "movies like {t}",
    "films similar to {t}",
    "recommend something like {t}",
    "I liked {t}",
    "what should I watch after {t}?",
    "hey, show me 3 films like {t}",
    'anything like "{t}"',
    "give me 4 recommendations based on {t}",
)


@pytest.fixture(scope="module")
def backend() -> LocalBackend:
    return LocalBackend(
        recommender=load_recommender(MODEL, "combined_tfidf"),
        countries=build_country_index(MERGED),
    )


class TestCountryIndex:
    def test_index_is_populated(self) -> None:
        index = build_country_index(MERGED)
        assert len(index) > 50_000

    def test_missing_file_yields_an_empty_index(self, tmp_path: Path) -> None:
        assert build_country_index(tmp_path / "absent.csv") == {}

    def test_known_film_resolves_to_its_country(self) -> None:
        index = build_country_index(MERGED)
        assert index[("toy story", 1995)] == "United States of America"

    def test_multi_country_films_keep_every_country(self) -> None:
        index = build_country_index(MERGED)
        assert index[("sense and sensibility", 1995)] == (
            "United Kingdom, United States of America"
        )


class TestEndToEnd:
    def test_every_phrasing_finds_the_same_film(self, backend: LocalBackend) -> None:
        for phrasing in PHRASINGS:
            intent = parse(phrasing.format(t="Toy Story"))
            result = suggest_first_match(backend, intent.candidates, intent.top_k)
            assert result.seed == "Toy Story", phrasing

    def test_requested_count_is_honoured(self, backend: LocalBackend) -> None:
        intent = parse("show me 3 films like Toy Story")
        result = suggest_first_match(backend, intent.candidates, intent.top_k)
        assert len(result.suggestions) == 3

    def test_every_field_the_brief_asks_for_is_populated(self, backend: LocalBackend) -> None:
        intent = parse("movies like Toy Story")
        result = suggest_first_match(backend, intent.candidates, intent.top_k)
        assert result.suggestions
        for suggestion in result.suggestions:
            assert suggestion.title
            assert 0.0 <= suggestion.score <= 1.0
            assert suggestion.country
            assert suggestion.release_year is None or 1870 < suggestion.release_year < 2100

    def test_results_are_ordered_by_descending_score(self, backend: LocalBackend) -> None:
        intent = parse("20 films like Toy Story")
        result = suggest_first_match(backend, intent.candidates, intent.top_k)
        scores = [s.score for s in result.suggestions]
        assert scores == sorted(scores, reverse=True)

    def test_the_seed_is_never_recommended_back(self, backend: LocalBackend) -> None:
        intent = parse("10 films like Toy Story")
        result = suggest_first_match(backend, intent.candidates, intent.top_k)
        assert all(s.title != result.seed for s in result.suggestions)

    def test_unknown_title_raises(self, backend: LocalBackend) -> None:
        intent = parse("movies like Zzzqx Nonexistent Film")
        with pytest.raises(TitleNotFoundError):
            suggest_first_match(backend, intent.candidates, intent.top_k)

    @pytest.mark.parametrize("seed", [11, 23, 47])
    def test_random_real_titles_survive_every_phrasing(
        self, backend: LocalBackend, seed: int
    ) -> None:
        """The parser must not mangle ordinary catalogue titles.

        Titles are sampled from the catalogue itself, so this exercises the
        awkward real ones - digits, leading connectors, single common words -
        rather than examples chosen to pass.
        """
        rng = random.Random(seed)
        titles = [t for t in rng.sample(backend.recommender.titles, 60) if t.strip()]

        failures: list[tuple[str, str]] = []
        for title in titles:
            for phrasing in PHRASINGS:
                intent = parse(phrasing.format(t=title))
                try:
                    suggest_first_match(backend, intent.candidates, intent.top_k)
                except TitleNotFoundError:
                    failures.append((title, phrasing))

        attempts = len(titles) * len(PHRASINGS)
        # Some catalogue titles are genuinely unresolvable through a phrasing -
        # a film literally called "Up" inside "show me 3 films like Up" is
        # ambiguous by construction. A small failure rate is expected; a large
        # one means the parser is eating titles.
        assert len(failures) / attempts < 0.05, failures[:15]


class TestCountryCoverage:
    def test_most_recommendations_carry_a_country(self, backend: LocalBackend) -> None:
        rng = random.Random(7)
        titles = rng.sample(backend.recommender.titles, 40)
        total = known = 0
        for title in titles:
            try:
                result = backend.suggest(title, 10)
            except TitleNotFoundError:
                continue
            for suggestion in result.suggestions:
                total += 1
                known += suggestion.country != UNKNOWN_COUNTRY
        assert total > 0
        # 92.7% of the evaluation catalogue carries a country in the merged file.
        assert known / total > 0.80, f"{known}/{total}"
