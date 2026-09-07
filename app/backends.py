"""Recommendation backends for the Streamlit app.

Two ways of answering the same question, selectable in the sidebar:

``LocalBackend``
    Deserialises ``models/baseline_v0/baseline_v0.joblib`` and scores in
    process, reusing the ``Recommender`` from :mod:`app.api` rather than
    reimplementing title resolution. One process, no network.

``ServiceBackend``
    Posts to the FastAPI service's ``/recommendations``. Keeps a single copy of
    the model behind an HTTP boundary, at the cost of needing uvicorn running.

Both return the same :class:`Result`, so the chat page never learns which is in
use. Nothing here imports Streamlit: caching belongs to the UI layer, and
keeping this module free of it means the whole thing can be tested as ordinary
Python.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import joblib
import polars as pl
import requests

from app.api import Recommender, TitleNotFoundError

__all__ = [
    "BackendUnavailableError",
    "LocalBackend",
    "Result",
    "ServiceBackend",
    "Suggestion",
    "TitleNotFoundError",
    "build_country_index",
    "load_recommender",
    "suggest_first_match",
]

UNKNOWN_COUNTRY: Final = "—"
_HTTP_TIMEOUT: Final = 15.0


class BackendUnavailableError(RuntimeError):
    """Raised when a backend cannot be reached or fails to answer."""


@dataclass(frozen=True, slots=True)
class Suggestion:
    """One recommended film."""

    title: str
    score: float
    country: str
    release_year: int | None


@dataclass(frozen=True, slots=True)
class Result:
    """Recommendations plus the seed film they were derived from."""

    seed: str
    suggestions: tuple[Suggestion, ...]


# --------------------------------------------------------------------------- #
# Catalogue enrichment
# --------------------------------------------------------------------------- #


def _key(title: str, year: int | None) -> tuple[str, int | None]:
    return " ".join(str(title).split()).casefold(), year


def build_country_index(csv_path: Path) -> dict[tuple[str, int | None], str]:
    """Map ``(title, year)`` to production countries from the merged dataset.

    The model bundle's serving catalogue carries only seven columns and country
    is not among them, so the country has to come from ``movies_merged.csv``.

    The join key is the title and release year rather than ``movie_id``, which
    would be exact, because the FastAPI service returns neither an id nor a row
    index — only a title, a score and a year. Using one key for both backends
    keeps a single enrichment path instead of two that could disagree. The cost
    is measured rather than assumed: across the 25,000 films in the evaluation
    catalogue, ``(title, year)`` collides on 18 keys covering 36 rows, or 0.14%.

    Only three columns are read, so the 87 MB file costs little to scan.
    """
    if not csv_path.is_file():
        return {}

    frame = (
        pl.scan_csv(csv_path, infer_schema_length=0)
        .select("title_clean", "release_year", "production_countries")
        .filter(
            pl.col("production_countries").is_not_null()
            & (pl.col("production_countries") != "")
        )
        .collect()
    )

    index: dict[tuple[str, int | None], str] = {}
    for title, year, countries in frame.iter_rows():
        if not title:
            continue
        # release_year is read as text to avoid schema inference on a 52-column
        # file; a handful of rows have no year at all.
        parsed = int(float(year)) if year else None
        # The source is pipe-delimited because CSV cannot hold a list column.
        index[_key(title, parsed)] = ", ".join(c for c in countries.split("|") if c)
    return index


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


class Backend(Protocol):
    """What the chat page needs from a source of recommendations."""

    label: str

    def suggest(self, title: str, top_k: int) -> Result:
        """Recommend ``top_k`` films similar to ``title``.

        Raises:
            TitleNotFoundError: the title is not in the catalogue.
            BackendUnavailableError: the backend could not be reached.
        """


def load_recommender(model_path: Path, representation: str) -> Recommender:
    """Deserialise the joblib bundle into the API's ``Recommender``.

    The bundle holds plain data only, so this works without importing anything
    the training notebook defined. It is around 130 MB and takes several
    seconds, which is why the UI layer wraps this in ``st.cache_resource``.
    """
    if not model_path.is_file():
        msg = f"model bundle not found at {model_path}"
        raise BackendUnavailableError(msg)
    return Recommender.from_bundle(joblib.load(model_path), representation)


@dataclass(frozen=True, slots=True)
class LocalBackend:
    """Score in process against a loaded ``Recommender``.

    Safe to share across Streamlit sessions: the recommender is frozen and
    ``recommend`` mutates only arrays it derives itself, so nothing here is
    modified after construction.
    """

    recommender: Recommender
    countries: Mapping[tuple[str, int | None], str]
    label: str = "In-process model"

    def suggest(self, title: str, top_k: int) -> Result:
        row = self.recommender.resolve(title)  # raises TitleNotFoundError
        hits = self.recommender.recommend(row, top_k)
        return Result(
            seed=self.recommender.titles[row],
            suggestions=tuple(
                Suggestion(
                    title=name,
                    score=score,
                    country=self.countries.get(_key(name, year), UNKNOWN_COUNTRY),
                    release_year=year,
                )
                for name, score, year in hits
            ),
        )


@dataclass(frozen=True, slots=True)
class ServiceBackend:
    """Call the FastAPI service over HTTP."""

    base_url: str
    countries: Mapping[tuple[str, int | None], str]
    label: str = "FastAPI service"

    def suggest(self, title: str, top_k: int) -> Result:
        url = f"{self.base_url.rstrip('/')}/recommendations"
        try:
            response = requests.post(
                url,
                json={"name": title, "topK": top_k},
                timeout=_HTTP_TIMEOUT,
            )
        except requests.RequestException as exc:
            msg = f"could not reach {url}: {exc}"
            raise BackendUnavailableError(msg) from exc

        if response.status_code == requests.codes.not_found:
            raise TitleNotFoundError(title)
        if not response.ok:
            msg = f"{url} returned {response.status_code}: {response.text[:200]}"
            raise BackendUnavailableError(msg)

        payload: dict[str, Any] = response.json()
        # The service reports no seed of its own, so the query stands in for it.
        return Result(
            seed=title,
            suggestions=tuple(
                Suggestion(
                    title=rec["title"],
                    score=float(rec["score"]),
                    country=self.countries.get(
                        _key(rec["title"], rec.get("releaseYear")), UNKNOWN_COUNTRY
                    ),
                    release_year=rec.get("releaseYear"),
                )
                for rec in payload.get("recs", [])
            ),
        )


def suggest_first_match(backend: Backend, candidates: Iterable[str], top_k: int) -> Result:
    """Try each parsed title against ``backend`` and return the first that resolves.

    This is where the intent parser's ambiguity is settled. The parser cannot
    know whether "like Toy Story" names a film called *like Toy Story*; the
    catalogue can, so the candidates are tried in order and the first hit wins.
    """
    tried: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        tried.append(candidate)
        try:
            return backend.suggest(candidate, top_k)
        except TitleNotFoundError:
            continue
    raise TitleNotFoundError(tried[0] if tried else "")
