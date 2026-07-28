"""Build the merged, model-ready movie dataset for the content-based recommender.

This module fuses the four raw corpora found under ``datasets/raw`` into a single
item table (one row per MovieLens ``movieId``) plus a separate interactions table.

Spine
-----
The union of ``movielens-full/movies.csv`` and ``movielens-latest/movies.csv``
(89,178 films). The two snapshots were verified to be mutually consistent: zero
title, genre or IMDb-id conflicts across the 55,457 shared identifiers.

Content precedence
------------------
For every spine row, descriptive content is resolved in this order and the winner
recorded in the ``content_source`` column:

1. ``the-movies-dataset``  - richest source; the only one carrying cast and crew.
2. ``tmdb-930k``           - broad TMDB dump; fills overview/genres/keywords.
3. ``movielens``           - title and pipe-delimited genres backfill.
4. ``none``                - no usable content.

Outputs
-------
``movies_merged.parquet``  the item table (also returned in memory)
``interactions.parquet``   user-item ratings, streamed straight to disk
``build_manifest.json``    provenance: source hashes, per-stage row counts,
                           drop reasons, weighted-rating constants, versions

Example:
-------
>>> from pathlib import Path
>>> items = build_merged_dataset(
...     raw_dir=Path("datasets/raw"),
...     output_dir=Path("datasets/processed"),
... )
>>> items.select("movie_id", "title_clean", "weighted_rating").head()

"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import polars as pl
from polars.lazyframe.in_process import InProcessQuery

if TYPE_CHECKING:  # pragma: no cover - import cost avoided at runtime
    import pandas as pd

__all__ = [
    "BuildConfig",
    "ContentSource",
    "DatasetSchemaError",
    "MovieLensSnapshot",
    "OutputFormat",
    "WeightedRatingSource",
    "build_merged_dataset",
]

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: A single-quoted or double-quoted Python string literal, honouring backslash escapes.
#: ``the-movies-dataset`` stores ``repr()``-style JSON, so names containing an
#: apostrophe appear double-quoted (e.g. ``"based on children's book"``).
_QUOTED: Final[str] = r"""(?:'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")"""

_NAME_FIELD_RE: Final[str] = rf"'name':\s*{_QUOTED}"
_DIRECTOR_RE: Final[str] = rf"'job':\s*'Director'.*?'name':\s*{_QUOTED}"
_STRIP_TO_NAME_RE: Final[str] = r"(?s)^.*?'name':\s*"
_STRIP_TO_LAST_NAME_RE: Final[str] = r"(?s)^.*'name':\s*"
_UNESCAPE_RE: Final[str] = r"\\(['\"])"

_TITLE_YEAR_RE: Final[str] = r"^(.*)\s+\((\d{4})\)\s*$"
_TRAILING_ARTICLE_RE: Final[str] = (
    r"^(.*),\s+(The|A|An|La|Le|Les|El|Los|Las|Il|Der|Die|Das|Ein|Eine|Den|Det)$"
)

_NO_GENRES_SENTINEL: Final[str] = "(no genres listed)"
_SOUP_SEPARATOR: Final[str] = " "
_EMPTY_LIST: Final[pl.Expr] = pl.lit([], dtype=pl.List(pl.String))

#: Logical field -> candidate physical column names in the 930k TMDB dump.
#: Resolved case-insensitively at runtime because the two available vintages of
#: that file could not be inspected ahead of time (both exceed the transfer limit).
_TMDB_DUMP_FIELDS: Final[Mapping[str, tuple[str, ...]]] = {
    "tmdb_id": ("id", "tmdb_id"),
    "overview": ("overview",),
    "tagline": ("tagline",),
    "original_title": ("original_title",),
    "original_language": ("original_language",),
    "status": ("status",),
    "runtime": ("runtime",),
    "budget": ("budget",),
    "revenue": ("revenue",),
    "adult": ("adult",),
    "popularity": ("popularity",),
    "vote_average": ("vote_average",),
    "vote_count": ("vote_count",),
    "homepage": ("homepage",),
    "poster_path": ("poster_path",),
    "release_date": ("release_date",),
    "genres": ("genres",),
    "keywords": ("keywords",),
    "production_companies": ("production_companies",),
    "production_countries": ("production_countries",),
    "spoken_languages": ("spoken_languages",),
}
_TMDB_DUMP_REQUIRED: Final[frozenset[str]] = frozenset({"tmdb_id"})

_LIST_COLUMNS: Final[tuple[str, ...]] = (
    "genres",
    "keywords",
    "cast_top",
    "directors",
    "production_companies",
    "production_countries",
    "spoken_languages",
    "ml_tags",
)


class DatasetSchemaError(RuntimeError):
    """Raised when a raw source file does not expose the columns we require."""


class MovieLensSnapshot(StrEnum):
    """The MovieLens extracts present under ``datasets/raw``."""

    FULL = "movielens-full"
    LATEST = "movielens-latest"


class ContentSource(StrEnum):
    """Provenance of the descriptive content attached to a spine row."""

    MOVIES_DATASET = "the-movies-dataset"
    TMDB_DUMP = "tmdb-930k"
    MOVIELENS = "movielens"
    NONE = "none"


class WeightedRatingSource(StrEnum):
    """How the coalesced ``weighted_rating`` column was derived for a row."""

    BLENDED = "blended"
    MOVIELENS = "movielens_wr"
    TMDB = "tmdb_wr"
    RAW_MOVIELENS = "raw_movielens_mean"
    RAW_TMDB = "raw_tmdb_mean"
    NONE = "none"


class OutputFormat(StrEnum):
    """On-disk format for the two table artefacts.

    CSV is the default because it opens anywhere without a reader library. It
    cannot represent nested columns, so the eight ``List(String)`` columns are
    flattened to delimited strings on the way out; see
    :func:`_flatten_list_columns`. Parquet preserves them natively and is the
    better choice for reloading into the modelling notebooks.
    """

    CSV = "csv"
    PARQUET = "parquet"


@dataclass(frozen=True, slots=True, kw_only=True)
class BuildConfig:
    """Tunable parameters for :func:`build_merged_dataset`.

    Attributes
    ----------
    ratings_snapshot:
        Which MovieLens extract supplies the interactions table. Defaults to
        :attr:`MovieLensSnapshot.LATEST`; the two snapshots use *different*
        ``userId`` spaces, so they are never concatenated.
    tmdb_dump_filename:
        Preferred file inside ``tmdb-movies-930k``. Falls back to
        ``tmdb_dump_fallback_filename`` when absent.
    use_tmdb_dump:
        Enable the second-tier content layer.
    include_ml_tags:
        Fold the top ``max_ml_tags`` user-supplied MovieLens tags per film into
        ``ml_tags`` and ``metadata_soup``.
    max_cast:
        Number of leading cast members retained (TMDB orders cast by billing).
    quantile_for_m:
        Quantile of the vote-count distribution used as ``m`` in the IMDb
        weighted-rating formula.
    min_votes_for_wr:
        Vote count below which a source's weighted rating is treated as absent
        when coalescing.
    blend_weights:
        ``(movielens, tmdb)`` weights applied when both weighted ratings exist.
    drop_orphan_interactions:
        Exclude ratings whose ``movieId`` is absent from the item table. Older
        ratings extracts reference films that later snapshots retired; keeping
        them yields interactions that can never be scored, which silently
        depresses recall@k and NDCG@k.
    soup_fields:
        Columns concatenated into ``metadata_soup``, in order.
    hash_sources:
        Record SHA-256 digests of every input file in the manifest. Costs a few
        seconds of I/O over ~2.5 GB but makes the build fully auditable.

    """

    ratings_snapshot: MovieLensSnapshot = MovieLensSnapshot.LATEST
    tmdb_dump_filename: str = "TMDB_movie_dataset_v11.csv"
    tmdb_dump_fallback_filename: str = "tmdb_movie_dataset.csv"
    use_tmdb_dump: bool = True
    include_ml_tags: bool = True
    max_ml_tags: int = 10
    max_cast: int = 3
    quantile_for_m: float = 0.90
    min_votes_for_wr: int = 1
    blend_weights: tuple[float, float] = (0.5, 0.5)
    drop_orphan_interactions: bool = True
    soup_fields: tuple[str, ...] = (
        "genres",
        "keywords",
        "cast_top",
        "directors",
        "collection_name",
        "ml_tags",
    )
    hash_sources: bool = True
    output_format: OutputFormat = OutputFormat.CSV
    list_separator: str = "|"
    parquet_compression: str = "snappy"

    def __post_init__(self) -> None:
        """Validate the configuration eagerly, before any I/O is attempted."""
        if not self.list_separator:  # NEW
            msg = "list_separator must be a non-empty string"
            raise ValueError(msg)
        if not 0.0 < self.quantile_for_m < 1.0:
            msg = f"quantile_for_m must lie in (0, 1); got {self.quantile_for_m!r}"
            raise ValueError(msg)
        if self.max_cast < 1:
            msg = f"max_cast must be >= 1; got {self.max_cast!r}"
            raise ValueError(msg)
        if sum(self.blend_weights) <= 0:
            msg = f"blend_weights must sum to a positive value; got {self.blend_weights!r}"
            raise ValueError(msg)


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #

def _collect(frame: pl.LazyFrame, *, streaming: bool = True) -> pl.DataFrame | InProcessQuery:
    """Materialise ``frame``, preferring the streaming engine where available."""
    if not streaming:
        return frame.collect()
    try:
        return frame.collect(engine="streaming")
    except TypeError:  # pragma: no cover - polars < 1.24 keyword
        return frame.collect(streaming=True)


def _sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of ``path``, read incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(path: Path) -> Path:
    """Return ``path`` if it exists, else raise a directive :class:`FileNotFoundError`."""
    if not path.is_file():
        msg = f"Required raw input not found: {path}"
        raise FileNotFoundError(msg)
    return path


def _scan_all_strings(path: Path) -> pl.LazyFrame:
    """Lazily scan a CSV with every column typed as ``String``.

    Type inference is disabled deliberately: ``movies_metadata.csv`` contains
    three structurally corrupt rows that shift fields sideways, which derails
    inference and silently nulls whole columns. We cast explicitly instead.
    """
    return pl.scan_csv(
        path,
        infer_schema_length=0,
        ignore_errors=True,
        truncate_ragged_lines=True,
        quote_char='"',
    )


def _extract_names(column: str) -> pl.Expr:
    """Vectorised extraction of every ``'name': ...`` value from a JSON-ish blob.

    Two orders of magnitude faster than :func:`ast.literal_eval` and it keeps the
    work inside polars. :func:`_repair_names` re-parses any row this misses.
    """
    return (
        pl.col(column)
        .str.extract_all(_NAME_FIELD_RE)
        .list.eval(
            pl.element()
            .str.replace(_STRIP_TO_NAME_RE, "")
            .str.strip_chars("'\"")
            .str.replace_all(_UNESCAPE_RE, "$1")
        )
        .alias(column)
    )


def _extract_directors(column: str) -> pl.Expr:
    """Extract crew members whose ``job`` is exactly ``Director``.

    The trailing quote in ``'job': 'Director'`` is load-bearing: it prevents
    ``Director of Photography`` from matching, while ``Assistant Director`` never
    matches because the literal is anchored on the ``'job': `` prefix.
    """
    return (
        pl.col(column)
        .str.extract_all(_DIRECTOR_RE)
        .list.eval(
            pl.element()
            .str.replace(_STRIP_TO_LAST_NAME_RE, "")
            .str.strip_chars("'\"")
            .str.replace_all(_UNESCAPE_RE, "$1")
        )
        .alias("directors")
    )


def _names_from_literal(blob: str | None) -> list[str]:
    """Fallback parser: evaluate a ``repr``-style list of dicts and pull names."""
    if not blob:
        return []
    try:
        parsed = ast.literal_eval(blob)
    except (SyntaxError, ValueError, MemoryError, RecursionError, TypeError):
        return []
    if not isinstance(parsed, Iterable):
        return []
    return [
        str(entry["name"])
        for entry in parsed
        if isinstance(entry, Mapping) and entry.get("name") is not None
    ]


def _directors_from_literal(blob: str | None) -> list[str]:
    """Fallback parser for the crew column."""
    if not blob:
        return []
    try:
        parsed = ast.literal_eval(blob)
    except (SyntaxError, ValueError, MemoryError, RecursionError, TypeError):
        return []
    if not isinstance(parsed, Iterable):
        return []
    return [
        str(entry["name"])
        for entry in parsed
        if isinstance(entry, Mapping)
        and entry.get("job") == "Director"
        and entry.get("name") is not None
    ]


def _repair_names(
    frame: pl.DataFrame,
    *,
    raw_column: str,
    parsed_column: str,
    parser: Callable[[str | None], list[str]],
    must_contain: str | None = None,
) -> pl.DataFrame:
    """Re-parse rows where the regex found nothing but the raw blob was non-empty.

    Guards against exotic escaping in the source repr strings. In practice this
    touches a negligible number of rows; it exists so correctness never depends
    on the regex being exhaustive.

    ``raw_column`` and ``parsed_column`` must differ, so the original blob is
    still available to re-parse.

    ``must_contain`` narrows the suspicion test to rows whose raw blob contains a
    literal substring. Without it the director extractor would flag every film
    that simply has no credited director as a parse failure.
    """
    if raw_column == parsed_column:
        msg = (
            f"_repair_names needs the unparsed blob: {raw_column!r} was overwritten "
            f"by the parsed output. Alias the raw column before parsing."
        )
        raise ValueError(msg)
    suspicious = (
        pl.col(raw_column).is_not_null()
        & (pl.col(raw_column).str.strip_chars() != "[]")
        & (pl.col(raw_column).str.strip_chars() != "")
        & (pl.col(parsed_column).list.len() == 0)
    )
    if must_contain is not None:
        suspicious = suspicious & pl.col(raw_column).str.contains(
            must_contain, literal=True
        )
    affected = int(frame.select(suspicious.sum()).item())
    if affected == 0:
        return frame
    LOGGER.warning(
        "Regex parse missed %d row(s) for %r; falling back to ast.literal_eval",
        affected,
        raw_column,
    )
    return frame.with_columns(
        pl.when(suspicious)
        .then(pl.col(raw_column).map_elements(parser, return_dtype=pl.List(pl.String)))
        .otherwise(pl.col(parsed_column))
        .alias(parsed_column)
    )


def _split_delimited_names(column: str) -> pl.Expr:
    """Parse the 930k dump's list-ish fields.

    That dump stores plain ``", "``-delimited strings (``"Animation, Comedy"``)
    rather than JSON, but the exact encoding differs between its two vintages, so
    both shapes are handled.
    """
    raw = pl.col(column).fill_null("")
    return (
        pl.when(raw.str.starts_with("["))
        .then(
            raw.str.extract_all(_NAME_FIELD_RE).list.eval(
                pl.element().str.replace(_STRIP_TO_NAME_RE, "").str.strip_chars("'\"")
            )
        )
        .otherwise(
            raw.str.split(",").list.eval(
                pl.element().str.strip_chars().filter(pl.element().str.len_chars() > 0)
            )
        )
        .alias(column)
    )


def _tokenise(column: str) -> pl.Expr:
    """Lowercase a list column and strip internal whitespace.

    ``"Tom Hanks" -> "tomhanks"`` keeps multi-word entities as single vocabulary
    terms under ``CountVectorizer``/``TfidfVectorizer``, so two films sharing an
    actor are not credited for merely sharing a first name.
    """
    return pl.col(column).list.eval(
        pl.element().str.to_lowercase().str.replace_all(r"\s+", "")
    )


def _resolve_columns(
    available: Sequence[str],
    wanted: Mapping[str, tuple[str, ...]],
    *,
    required: frozenset[str],
    source: Path,
) -> dict[str, str]:
    """Map logical field names onto the physical columns present in ``source``."""
    lookup = {name.casefold(): name for name in available}
    resolved: dict[str, str] = {}
    for logical, candidates in wanted.items():
        for candidate in candidates:
            physical = lookup.get(candidate.casefold())
            if physical is not None:
                resolved[logical] = physical
                break
    missing = required - resolved.keys()
    if missing:
        msg = (
            f"{source.name} is missing required column(s) {sorted(missing)}; "
            f"found {sorted(available)}"
        )
        raise DatasetSchemaError(msg)
    absent = wanted.keys() - resolved.keys()
    if absent:
        LOGGER.info("%s: optional column(s) absent: %s", source.name, sorted(absent))
    return resolved


def _weighted_rating(mean: pl.Expr, votes: pl.Expr, *, m: float, c: float) -> pl.Expr:
    """IMDb weighted rating ``WR = v/(v+m)*R + m/(v+m)*C``."""
    v = votes.fill_null(0.0)
    r = mean.fill_null(c)
    return (v / (v + m)) * r + (m / (v + m)) * c


# --------------------------------------------------------------------------- #
# Stage 1 - spine
# --------------------------------------------------------------------------- #


def _load_spine(raw_dir: Path, stats: dict[str, Any]) -> pl.DataFrame:
    """Build the union spine of both MovieLens snapshots, one row per ``movieId``."""
    frames: list[pl.DataFrame] = []
    ids: dict[MovieLensSnapshot, set[int]] = {}

    # LATEST is loaded first so ``unique(keep="first")`` prefers the newer vintage.
    for snapshot in (MovieLensSnapshot.LATEST, MovieLensSnapshot.FULL):
        movies = _collect(
            _scan_all_strings(_require(raw_dir / snapshot / "movies.csv")).select(
                pl.col("movieId").cast(pl.Int32, strict=False).alias("movie_id"),
                pl.col("title").alias("title_raw"),
                pl.col("genres").alias("ml_genres_raw"),
            ),
            streaming=False,
        ).drop_nulls("movie_id")
        ids[snapshot] = set(movies["movie_id"].to_list())
        frames.append(movies)
        LOGGER.info("%s: %d film(s)", snapshot, movies.height)

    spine = pl.concat(frames, how="vertical").unique(
        subset="movie_id", keep="first", maintain_order=True
    )
    spine = spine.with_columns(
        pl.col("movie_id").is_in(list(ids[MovieLensSnapshot.FULL])).alias("in_ml_full"),
        pl.col("movie_id")
        .is_in(list(ids[MovieLensSnapshot.LATEST]))
        .alias("in_ml_latest"),
    )

    stats["spine"] = {
        "movielens_full": len(ids[MovieLensSnapshot.FULL]),
        "movielens_latest": len(ids[MovieLensSnapshot.LATEST]),
        "shared": len(ids[MovieLensSnapshot.FULL] & ids[MovieLensSnapshot.LATEST]),
        "full_only": len(ids[MovieLensSnapshot.FULL] - ids[MovieLensSnapshot.LATEST]),
        "latest_only": len(ids[MovieLensSnapshot.LATEST] - ids[MovieLensSnapshot.FULL]),
        "union": spine.height,
    }
    LOGGER.info("Union spine: %d film(s)", spine.height)
    return spine


def _load_links(raw_dir: Path, stats: dict[str, Any]) -> pl.DataFrame:
    """Union both ``links.csv`` files, preferring the newer snapshot on conflict."""
    frames: list[pl.DataFrame] = []
    for snapshot in (MovieLensSnapshot.LATEST, MovieLensSnapshot.FULL):
        frames.append(
            _collect(
                _scan_all_strings(_require(raw_dir / snapshot / "links.csv")).select(
                    pl.col("movieId").cast(pl.Int32, strict=False).alias("movie_id"),
                    # imdbId is zero-padded; keep the canonical ``tt`` form.
                    pl.col("imdbId").alias("imdb_id_raw"),
                    pl.col("tmdbId")
                    .cast(pl.Float64, strict=False)
                    .cast(pl.Int32, strict=False)
                    .alias("tmdb_id"),
                ),
                streaming=False,
            ).drop_nulls("movie_id")
        )

    latest, full = frames
    conflicts = latest.join(full, on="movie_id", how="inner", suffix="_full").filter(
        pl.col("tmdb_id").is_not_null()
        & pl.col("tmdb_id_full").is_not_null()
        & (pl.col("tmdb_id") != pl.col("tmdb_id_full"))
    )
    if conflicts.height:
        LOGGER.warning(
            "%d movie_id(s) disagree on tmdbId between snapshots; preferring %s. "
            "Affected ids: %s",
            conflicts.height,
            MovieLensSnapshot.LATEST,
            conflicts["movie_id"].to_list(),
        )
    stats["tmdb_id_conflicts"] = conflicts.height

    # Full outer join rather than concat + unique: the newer snapshot wins on a
    # genuine disagreement, but where it simply has no id we still fall back to
    # the older one instead of discarding a usable link.
    links = latest.join(full, on="movie_id", how="full", suffix="_full", coalesce=True)
    recovered = int(
        links.select(
            (pl.col("tmdb_id").is_null() & pl.col("tmdb_id_full").is_not_null()).sum()
        ).item()
    )
    if recovered:
        LOGGER.info(
            "Recovered %d tmdbId(s) from %s where %s had none",
            recovered,
            MovieLensSnapshot.FULL,
            MovieLensSnapshot.LATEST,
        )
    stats["tmdb_id_recovered_from_full"] = recovered

    links = links.select(
        "movie_id",
        pl.coalesce("tmdb_id", "tmdb_id_full").alias("tmdb_id"),
        pl.coalesce("imdb_id_raw", "imdb_id_raw_full").alias("_imdb_id_raw"),
    ).with_columns(
        pl.when(pl.col("_imdb_id_raw").is_not_null())
        .then(pl.concat_str([pl.lit("tt"), pl.col("_imdb_id_raw")]))
        .otherwise(None)
        .alias("imdb_id")
    )
    links = links.drop("_imdb_id_raw")
    stats["links_union"] = links.height
    return links


# --------------------------------------------------------------------------- #
# Stage 2 - content layers
# --------------------------------------------------------------------------- #


def _load_movies_dataset_content(
    raw_dir: Path, config: BuildConfig, stats: dict[str, Any]
) -> pl.DataFrame:
    """Tier-1 content: ``the-movies-dataset`` metadata + credits + keywords."""
    base = raw_dir / "the-movies-dataset"
    metadata_path = _require(base / "movies_metadata.csv")

    raw = _collect(_scan_all_strings(metadata_path), streaming=False)
    stats["movies_metadata_rows_read"] = raw.height

    # The ``id`` column carries three release dates instead of TMDB ids.
    typed = raw.with_columns(pl.col("id").cast(pl.Int64, strict=False).alias("tmdb_id"))
    corrupt = typed.filter(pl.col("tmdb_id").is_null())
    if corrupt.height:
        LOGGER.warning(
            "Dropping %d structurally corrupt movies_metadata row(s) with non-numeric id: %s",
            corrupt.height,
            corrupt["id"].to_list(),
        )
    stats["movies_metadata_corrupt_dropped"] = corrupt.height

    typed = (
        typed.drop_nulls("tmdb_id")
        .with_columns(
            pl.col("vote_count").cast(pl.Float64, strict=False).alias("_vote_count_sort")
        )
        .with_row_index("_source_row")
    )
    before = typed.height
    # Deduplicate on TMDB id, retaining the better-attested record. ``_source_row``
    # is the tie-break: duplicates sharing a vote count would otherwise be ordered
    # arbitrarily and the surviving row could change between builds.
    typed = typed.sort(
        ["_vote_count_sort", "_source_row"],
        descending=[True, False],
        nulls_last=True,
    ).unique(subset="tmdb_id", keep="first", maintain_order=True)
    stats["movies_metadata_duplicates_dropped"] = before - typed.height

    content = typed.select(
        pl.col("tmdb_id").cast(pl.Int32),
        pl.col("overview").alias("overview_text"),
        pl.col("tagline"),
        pl.col("original_title"),
        pl.col("original_language"),
        pl.col("status"),
        pl.col("homepage"),
        pl.col("poster_path"),
        pl.col("release_date"),
        pl.col("runtime").cast(pl.Float32, strict=False).alias("runtime_minutes"),
        pl.col("budget").cast(pl.Float64, strict=False).alias("budget"),
        pl.col("revenue").cast(pl.Float64, strict=False).alias("revenue"),
        (pl.col("adult").str.to_lowercase() == "true").alias("adult"),
        (pl.col("video").str.to_lowercase() == "true").alias("video"),
        pl.col("popularity").cast(pl.Float32, strict=False).alias("tmdb_popularity"),
        pl.col("vote_average").cast(pl.Float32, strict=False).alias("tmdb_vote_average"),
        pl.col("vote_count").cast(pl.Float32, strict=False).alias("tmdb_vote_count"),
        pl.col("belongs_to_collection").alias("_collection_raw"),
        pl.col("genres").alias("_genres_raw"),
        pl.col("production_companies").alias("_companies_raw"),
        pl.col("production_countries").alias("_countries_raw"),
        pl.col("spoken_languages").alias("_languages_raw"),
    )

    content = content.with_columns(
        _extract_names("_genres_raw").alias("genres"),
        _extract_names("_companies_raw").alias("production_companies"),
        _extract_names("_countries_raw").alias("production_countries"),
        _extract_names("_languages_raw").alias("spoken_languages"),
        _extract_names("_collection_raw").list.first().alias("collection_name"),
    )
    for raw_col, parsed_col in (
        ("_genres_raw", "genres"),
        ("_companies_raw", "production_companies"),
        ("_countries_raw", "production_countries"),
        ("_languages_raw", "spoken_languages"),
    ):
        content = _repair_names(
            content,
            raw_column=raw_col,
            parsed_column=parsed_col,
            parser=_names_from_literal,
        )
    content = content.drop(
        "_genres_raw",
        "_companies_raw",
        "_countries_raw",
        "_languages_raw",
        "_collection_raw",
    )

    credits_df = _load_credits(base / "credits.csv", config, stats)
    keywords_df = _load_keywords(base / "keywords.csv", stats)

    return (
        content.join(credits_df, on="tmdb_id", how="left")
        .join(keywords_df, on="tmdb_id", how="left")
        .with_columns(
            pl.col("cast_top").fill_null(_EMPTY_LIST),
            pl.col("directors").fill_null(_EMPTY_LIST),
            pl.col("keywords").fill_null(_EMPTY_LIST),
        )
    )


def _load_credits(
    path: Path, config: BuildConfig, stats: dict[str, Any]
) -> pl.DataFrame:
    """Parse ``credits.csv`` into leading cast and director lists."""
    raw = _collect(_scan_all_strings(_require(path)), streaming=False).with_columns(
        pl.col("id").cast(pl.Int32, strict=False).alias("tmdb_id")
    )
    before = raw.height
    raw = raw.drop_nulls("tmdb_id").unique(
        subset="tmdb_id", keep="first", maintain_order=True
    )
    stats["credits_duplicates_dropped"] = before - raw.height

    parsed = raw.with_columns(
        _extract_names("cast").alias("cast_all"),
        _extract_directors("crew"),
    )
    parsed = _repair_names(
        parsed, raw_column="cast", parsed_column="cast_all", parser=_names_from_literal
    )
    parsed = _repair_names(
        parsed,
        raw_column="crew",
        parsed_column="directors",
        parser=_directors_from_literal,
        # Only films whose crew mentions a director at all can be parse failures.
        must_contain="'job': 'Director'",
    )
    return parsed.select(
        "tmdb_id",
        pl.col("cast_all").list.head(config.max_cast).alias("cast_top"),
        "directors",
    )


def _load_keywords(path: Path, stats: dict[str, Any]) -> pl.DataFrame:
    """Parse ``keywords.csv`` into a list column."""
    raw = _collect(_scan_all_strings(_require(path)), streaming=False).with_columns(
        pl.col("id").cast(pl.Int32, strict=False).alias("tmdb_id")
    )
    before = raw.height
    raw = raw.drop_nulls("tmdb_id").unique(
        subset="tmdb_id", keep="first", maintain_order=True
    )
    stats["keywords_duplicates_dropped"] = before - raw.height

    parsed = raw.rename({"keywords": "_keywords_raw"}).with_columns(
        _extract_names("_keywords_raw").alias("keywords")
    )
    parsed = _repair_names(
        parsed,
        raw_column="_keywords_raw",
        parsed_column="keywords",
        parser=_names_from_literal,
    )
    return parsed.select("tmdb_id", "keywords")


def _load_tmdb_dump_content(
    raw_dir: Path, config: BuildConfig, stats: dict[str, Any]
) -> pl.DataFrame | None:
    """Tier-2 content: the ~1M-row TMDB dump, streamed and column-introspected."""
    directory = raw_dir / "tmdb-movies-930k"
    path = directory / config.tmdb_dump_filename
    if not path.is_file():
        path = directory / config.tmdb_dump_fallback_filename
    if not path.is_file():
        LOGGER.warning("No TMDB dump found in %s; skipping tier-2 content", directory)
        return None

    lazy = _scan_all_strings(path)
    resolved = _resolve_columns(
        lazy.collect_schema().names(),
        _TMDB_DUMP_FIELDS,
        required=_TMDB_DUMP_REQUIRED,
        source=path,
    )
    stats["tmdb_dump_file"] = path.name
    stats["tmdb_dump_columns_resolved"] = resolved

    projections: list[pl.Expr] = [
        pl.col(resolved["tmdb_id"]).cast(pl.Int32, strict=False).alias("tmdb_id")
    ]
    scalar_specs: tuple[tuple[str, str, Any], ...] = (
        ("overview", "overview_text", None),
        ("tagline", "tagline", None),
        ("original_title", "original_title", None),
        ("original_language", "original_language", None),
        ("status", "status", None),
        ("homepage", "homepage", None),
        ("poster_path", "poster_path", None),
        ("release_date", "release_date", None),
        ("runtime", "runtime_minutes", pl.Float32),
        ("budget", "budget", pl.Float64),
        ("revenue", "revenue", pl.Float64),
        ("popularity", "tmdb_popularity", pl.Float32),
        ("vote_average", "tmdb_vote_average", pl.Float32),
        ("vote_count", "tmdb_vote_count", pl.Float32),
    )
    for logical, alias, dtype in scalar_specs:
        physical = resolved.get(logical)
        if physical is None:
            continue
        expr = pl.col(physical)
        projections.append(
            (expr.cast(dtype, strict=False) if dtype is not None else expr).alias(alias)
        )

    if (physical := resolved.get("adult")) is not None:
        projections.append((pl.col(physical).str.to_lowercase() == "true").alias("adult"))

    list_specs = (
        ("genres", "genres"),
        ("keywords", "keywords"),
        ("production_companies", "production_companies"),
        ("production_countries", "production_countries"),
        ("spoken_languages", "spoken_languages"),
    )
    for logical, alias in list_specs:
        physical = resolved.get(logical)
        if physical is not None:
            projections.append(_split_delimited_names(physical).alias(alias))

    dump = _collect(lazy.select(projections)).drop_nulls("tmdb_id")
    before = dump.height
    dump = dump.unique(subset="tmdb_id", keep="first", maintain_order=True)
    stats["tmdb_dump_rows"] = dump.height
    stats["tmdb_dump_duplicates_dropped"] = before - dump.height
    LOGGER.info("TMDB dump (%s): %d unique film(s)", path.name, dump.height)
    return dump


# --------------------------------------------------------------------------- #
# Stage 3 - ratings and tags
# --------------------------------------------------------------------------- #


def _ratings_path(raw_dir: Path, config: BuildConfig) -> Path:
    """Locate the ratings file for the configured snapshot."""
    return _require(raw_dir / config.ratings_snapshot / "ratings.csv")


_RATINGS_SCHEMA: Final[Mapping[str, Any]] = {
    "userId": pl.Int32,
    "movieId": pl.Int32,
    "rating": pl.Float32,
    "timestamp": pl.Int64,
}


def _aggregate_ratings(
    raw_dir: Path, config: BuildConfig, stats: dict[str, Any]
) -> pl.DataFrame:
    """Stream the ratings file and reduce it to per-film aggregates."""
    path = _ratings_path(raw_dir, config)
    aggregates = _collect(
        pl.scan_csv(path, schema_overrides=dict(_RATINGS_SCHEMA))
        .group_by("movieId")
        .agg(
            pl.col("rating").mean().cast(pl.Float32).alias("ml_rating_mean"),
            pl.col("rating").count().cast(pl.Int32).alias("ml_rating_count"),
            pl.col("rating").sum().cast(pl.Float32).alias("ml_rating_sum"),
            pl.col("rating").std().cast(pl.Float32).alias("ml_rating_std"),
            pl.col("timestamp").max().alias("ml_last_rated_at"),
        )
        .rename({"movieId": "movie_id"})
    )
    stats["ratings_source"] = str(config.ratings_snapshot)
    stats["rated_films"] = aggregates.height
    stats["total_ratings"] = int(aggregates["ml_rating_count"].sum())
    LOGGER.info(
        "Aggregated %d rating(s) across %d film(s) from %s",
        stats["total_ratings"],
        aggregates.height,
        config.ratings_snapshot,
    )
    return aggregates

def _flatten_list_columns(frame: pl.DataFrame, separator: str) -> pl.DataFrame:
    """Join every ``List(String)`` column into a delimited string.

    CSV has no nested type, so ``write_csv`` raises on the eight list columns.
    Values already containing ``separator`` cannot survive a round trip; they are
    counted and reported rather than silently mangled, and Parquet avoids the
    problem entirely.
    """
    list_columns = [
        name for name, dtype in frame.schema.items() if dtype == pl.List(pl.String)
    ]
    if not list_columns:
        return frame
    collisions = int(
        frame.select(
            pl.sum_horizontal(
                [
                    pl.col(name)
                    .list.eval(pl.element().str.contains(separator, literal=True))
                    .list.any()
                    .fill_null(False)
                    for name in list_columns
                ]
            ).sum()
        ).item()
        or 0
    )
    if collisions:
        LOGGER.warning(
            "%d list value(s) already contain the separator %r and will not "
            "survive a CSV round trip; use output_format=parquet to avoid this",
            collisions,
            separator,
        )
    return frame.with_columns(
        [pl.col(name).list.join(separator).alias(name) for name in list_columns]
    )

def _write_frame(frame: pl.DataFrame, stem: Path, config: BuildConfig) -> Path:
    """Write ``frame`` in the configured format, returning the path written."""
    destination = stem.with_suffix(f".{config.output_format}")
    if config.output_format is OutputFormat.PARQUET:
        frame.write_parquet(destination, compression=config.parquet_compression)
    else:
        _flatten_list_columns(frame, config.list_separator).write_csv(destination)
    return destination


def _write_interactions(
    raw_dir: Path,
    output_dir: Path,
    config: BuildConfig,
    *,
    valid_movie_ids: pl.Series,
) -> Path:
    """Stream ratings straight to Parquet without materialising them in memory."""
    destination = (output_dir / "interactions").with_suffix(f".{config.output_format}")
    interactions = pl.scan_csv(
        _ratings_path(raw_dir, config), schema_overrides=dict(_RATINGS_SCHEMA)
    ).select(
        pl.col("userId").alias("user_id"),
        pl.col("movieId").alias("movie_id"),
        pl.col("rating"),
        pl.from_epoch(pl.col("timestamp"), time_unit="s").alias("rated_at"),
    )
    if config.drop_orphan_interactions:
        # Semi-join rather than ``is_in``: it streams, so peak memory stays flat
        # regardless of how large the ratings file is.
        catalogue = pl.LazyFrame({"movie_id": valid_movie_ids})
        interactions = interactions.join(catalogue, on="movie_id", how="semi")

    # Both sinks stream, so the ratings file is never held in memory whole.
    if config.output_format is OutputFormat.PARQUET:
        interactions.sink_parquet(destination, compression=config.parquet_compression)
    else:
        interactions.sink_csv(destination)
    LOGGER.info("Wrote interactions to %s", destination)
    return destination


def _aggregate_tags(
    raw_dir: Path, config: BuildConfig, stats: dict[str, Any]
) -> pl.DataFrame:
    """Return the ``max_ml_tags`` most frequent user tags per film."""
    path = raw_dir / config.ratings_snapshot / "tags.csv"
    if not path.is_file():
        path = raw_dir / MovieLensSnapshot.FULL / "tags.csv"
    if not path.is_file():
        LOGGER.warning("No MovieLens tags.csv available; skipping tag enrichment")
        return pl.DataFrame(schema={"movie_id": pl.Int32, "ml_tags": pl.List(pl.String)})

    tags = _collect(
        pl.scan_csv(path, infer_schema_length=0)
        .select(
            pl.col("movieId").cast(pl.Int32, strict=False).alias("movie_id"),
            pl.col("tag").str.strip_chars().str.to_lowercase().alias("tag"),
        )
        .drop_nulls()
        .filter(pl.col("tag").str.len_chars() > 0)
        .group_by("movie_id", "tag")
        .len()
        # Alphabetical tie-break on ``tag``: without it, equally frequent tags
        # are ordered arbitrarily and successive builds disagree on which ones
        # survive the head(), making metadata_soup irreproducible.
        .sort(["len", "tag"], descending=[True, False])
        .group_by("movie_id", maintain_order=True)
        .agg(pl.col("tag").head(config.max_ml_tags).alias("ml_tags"))
        .sort("movie_id")
    )
    stats["tagged_films"] = tags.height
    LOGGER.info("Attached tags for %d film(s) from %s", tags.height, path.parent.name)
    return tags


# --------------------------------------------------------------------------- #
# Stage 4 - assembly
# --------------------------------------------------------------------------- #


def _coalesce_content(spine: pl.DataFrame, dump: pl.DataFrame | None) -> pl.DataFrame:
    """Merge tier-2 content beneath tier-1, recording the winning source."""
    tier1 = pl.col("overview_text").is_not_null() | (pl.col("genres").list.len() > 0)
    if dump is None:
        return spine.with_columns(
            pl.when(tier1)
            .then(pl.lit(str(ContentSource.MOVIES_DATASET)))
            .otherwise(pl.lit(str(ContentSource.NONE)))
            .alias("content_source")
        )

    merged = spine.join(dump, on="tmdb_id", how="left", suffix="_dump")
    tier2_columns = [name for name in dump.columns if name != "tmdb_id"]

    tier2_present = pl.lit(False)
    if "overview_text_dump" in merged.columns:
        tier2_present = tier2_present | pl.col("overview_text_dump").is_not_null()
    if "genres_dump" in merged.columns:
        tier2_present = tier2_present | (pl.col("genres_dump").list.len() > 0)

    merged = merged.with_columns(
        pl.when(tier1)
        .then(pl.lit(str(ContentSource.MOVIES_DATASET)))
        .when(tier2_present)
        .then(pl.lit(str(ContentSource.TMDB_DUMP)))
        .otherwise(pl.lit(str(ContentSource.NONE)))
        .alias("content_source")
    )

    coalesced: list[pl.Expr] = []
    for name in tier2_columns:
        dump_name = f"{name}_dump"
        if dump_name not in merged.columns:
            continue
        if merged.schema[name] == pl.List(pl.String):
            coalesced.append(
                pl.when(pl.col(name).list.len() > 0)
                .then(pl.col(name))
                .otherwise(pl.col(dump_name).fill_null(_EMPTY_LIST))
                .alias(name)
            )
        else:
            coalesced.append(pl.coalesce(pl.col(name), pl.col(dump_name)).alias(name))

    merged = merged.with_columns(coalesced)
    return merged.drop([c for c in merged.columns if c.endswith("_dump")])


def _backfill_from_movielens(frame: pl.DataFrame) -> pl.DataFrame:
    """Guarantee every row has a title, a year and at least genre-level content."""
    ml_genres = (
        pl.when(
            pl.col("ml_genres_raw").is_null()
            | (pl.col("ml_genres_raw") == _NO_GENRES_SENTINEL)
        )
        .then(_EMPTY_LIST)
        .otherwise(pl.col("ml_genres_raw").str.split("|"))
    )
    title_stripped = pl.col("title_raw").str.extract(_TITLE_YEAR_RE, 1)
    return (
        frame.with_columns(
            pl.coalesce(title_stripped, pl.col("title_raw")).alias("_title_no_year"),
            pl.col("title_raw")
            .str.extract(_TITLE_YEAR_RE, 2)
            .cast(pl.Int16, strict=False)
            .alias("_ml_year"),
            ml_genres.alias("_ml_genres"),
        )
        .with_columns(
            # "American President, The" -> "The American President"
            pl.when(pl.col("_title_no_year").str.contains(_TRAILING_ARTICLE_RE))
            .then(
                pl.concat_str(
                    pl.col("_title_no_year").str.extract(_TRAILING_ARTICLE_RE, 2),
                    pl.lit(" "),
                    pl.col("_title_no_year").str.extract(_TRAILING_ARTICLE_RE, 1),
                )
            )
            .otherwise(pl.col("_title_no_year"))
            .alias("title_clean"),
            pl.when(pl.col("genres").list.len() > 0)
            .then(pl.col("genres"))
            .otherwise(pl.col("_ml_genres"))
            .alias("genres"),
            pl.coalesce(
                pl.col("release_date").str.slice(0, 4).cast(pl.Int16, strict=False),
                pl.col("_ml_year"),
            ).alias("release_year"),
            pl.when(pl.col("content_source") == str(ContentSource.NONE))
            .then(
                pl.when(pl.col("_ml_genres").list.len() > 0)
                .then(pl.lit(str(ContentSource.MOVIELENS)))
                .otherwise(pl.lit(str(ContentSource.NONE)))
            )
            .otherwise(pl.col("content_source"))
            .alias("content_source"),
        )
        .drop("_title_no_year", "_ml_year", "_ml_genres")
    )


def _add_quality_flags(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach explicit completeness booleans instead of filtering rows away."""
    return frame.with_columns(
        (
            pl.col("overview_text").is_not_null()
            & (pl.col("overview_text").str.strip_chars().str.len_chars() > 0)
        ).alias("has_overview"),
        (pl.col("cast_top").list.len() > 0).alias("has_credits"),
        (pl.col("keywords").list.len() > 0).alias("has_keywords"),
        (pl.col("genres").list.len() > 0).alias("has_genres"),
        (pl.col("content_source") != str(ContentSource.NONE)).alias("has_content"),
    )


def _add_demographic_scores(
    frame: pl.DataFrame, config: BuildConfig, stats: dict[str, Any]
) -> pl.DataFrame:
    """Compute raw aggregates, both IMDb weighted ratings, and their coalescence."""
    qualifying_tmdb = frame.filter(pl.col("tmdb_vote_count") > 0)
    qualifying_ml = frame.filter(pl.col("ml_rating_count") > 0)

    m_tmdb = float(
        qualifying_tmdb["tmdb_vote_count"].quantile(config.quantile_for_m) or 0.0
    )
    c_tmdb = float(qualifying_tmdb["tmdb_vote_average"].mean() or 0.0)
    m_ml = float(
        qualifying_ml["ml_rating_count"].cast(pl.Float64).quantile(config.quantile_for_m)
        or 0.0
    )
    c_ml = float(qualifying_ml["ml_rating_mean"].mean() or 0.0)

    stats["weighted_rating_constants"] = {
        "quantile": config.quantile_for_m,
        "m_tmdb": m_tmdb,
        "C_tmdb": c_tmdb,
        "m_movielens": m_ml,
        "C_movielens": c_ml,
    }
    LOGGER.info(
        "Weighted-rating constants: m_tmdb=%.1f C_tmdb=%.3f m_ml=%.1f C_ml=%.3f",
        m_tmdb,
        c_tmdb,
        m_ml,
        c_ml,
    )

    w_ml, w_tmdb = config.blend_weights
    total = w_ml + w_tmdb
    threshold = config.min_votes_for_wr

    frame = frame.with_columns(
        _weighted_rating(
            pl.col("tmdb_vote_average"), pl.col("tmdb_vote_count"), m=m_tmdb, c=c_tmdb
        )
        .cast(pl.Float32)
        .alias("wr_tmdb"),
        _weighted_rating(
            pl.col("ml_rating_mean"),
            pl.col("ml_rating_count").cast(pl.Float64),
            m=m_ml,
            c=c_ml,
        )
        .cast(pl.Float32)
        .alias("wr_ml"),
        pl.col("tmdb_vote_count")
        .rank("average")
        .truediv(pl.len())
        .cast(pl.Float32)
        .alias("tmdb_vote_count_pct_rank"),
        pl.col("ml_rating_count")
        .rank("average")
        .truediv(pl.len())
        .cast(pl.Float32)
        .alias("ml_rating_count_pct_rank"),
    )

    has_ml = pl.col("ml_rating_count").fill_null(0) >= threshold
    has_tmdb = pl.col("tmdb_vote_count").fill_null(0) >= threshold

    return frame.with_columns(
        pl.when(has_ml & has_tmdb)
        .then(((pl.col("wr_ml") * w_ml) + (pl.col("wr_tmdb") * w_tmdb)) / total)
        .when(has_ml)
        .then(pl.col("wr_ml"))
        .when(has_tmdb)
        .then(pl.col("wr_tmdb"))
        .otherwise(pl.coalesce(pl.col("ml_rating_mean"), pl.col("tmdb_vote_average")))
        .cast(pl.Float32)
        .alias("weighted_rating"),
        pl.when(has_ml & has_tmdb)
        .then(pl.lit(str(WeightedRatingSource.BLENDED)))
        .when(has_ml)
        .then(pl.lit(str(WeightedRatingSource.MOVIELENS)))
        .when(has_tmdb)
        .then(pl.lit(str(WeightedRatingSource.TMDB)))
        .when(pl.col("ml_rating_mean").is_not_null())
        .then(pl.lit(str(WeightedRatingSource.RAW_MOVIELENS)))
        .when(pl.col("tmdb_vote_average").is_not_null())
        .then(pl.lit(str(WeightedRatingSource.RAW_TMDB)))
        .otherwise(pl.lit(str(WeightedRatingSource.NONE)))
        .alias("weighted_rating_source"),
    )


def _build_soup(frame: pl.DataFrame, config: BuildConfig) -> pl.DataFrame:
    """Concatenate the configured fields into a single vectorisable string."""
    parts: list[pl.Expr] = []
    for name in config.soup_fields:
        if name not in frame.columns:
            LOGGER.debug("soup field %r absent; skipping", name)
            continue
        if frame.schema[name] == pl.List(pl.String):
            parts.append(_tokenise(name).list.join(_SOUP_SEPARATOR))
        else:
            parts.append(
                pl.col(name).fill_null("").str.to_lowercase().str.replace_all(r"\s+", "")
            )
    if not parts:
        return frame.with_columns(pl.lit("").alias("metadata_soup"))
    return frame.with_columns(
        pl.concat_str(parts, separator=_SOUP_SEPARATOR, ignore_nulls=True)
        .str.replace_all(r"\s{2,}", " ")
        .str.strip_chars()
        .alias("metadata_soup")
    )


_FINAL_COLUMN_ORDER: Final[tuple[str, ...]] = (
    # identity
    "movie_id",
    "tmdb_id",
    "imdb_id",
    "title_raw",
    "title_clean",
    "release_year",
    # provenance
    "in_ml_full",
    "in_ml_latest",
    "content_source",
    "has_content",
    "has_overview",
    "has_credits",
    "has_keywords",
    "has_genres",
    # structured content
    "genres",
    "keywords",
    "cast_top",
    "directors",
    "production_companies",
    "production_countries",
    "spoken_languages",
    "ml_tags",
    # scalar content
    "overview_text",
    "tagline",
    "original_title",
    "original_language",
    "status",
    "collection_name",
    "runtime_minutes",
    "budget",
    "revenue",
    "adult",
    "video",
    "homepage",
    "poster_path",
    "release_date",
    # derived text
    "metadata_soup",
    # demographic
    "tmdb_vote_average",
    "tmdb_vote_count",
    "tmdb_popularity",
    "tmdb_vote_count_pct_rank",
    "ml_rating_mean",
    "ml_rating_count",
    "ml_rating_sum",
    "ml_rating_std",
    "ml_rating_count_pct_rank",
    "ml_last_rated_at",
    "wr_tmdb",
    "wr_ml",
    "weighted_rating",
    "weighted_rating_source",
)


def _finalise(frame: pl.DataFrame) -> pl.DataFrame:
    """Apply the canonical column order and normalise empty containers."""
    frame = frame.with_columns(
        [
            pl.col(name).fill_null(_EMPTY_LIST)
            for name in _LIST_COLUMNS
            if name in frame.columns
        ]
    )
    ordered = [name for name in _FINAL_COLUMN_ORDER if name in frame.columns]
    remainder = [
        name for name in frame.columns if name not in ordered and not name.startswith("_")
    ]
    if remainder:
        LOGGER.debug("Appending unlisted columns: %s", remainder)
    return frame.select(*ordered, *remainder).sort("movie_id")


def _write_manifest(
    output_dir: Path, raw_dir: Path, config: BuildConfig, stats: dict[str, Any]
) -> Path:
    """Persist a reproducibility record beside the Parquet artifacts."""
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "raw_dir": str(raw_dir.resolve()),
        "config": {
            "ratings_snapshot": str(config.ratings_snapshot),
            "use_tmdb_dump": config.use_tmdb_dump,
            "include_ml_tags": config.include_ml_tags,
            "max_ml_tags": config.max_ml_tags,
            "max_cast": config.max_cast,
            "quantile_for_m": config.quantile_for_m,
            "min_votes_for_wr": config.min_votes_for_wr,
            "blend_weights": list(config.blend_weights),
            "drop_orphan_interactions": config.drop_orphan_interactions,
            "output_format": str(config.output_format),
            "list_separator": config.list_separator,
            "soup_fields": list(config.soup_fields),
        },
        "statistics": stats,
        "environment": {
            "python": sys.version.split()[0],
            "polars": pl.__version__,
        },
    }
    destination = output_dir / "build_manifest.json"
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    LOGGER.info("Wrote manifest to %s", destination)
    return destination


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def build_merged_dataset(
    *,
    raw_dir: Path,
    output_dir: Path | None = None,
    config: BuildConfig | None = None,
    to_pandas: bool = False,
) -> pl.DataFrame | pd.DataFrame:
    """Construct the merged, model-ready item table for the recommender baselines.

    Parameters
    ----------
    raw_dir:
        Directory containing ``movielens-full``, ``movielens-latest``,
        ``the-movies-dataset`` and ``tmdb-movies-930k``.
    output_dir:
        When given, ``movies_merged.parquet``, ``interactions.parquet`` and
        ``build_manifest.json`` are written here (created if necessary). When
        ``None`` the build is performed entirely in memory and nothing is
        persisted.
    config:
        Build parameters; see :class:`BuildConfig`. Defaults are used when omitted.
    to_pandas:
        Convert the result to a ``pandas.DataFrame`` before returning, for
        notebooks that are already pandas-based. Requires ``pyarrow``.

    Returns
    -------
    One row per MovieLens ``movieId`` (the union of both snapshots), carrying
    identity, provenance, structured content, a prebuilt ``metadata_soup`` and
    the demographic scores.

    Raises
    ------
    FileNotFoundError
        A mandatory raw input is missing.
    NotADirectoryError
        ``raw_dir`` does not exist.
    DatasetSchemaError
        A raw input lacks a column the pipeline requires.

    """
    config = config or BuildConfig()
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        msg = f"raw_dir does not exist or is not a directory: {raw_dir}"
        raise NotADirectoryError(msg)

    started = time.perf_counter()
    stats: dict[str, Any] = {}

    LOGGER.info("Stage 1/5: building union spine")
    spine = _load_spine(raw_dir, stats)
    links = _load_links(raw_dir, stats)
    frame = spine.join(links, on="movie_id", how="left")

    LOGGER.info("Stage 2/5: resolving content layers")
    tier1 = _load_movies_dataset_content(raw_dir, config, stats)
    frame = frame.join(tier1, on="tmdb_id", how="left")
    tier2 = (
        _load_tmdb_dump_content(raw_dir, config, stats) if config.use_tmdb_dump else None
    )
    frame = _coalesce_content(frame, tier2)
    frame = _backfill_from_movielens(frame)
    frame = _add_quality_flags(frame)

    LOGGER.info("Stage 3/5: aggregating ratings and tags")
    rating_aggregates = _aggregate_ratings(raw_dir, config, stats)
    orphans = rating_aggregates.join(
        frame.select("movie_id"), on="movie_id", how="anti"
    )
    stats["orphan_rated_films"] = orphans.height
    stats["orphan_ratings"] = int(orphans["ml_rating_count"].sum() or 0)
    if orphans.height:
        LOGGER.warning(
            "%d rated film(s) (%d rating(s)) are absent from the catalogue; "
            "%s from interactions.parquet",
            orphans.height,
            stats["orphan_ratings"],
            "dropping" if config.drop_orphan_interactions else "retaining",
        )
    frame = frame.join(rating_aggregates, on="movie_id", how="left")
    if config.include_ml_tags:
        frame = frame.join(
            _aggregate_tags(raw_dir, config, stats), on="movie_id", how="left"
        )

    LOGGER.info("Stage 4/5: deriving features and demographic scores")
    frame = _add_demographic_scores(frame, config, stats)
    frame = _build_soup(frame, config)
    frame = _finalise(frame)

    stats["final_rows"] = frame.height
    stats["content_source_distribution"] = dict(
        frame.group_by("content_source").len().sort("len", descending=True).iter_rows()
    )
    stats["coverage"] = {
        name: int(frame[name].sum())
        for name in (
            "has_content",
            "has_overview",
            "has_credits",
            "has_keywords",
            "has_genres",
        )
    }

    LOGGER.info("Stage 5/5: persisting artifacts")
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        items_path = output_dir / "movies_merged.parquet"
        # frame.write_parquet(items_path, compression=config.parquet_compression)
        # frame.write_csv(items_path, compression="uncompressed", separator=";")
        _write_frame(frame, items_path, config)
        LOGGER.info("Wrote item table (%d rows) to %s", frame.height, items_path)
        _write_interactions(
            raw_dir, output_dir, config, valid_movie_ids=frame["movie_id"]
        )
        if config.hash_sources:
            stats["source_files"] = {
                str(path.relative_to(raw_dir)): {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in sorted(raw_dir.rglob("*.csv"))
            }
        _write_manifest(output_dir, raw_dir, config, stats)

    LOGGER.info(
        "Build complete: %d film(s), %.1f%% with real content, in %.1fs",
        frame.height,
        100.0 * stats["coverage"]["has_content"] / max(frame.height, 1),
        time.perf_counter() - started,
    )

    if to_pandas:
        try:
            return frame.to_pandas()
        except ModuleNotFoundError as error:  # pragma: no cover - env dependent
            msg = (
                "to_pandas=True requires both pandas and pyarrow. "
                "Install them with: uv add pandas pyarrow"
            )
            raise ModuleNotFoundError(msg) from error
    return frame


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="build_dataset",
        description="Build the merged content-based recommender dataset.",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/processed"))
    parser.add_argument(
        "--ratings-snapshot",
        type=MovieLensSnapshot,
        choices=tuple(MovieLensSnapshot),
        default=MovieLensSnapshot.LATEST,
    )
    parser.add_argument(
        "--no-tmdb-dump", action="store_true", help="skip the tier-2 content layer"
    )
    parser.add_argument(
        "--no-ml-tags", action="store_true", help="skip MovieLens tag enrichment"
    )
    parser.add_argument(
        "--no-hash", action="store_true", help="skip SHA-256 source hashing"
    )
    parser.add_argument(
        "--keep-orphan-interactions",
        action="store_true",
        help="retain ratings whose movieId is absent from the catalogue",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        type=OutputFormat,
        choices=tuple(OutputFormat),
        default=OutputFormat.CSV,
        help="output format for both tables (default: csv)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point. Returns a process exit code."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    try:
        build_merged_dataset(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            config=BuildConfig(
                ratings_snapshot=args.ratings_snapshot,
                use_tmdb_dump=not args.no_tmdb_dump,
                include_ml_tags=not args.no_ml_tags,
                hash_sources=not args.no_hash,
                drop_orphan_interactions=not args.keep_orphan_interactions,
                output_format=args.output_format,
            ),
        )
    except (FileNotFoundError, NotADirectoryError, DatasetSchemaError):
        LOGGER.exception("Build failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
