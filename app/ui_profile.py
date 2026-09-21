"""Data profiling page, backed by ydata-profiling.

Profiling all 89,178 rows and 52 columns explanatively takes minutes and a lot
of memory, so the page defaults to a 5,000-row sample of one column group in
minimal mode and lets the user dial it up. The report is cached per parameter
set, so changing a control and changing it back is free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import polars as pl
import streamlit as st

__all__ = ["render"]

# Presets over the merged catalogue's 52 columns. Profiling the whole frame at
# once produces a report too large to read; these group the columns the way the
# exploratory analysis treats them.
COLUMN_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "Ratings and popularity": (
        "title_clean", "release_year", "tmdb_vote_average", "tmdb_vote_count",
        "tmdb_popularity", "ml_rating_mean", "ml_rating_count", "ml_rating_std",
        "wr_tmdb", "wr_ml", "weighted_rating", "weighted_rating_source",
    ),
    "Content coverage": (
        "content_source", "has_content", "has_overview", "has_credits",
        "has_keywords", "has_genres", "genres", "original_language", "release_year",
    ),
    "Production attributes": (
        "title_clean", "release_year", "runtime_minutes", "budget", "revenue",
        "status", "adult", "video", "original_language", "production_countries",
    ),
    "Identity and provenance": (
        "movie_id", "tmdb_id", "imdb_id", "title_clean", "release_year",
        "in_ml_full", "in_ml_latest", "content_source",
    ),
    "Everything (52 columns)": (),
}

# Read as text and cast deliberately. Inferring types across 52 columns of a
# heterogeneous 87 MB file is slow and gets the sentinel columns wrong.
_NUMERIC: Final[frozenset[str]] = frozenset({
    "movie_id", "tmdb_id", "release_year", "runtime_minutes", "budget", "revenue",
    "tmdb_vote_average", "tmdb_vote_count", "tmdb_popularity", "ml_rating_mean",
    "ml_rating_count", "ml_rating_sum", "ml_rating_std", "wr_tmdb", "wr_ml",
    "weighted_rating", "tmdb_vote_count_pct_rank", "ml_rating_count_pct_rank",
})
_BOOLEAN: Final[frozenset[str]] = frozenset({
    "in_ml_full", "in_ml_latest", "has_content", "has_overview", "has_credits",
    "has_keywords", "has_genres", "adult", "video",
})

# Zero means "not recorded" in these columns, not "cost nothing". Left as zero
# they drag every distribution and correlation towards it.
_ZERO_IS_MISSING: Final[frozenset[str]] = frozenset({"budget", "revenue", "runtime_minutes"})


@st.cache_data(show_spinner="Reading the merged catalogue…")
def _load(csv_path: Path, mtime: float, columns: tuple[str, ...]) -> pd.DataFrame:
    """Load ``columns`` from the merged CSV as a typed pandas frame.

    ``mtime`` is not used in the body. It is a cache key: rebuilding the dataset
    changes the file's modification time, which invalidates this entry instead
    of silently serving a profile of the previous build.
    """
    lazy = pl.scan_csv(csv_path, infer_schema_length=0)
    if columns:
        lazy = lazy.select(list(dict.fromkeys(columns)))
    frame = lazy.collect()

    casts = [
        pl.col(name).cast(pl.Float64, strict=False)
        for name in frame.columns
        if name in _NUMERIC
    ] + [
        (pl.col(name) == "true").alias(name)
        for name in frame.columns
        if name in _BOOLEAN
    ]
    if casts:
        frame = frame.with_columns(casts)

    sentinels = [
        pl.when(pl.col(name) == 0).then(None).otherwise(pl.col(name)).alias(name)
        for name in frame.columns
        if name in _ZERO_IS_MISSING
    ]
    if sentinels:
        frame = frame.with_columns(sentinels)

    return frame.to_pandas()


@st.cache_resource(show_spinner="Profiling…", max_entries=4)
def _report(
    csv_path: Path, mtime: float, columns: tuple[str, ...], rows: int, minimal: bool
):
    """Build and cache a ProfileReport for one parameter set.

    Cached as a resource rather than as data because a ProfileReport is not
    usefully serialisable and the profiling extension needs the object itself.
    Reports are held per parameter set and capped at four so a session that
    sweeps the controls cannot grow the cache without bound.
    """
    from ydata_profiling import ProfileReport

    frame = _load(csv_path, mtime, columns)
    if 0 < rows < len(frame):
        # Seeded so the same controls give the same report between reruns.
        frame = frame.sample(n=rows, random_state=42).reset_index(drop=True)

    return ProfileReport(
        frame,
        title="Merged film catalogue",
        minimal=minimal,
        explorative=not minimal,
        progress_bar=False,
    )


def _display(report) -> None:
    """Render the report, preferring the Streamlit extension.

    ``streamlit-ydata-profiling`` declares ``Requires-Python >=3.10,<3.12`` and
    this project runs on 3.12, so it has to be installed with
    ``--ignore-requires-python``. The package is a thin component wrapper and
    works on 3.12, but the import is guarded: if the install did not take, the
    page falls back to embedding the same HTML directly rather than failing.
    """
    try:
        from streamlit_ydata_profiling import st_profile_report
    except ImportError:
        st.info(
            "`streamlit-ydata-profiling` is not installed, so the report is "
            "embedded directly. Install it with "
            "`pip install --ignore-requires-python streamlit-ydata-profiling==0.2.1`.",
            icon=":material/info:",
        )
        st.components.v1.html(report.to_html(), height=900, scrolling=True)
    else:
        st_profile_report(report, height=900, navbar=True)


def render(csv_path: Path) -> None:
    """Draw the profiling page."""
    st.title("Data profiling")
    st.caption(
        "Automated profile of `datasets/processed/movies_merged.csv`, the "
        "89,178-film catalogue produced by `scripts/build_dataset.py`."
    )

    if not csv_path.is_file():
        st.error(
            f"Merged dataset not found at `{csv_path}`. Build it with "
            "`python -m scripts.build_dataset` first.",
            icon=":material/error:",
        )
        return

    mtime = csv_path.stat().st_mtime

    left, middle, right = st.columns([2, 1, 1])
    group = left.selectbox(
        "Columns",
        list(COLUMN_GROUPS),
        help="Profiling all 52 columns at once produces a report that is hard to read.",
    )
    rows = middle.select_slider(
        "Rows sampled",
        options=[1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 0],
        value=5_000,
        format_func=lambda n: "All 89,178" if n == 0 else f"{n:,}",
        help="Sampling is seeded, so the same setting gives the same report.",
    )
    minimal = right.toggle(
        "Minimal mode",
        value=True,
        help=(
            "Minimal mode skips correlations, interactions and duplicate "
            "detection. Turning it off is far slower on large samples."
        ),
    )

    columns = COLUMN_GROUPS[group]
    if not minimal and (rows == 0 or rows > 10_000):
        st.warning(
            "Full profiling above 10,000 rows computes pairwise correlations and "
            "interactions and can take several minutes.",
            icon=":material/hourglass_top:",
        )
    if group in {"Production attributes", "Everything (52 columns)"}:
        st.caption(
            "`budget`, `revenue` and `runtime_minutes` encode 'not recorded' as "
            "zero in the source. They are converted to missing before profiling, "
            "so their statistics describe only the films that actually carry a value."
        )

    if not st.button("Generate report", type="primary"):
        st.info("Choose your settings and generate the report.", icon=":material/tune:")
        return

    report = _report(csv_path, mtime, columns, rows, minimal)
    _display(report)

    st.download_button(
        "Download report as HTML",
        data=report.to_html(),
        file_name="movies_merged_profile.html",
        mime="text/html",
        icon=":material/download:",
    )
