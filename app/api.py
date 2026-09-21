"""Content-based movie recommendation API.

Serves the v0 baseline dumped by ``03_baseline_model.ipynb``: given a film title,
return the most similar films by cosine similarity over its TF-IDF representation.

Run locally::

    uvicorn app.api:app --reload

Configuration is environment-driven with a ``RECSYS_`` prefix::

    RECSYS_MODEL_PATH=models/baseline_v0.pkl
    RECSYS_REPRESENTATION=combined_tfidf
"""

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Final, Self

import joblib
import numpy as np
import scipy.sparse as sp
from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from pydantic_settings import BaseSettings, SettingsConfigDict
from sklearn.utils.extmath import safe_sparse_dot

__all__ = ["Recommender", "Settings", "app"]

logger: Final = logging.getLogger("recsys.api")

_MISSING_TITLE: Final = "unknown title"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


class Settings(BaseSettings):
    """Runtime configuration, overridable from the environment or a ``.env`` file."""

    # protected_namespaces=() is required: Pydantic reserves the `model_` prefix,
    # and `model_path` would otherwise emit a shadowing warning.
    model_config = SettingsConfigDict(
        env_prefix="RECSYS_", env_file=".env", extra="ignore", protected_namespaces=()
    )

    model_path: Path = Path("models/baseline_v0.pkl")
    representation: str = "combined_tfidf"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, parsed from the environment once."""
    return Settings()


# --------------------------------------------------------------------------- #
# Domain
# --------------------------------------------------------------------------- #


class TitleNotFoundError(LookupError):
    """Raised when a requested title has no match in the catalogue."""


@dataclass(frozen=True, slots=True)
class Recommender:
    """Cosine retrieval over one row-normalised representation.

    The matrix rows are L2-normalised by the training notebook, so cosine
    similarity is a plain dot product and no renormalisation happens per request.
    """

    matrix: sp.csr_matrix
    titles: tuple[str, ...]
    years: tuple[int | None, ...]
    exact: dict[str, int]
    popularity: tuple[float, ...]
    representation: str

    @classmethod
    def from_bundle(cls, bundle: dict[str, Any], representation: str) -> Self:
        """Build from the joblib bundle written by the baseline notebook."""
        available = bundle.get("representations", {})
        if representation not in available:
            msg = (
                f"representation {representation!r} not in bundle; "
                f"available: {sorted(available)}"
            )
            raise KeyError(msg)

        catalogue = bundle["catalogue"]
        titles = [str(t) for t in catalogue["title_clean"]]
        years = [None if v is None or v != v else int(v) for v in catalogue["release_year"]]

        # Titles are not unique. Where several films share one, the row with the
        # most ratings wins, which is what a user searching that title almost
        # always means: "The Dark Knight" should resolve to the 2008 film rather
        # than a direct-to-video animation of a similar name.
        weight = catalogue["ml_rating_count"].fillna(0).to_numpy()
        exact: dict[str, int] = {}
        for row, title in enumerate(titles):
            key = cls._normalise(title)
            best = exact.get(key)
            if best is None or weight[row] > weight[best]:
                exact[key] = row

        return cls(
            matrix=available[representation]["matrix"].tocsr(),
            titles=tuple(titles),
            years=tuple(years),
            exact=exact,
            popularity=tuple(float(w) for w in weight),
            representation=representation,
        )

    @staticmethod
    def _normalise(title: str) -> str:
        return " ".join(title.split()).casefold()

    def resolve(self, name: str) -> int:
        """Find the catalogue row for ``name``.

        Three passes, each ranked by rating count so the best-known film wins:

        1. Exact title match.
        2. Titles *starting* with the query.
        3. Titles containing the query as a whole word.

        Pass 3 is word-bounded rather than a bare ``in`` test. A plain substring
        search for "toy" also matches "destroy", and for "story" matches
        "history", which is how an earlier version answered a search for "Toy"
        with an unrelated film.
        """
        key = self._normalise(name)
        if (row := self.exact.get(key)) is not None:
            return row

        normalised = [self._normalise(t) for t in self.titles]
        prefix = [i for i, t in enumerate(normalised) if t.startswith(key)]
        if prefix:
            return max(prefix, key=lambda i: self.popularity[i])

        words = f" {key} "
        contained = [i for i, t in enumerate(normalised) if words in f" {t} "]
        if contained:
            return max(contained, key=lambda i: self.popularity[i])

        raise TitleNotFoundError(name)

    def recommend(self, row: int, top_k: int) -> list[tuple[str, float, int | None]]:
        """Return the ``top_k`` most similar films, highest score first."""
        product: Any = safe_sparse_dot(self.matrix, self.matrix[row].T)
        # np.asarray on a sparse result yields a 0-d object array, so sparse
        # output has to be densified explicitly.
        dense = product.toarray() if sp.issparse(product) else np.asarray(product)
        scores = np.asarray(dense, dtype=np.float64).ravel()
        scores[row] = -np.inf

        top_k = min(top_k, scores.size - 1)
        # argpartition is O(n) against O(n log n) for a full sort, which matters
        # at 25k items on every request.
        partition = np.argpartition(-scores, top_k)[:top_k]
        ordered = partition[np.argsort(-scores[partition])]
        return [(self.titles[i], float(scores[i]), self.years[i]) for i in ordered]


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class CamelModel(BaseModel):
    """Base schema: snake_case in Python, camelCase on the wire."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class RecommendationRequest(CamelModel):
    name: Annotated[str, Field(min_length=1, max_length=200, examples=["Toy Story"])]
    top_k: Annotated[int, Field(default=5, ge=1, le=50, examples=[5])]


class Recommendation(CamelModel):
    title: str
    score: Annotated[float, Field(ge=-1.0, le=1.0)]
    release_year: int | None = None


class RecommendationResponse(CamelModel):
    recs: list[Recommendation]


class HealthResponse(CamelModel):
    status: str
    representation: str
    catalogue_size: int


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Load the model once at startup rather than per request.

    The bundle is ~21 MB and deserialising it takes most of a second; doing that
    inside a request handler would dominate every response.
    """
    settings = get_settings()
    started = time.perf_counter()
    if not settings.model_path.is_file():
        msg = f"model bundle not found at {settings.model_path}"
        raise RuntimeError(msg)

    bundle = joblib.load(settings.model_path)
    app.state.recommender = Recommender.from_bundle(bundle, settings.representation)
    logger.info(
        "loaded %s (%d films, representation=%s) in %.2fs",
        settings.model_path,
        len(app.state.recommender.titles),
        settings.representation,
        time.perf_counter() - started,
    )
    yield
    app.state.recommender = None


app = FastAPI(
    title="Content-Based Movie Recommender",
    version="0.1.0",
    summary="Similar-film recommendations from the CST4275 v0 baseline model.",
    lifespan=lifespan,
)


def get_recommender(request: Request) -> Recommender:
    """Dependency handing the loaded model to a handler."""
    recommender: Recommender | None = getattr(request.app.state, "recommender", None)
    if recommender is None:  # pragma: no cover - only during startup or shutdown
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="model not loaded"
        )
    return recommender


RecommenderDep = Annotated[Recommender, Depends(get_recommender)]


@app.post("/recommendations", response_model=RecommendationResponse, tags=["recommendations"])
def recommendations(payload: RecommendationRequest, model: RecommenderDep) -> RecommendationResponse:
    """Recommend films similar to ``name``, ordered by descending similarity.

    Deliberately a sync ``def``: scoring is CPU-bound NumPy/SciPy work, and
    Starlette runs sync handlers in a threadpool. Declaring it ``async def``
    would block the event loop for the duration of every sparse matmul.
    """
    try:
        row = model.resolve(payload.name)
    except TitleNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no film matching {payload.name!r}",
        ) from exc

    hits = model.recommend(row, payload.top_k)
    return RecommendationResponse(
        recs=[
            Recommendation(title=title or _MISSING_TITLE, score=round(score, 6), release_year=year)
            for title, score, year in hits
        ]
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(model: RecommenderDep) -> HealthResponse:
    """Readiness probe. Returns 503 until the bundle has finished loading."""
    return HealthResponse(
        status="ok",
        representation=model.representation,
        catalogue_size=len(model.titles),
    )