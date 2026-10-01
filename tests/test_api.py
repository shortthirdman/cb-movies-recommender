"""Tests for the ``model`` query parameter on the FastAPI service.

The id-matching check runs anywhere. The endpoint tests run against the real
artefacts and are skipped when they are absent, as in the integration suite.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import Recommender, app, get_settings

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "models" / "baseline_v0" / "baseline_v0.joblib"
TF_DIR = ROOT / "models" / "tf_v1"

# Printed by 04_tensorflow_based_recomm.ipynb, section 6. The service returning
# the same list proves the embeddings were matched to the right titles.
TOY_STORY_TF = [
    "Sense and Sensibility",
    "Muppet Treasure Island",
    "Ransom",
    "Beauty and the Beast",
    "Phenomenon",
    "Matilda",
]


def _catalogue(ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movie_id": ids,
            "title_clean": [f"film {i}" for i in ids],
            "release_year": [2000] * len(ids),
            "ml_rating_count": [1.0] * len(ids),
        }
    )


def test_embeddings_are_matched_to_titles_by_movie_id() -> None:
    model = Recommender.from_embeddings(
        np.eye(3, dtype=np.float32), np.array([30, 10, 20]), _catalogue([10, 20, 30])
    )
    assert model.titles == ("film 30", "film 10", "film 20")


def test_mismatched_ids_fail_loudly() -> None:
    with pytest.raises(ValueError, match="do not match"):
        Recommender.from_embeddings(np.eye(2), np.array([1, 2]), _catalogue([1, 3]))


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    if not BASELINE.is_file() or not (TF_DIR / "item_embeddings.npz").is_file():
        pytest.skip("model artefacts not present")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("RECSYS_MODEL_PATH", str(BASELINE))
        patch.setenv("RECSYS_TF_MODEL_DIR", str(TF_DIR))
        get_settings.cache_clear()
        with TestClient(app) as test_client:
            yield test_client
    get_settings.cache_clear()


def _titles(client: TestClient, query: str = "") -> list[str]:
    response = client.post(f"/recommendations{query}", json={"name": "Toy Story", "topK": 6})
    assert response.status_code == 200, response.text
    return [rec["title"] for rec in response.json()["recs"]]


def test_omitting_the_parameter_serves_the_baseline(client: TestClient) -> None:
    assert _titles(client) == _titles(client, "?model=v0")


def test_tf_model_reproduces_the_notebook(client: TestClient) -> None:
    assert _titles(client, "?model=tf") == TOY_STORY_TF


def test_unknown_model_is_rejected(client: TestClient) -> None:
    response = client.post("/recommendations?model=v2", json={"name": "Toy Story"})
    assert response.status_code == 422


def test_health_reports_the_requested_model(client: TestClient) -> None:
    body = client.get("/health?model=tf").json()
    assert body == {"status": "ok", "representation": "tf_two_tower", "catalogueSize": 25_000}
