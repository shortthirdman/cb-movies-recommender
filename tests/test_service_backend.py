"""Tests for the HTTP backend, without needing a running service.

``requests.post`` is replaced with a stub so the contract with the FastAPI
service - the camelCase payload it sends, the camelCase response it reads, and
how it turns status codes into exceptions - is pinned down in isolation.
"""

from __future__ import annotations

from typing import Any

import pytest
import requests

from app.backends import (
    UNKNOWN_COUNTRY,
    BackendUnavailableError,
    ServiceBackend,
    TitleNotFoundError,
)

COUNTRIES = {
    ("toy story toons: hawaiian vacation", 2011): "United States of America",
    ("hot rod huckster", 1954): "United States of America",
}

PAYLOAD = {
    "recs": [
        {"title": "Toy Story Toons: Hawaiian Vacation", "score": 0.233076, "releaseYear": 2011},
        {"title": "Hot Rod Huckster", "score": 0.192649, "releaseYear": 1954},
        {"title": "A Film With No Country Row", "score": 0.1, "releaseYear": 1999},
    ]
}


class StubResponse:
    def __init__(self, status: int = 200, payload: Any = None, text: str = "") -> None:
        self.status_code = status
        self._payload = payload if payload is not None else PAYLOAD
        self.text = text

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    def json(self) -> Any:
        return self._payload


@pytest.fixture
def backend() -> ServiceBackend:
    return ServiceBackend(base_url="http://localhost:8000/", countries=COUNTRIES)


class TestRequest:
    def test_posts_camel_case_to_the_documented_path(
        self, backend: ServiceBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        def fake_post(url: str, **kwargs: Any) -> StubResponse:
            seen["url"] = url
            seen.update(kwargs)
            return StubResponse()

        monkeypatch.setattr(requests, "post", fake_post)
        backend.suggest("Toy Story", 3)

        # The trailing slash on base_url must not produce a doubled path.
        assert seen["url"] == "http://localhost:8000/recommendations"
        assert seen["json"] == {"name": "Toy Story", "topK": 3}
        assert seen["timeout"] > 0, "a request without a timeout can hang the app"


class TestResponse:
    def test_maps_the_response_onto_suggestions(
        self, backend: ServiceBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(requests, "post", lambda *a, **k: StubResponse())
        result = backend.suggest("Toy Story", 3)

        assert result.seed == "Toy Story"
        assert len(result.suggestions) == 3
        first = result.suggestions[0]
        assert first.title == "Toy Story Toons: Hawaiian Vacation"
        assert first.score == pytest.approx(0.233076)
        assert first.release_year == 2011
        assert first.country == "United States of America"

    def test_unmatched_films_get_the_placeholder_country(
        self, backend: ServiceBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(requests, "post", lambda *a, **k: StubResponse())
        result = backend.suggest("Toy Story", 3)
        assert result.suggestions[-1].country == UNKNOWN_COUNTRY

    def test_empty_recs_is_not_an_error(
        self, backend: ServiceBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(requests, "post", lambda *a, **k: StubResponse(payload={"recs": []}))
        assert backend.suggest("Toy Story", 3).suggestions == ()


class TestFailures:
    def test_404_becomes_title_not_found(
        self, backend: ServiceBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(requests, "post", lambda *a, **k: StubResponse(status=404))
        with pytest.raises(TitleNotFoundError):
            backend.suggest("Nonexistent", 3)

    @pytest.mark.parametrize("status", [500, 503, 422])
    def test_other_error_statuses_become_backend_unavailable(
        self, backend: ServiceBackend, monkeypatch: pytest.MonkeyPatch, status: int
    ) -> None:
        monkeypatch.setattr(
            requests, "post", lambda *a, **k: StubResponse(status=status, text="boom")
        )
        with pytest.raises(BackendUnavailableError, match=str(status)):
            backend.suggest("Toy Story", 3)

    @pytest.mark.parametrize(
        "error",
        [requests.ConnectionError("refused"), requests.Timeout("slow")],
    )
    def test_transport_errors_become_backend_unavailable(
        self, backend: ServiceBackend, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        def raise_it(*_: Any, **__: Any) -> None:
            raise error

        monkeypatch.setattr(requests, "post", raise_it)
        # The message must name the URL, because "it did not answer" is useless
        # to a user who has typed the wrong port into the sidebar.
        with pytest.raises(BackendUnavailableError, match="recommendations"):
            backend.suggest("Toy Story", 3)
