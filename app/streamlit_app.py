"""Streamlit front end for the content-based movie recommender.

Two pages:

* **Ask for a recommendation** - a chat interface over the baseline model.
* **Data profiling** - a ydata-profiling report over the merged catalogue.

Run from the project root::

    streamlit run app/streamlit_app.py

Configuration is environment-driven and shares the ``RECSYS_`` prefix with the
FastAPI service, so a single ``.env`` file serves both::

    RECSYS_MODEL_PATH=models/baseline_v0/baseline_v0.joblib
    RECSYS_REPRESENTATION=combined_tfidf
    RECSYS_SERVICE_URL=http://127.0.0.1:8000
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
# Streamlit puts the script's own directory on sys.path, not the project root,
# so `import app.*` needs the root added before anything else is imported.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
from pydantic_settings import BaseSettings, SettingsConfigDict  # noqa: E402

from app import ui_chat, ui_profile  # noqa: E402
from app.backends import (  # noqa: E402
    Backend,
    BackendUnavailableError,
    LocalBackend,
    ServiceBackend,
    build_country_index,
    load_recommender,
)

# Well-known titles present in the model's 25,000-film evaluation sample.
EXAMPLES: Final = ("Toy Story", "Alien", "The Lion King", "Reservoir Dogs")


class AppSettings(BaseSettings):
    """Runtime configuration, overridable from the environment or a ``.env`` file."""

    # protected_namespaces=() is required: Pydantic reserves the `model_` prefix
    # and `model_path` would otherwise emit a shadowing warning.
    model_config = SettingsConfigDict(
        env_prefix="RECSYS_", env_file=".env", extra="ignore", protected_namespaces=()
    )

    model_path: Path = ROOT / "models" / "baseline_v0" / "baseline_v0.joblib"
    merged_csv: Path = ROOT / "datasets" / "processed" / "movies_merged.csv"
    representation: str = "combined_tfidf"
    service_url: str = "http://127.0.0.1:8000"


@st.cache_resource(show_spinner="Loading the baseline model…")
def _recommender(model_path: Path, representation: str):
    """Load the model once per process.

    ``cache_resource`` hands every session the same object rather than a copy,
    which is only safe because ``Recommender`` is a frozen dataclass whose
    ``recommend`` mutates nothing but the arrays it derives itself.
    """
    return load_recommender(model_path, representation)


@st.cache_data(show_spinner="Indexing production countries…")
def _countries(csv_path: Path, mtime: float) -> dict[tuple[str, int | None], str]:
    """Build the country lookup once.

    ``mtime`` is unused in the body and present only as a cache key, so
    rebuilding the dataset invalidates this entry instead of serving countries
    from the previous build.
    """
    return build_country_index(csv_path)


def _sidebar(settings: AppSettings) -> Backend | None:
    """Draw the shared sidebar and return the backend the user selected."""
    st.sidebar.title("Recommender")
    st.sidebar.caption("CST4275 · Content-based movie recommendation system")

    countries = (
        _countries(settings.merged_csv, settings.merged_csv.stat().st_mtime)
        if settings.merged_csv.is_file()
        else {}
    )

    choice = st.sidebar.radio(
        "Recommendation source",
        ("In-process model", "FastAPI service"),
        help=(
            "The in-process model deserialises the joblib bundle into this "
            "process. The service option calls the same model over HTTP and "
            "needs `uvicorn app.api:app` running."
        ),
    )

    if choice == "FastAPI service":
        url = st.sidebar.text_input("Service base URL", value=settings.service_url)
        st.sidebar.caption(f"POSTs to `{url.rstrip('/')}/recommendations`")
        return ServiceBackend(base_url=url, countries=countries)

    try:
        recommender = _recommender(settings.model_path, settings.representation)
    except BackendUnavailableError as exc:
        st.sidebar.error(str(exc), icon=":material/error:")
        return None

    st.sidebar.success(
        f"{len(recommender.titles):,} films · `{recommender.representation}`",
        icon=":material/check_circle:",
    )
    if not countries:
        st.sidebar.warning(
            "Merged catalogue not found, so country will show as “—”.",
            icon=":material/warning:",
        )
    return LocalBackend(recommender=recommender, countries=countries)


def main() -> None:
    st.set_page_config(
        page_title="Movie recommender",
        page_icon=":material/movie:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    settings = AppSettings()
    backend = _sidebar(settings)

    def chat_page() -> None:
        if backend is None:
            st.title("Ask for a recommendation")
            st.error(
                f"No model at `{settings.model_path}`. Run "
                "`notebooks/03_baseline_model.ipynb` to produce it, set "
                "`RECSYS_MODEL_PATH`, or switch to the FastAPI service in the sidebar.",
                icon=":material/error:",
            )
            return
        ui_chat.render(backend, EXAMPLES)

    def profile_page() -> None:
        ui_profile.render(settings.merged_csv)

    # st.navigation is the current multipage API and must be called exactly once,
    # from the entrypoint. Using callables rather than a `pages/` directory keeps
    # the sidebar above under this file's control and lets each page receive its
    # dependencies as arguments instead of reaching for globals.
    navigation = st.navigation(
        [
            st.Page(
                chat_page,
                title="Ask for a recommendation",
                icon=":material/chat:",
                url_path="chat",
                default=True,
            ),
            st.Page(
                profile_page,
                title="Data profiling",
                icon=":material/analytics:",
                url_path="profiling",
            ),
        ]
    )
    navigation.run()


main()
