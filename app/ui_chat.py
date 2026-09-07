"""Chat page: ask for films in plain language, get recommendations back.

The conversation is driven by :mod:`app.intent`, which parses a message into a
title and a result count without calling a language model. Every film named in
a reply comes from the recommender, so the page cannot invent a title that is
not in the catalogue.
"""

from __future__ import annotations

from typing import Any, Final

import pandas as pd
import streamlit as st

from app.backends import (
    Backend,
    BackendUnavailableError,
    Result,
    TitleNotFoundError,
    suggest_first_match,
)
from app.intent import Kind, parse

__all__ = ["render"]

_HISTORY_KEY: Final = "chat_history"

_GREETING: Final = (
    "Hello. Name a film and I will find others like it. "
    'Try *"movies like Toy Story"* or *"show me 8 films similar to Alien"*.'
)

_HELP: Final = """
Ask for films similar to one you already know. These all work:

- `Toy Story`
- `movies like The Matrix`
- `recommend 8 films similar to Alien`
- `what should I watch after Se7en?`
- `anything like "Get Out"`

I read a title and a result count from your message. Put the title in quotes if
it contains words such as *like* or *after* that I might otherwise mistake for
part of the question. Results come from the content-based model, ranked by
cosine similarity, so every film listed is one the model actually holds.
"""


def _result_frame(result: Result) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Film": [s.title for s in result.suggestions],
            "Match": [s.score for s in result.suggestions],
            "Country": [s.country for s in result.suggestions],
            "Year": [s.release_year for s in result.suggestions],
        }
    )


def _render_result(result: Result) -> None:
    """Draw one set of recommendations as a table."""
    st.dataframe(
        _result_frame(result),
        hide_index=True,
        width="stretch",
        column_config={
            "Film": st.column_config.TextColumn(width="large"),
            # The score is a cosine similarity, already bounded to [0, 1] by the
            # L2-normalised representation, so it can be shown as a proportion
            # without any rescaling.
            "Match": st.column_config.ProgressColumn(
                "Match", format="%.3f", min_value=0.0, max_value=1.0
            ),
            "Country": st.column_config.TextColumn(width="medium"),
            "Year": st.column_config.NumberColumn(format="%d"),
        },
    )


def _render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if (result := message.get("result")) is not None:
            _render_result(result)


def _answer(backend: Backend, prompt: str) -> dict[str, Any]:
    """Turn one user message into an assistant message."""
    intent = parse(prompt)

    if intent.kind is Kind.GREETING:
        return {"role": "assistant", "content": _GREETING}
    if intent.kind is Kind.HELP:
        return {"role": "assistant", "content": _HELP}

    try:
        result = suggest_first_match(backend, intent.candidates, intent.top_k)
    except TitleNotFoundError:
        tried = ", ".join(f"*{c}*" for c in intent.candidates)
        return {
            "role": "assistant",
            "content": (
                f"I could not find that film in the catalogue. I looked for {tried}. "
                "The model holds a 25,000-film sample of the full catalogue, so a "
                "title can be missing simply because it fell outside that sample. "
                "Check the spelling, or put the title in quotes."
            ),
        }
    except BackendUnavailableError as exc:
        return {
            "role": "assistant",
            "content": (
                f"The **{backend.label}** backend did not answer.\n\n```\n{exc}\n```\n\n"
                "If you are using the FastAPI backend, start it with "
                "`uvicorn app.api:app --reload` and check the base URL in the sidebar."
            ),
        }

    if not result.suggestions:
        return {
            "role": "assistant",
            "content": f"I found **{result.seed}** but the model returned nothing similar.",
        }

    count = len(result.suggestions)
    plural = "film" if count == 1 else "films"
    return {
        "role": "assistant",
        "content": f"{count} {plural} similar to **{result.seed}**, most similar first:",
        "result": result,
    }


def render(backend: Backend, examples: tuple[str, ...]) -> None:
    """Draw the chat page."""
    st.title("Ask for a recommendation")
    st.caption(
        "Name a film you like. Titles and scores come from the content-based "
        "baseline model; country and release year are joined from the merged "
        "catalogue."
    )

    if _HISTORY_KEY not in st.session_state:
        st.session_state[_HISTORY_KEY] = [{"role": "assistant", "content": _GREETING}]

    # Example buttons write into the same queue as the chat box, so both paths
    # take exactly one code path through _answer.
    queued: str | None = None
    st.write("Try one of these:")
    for column, example in zip(st.columns(len(examples)), examples, strict=True):
        if column.button(example, width="stretch", key=f"example-{example}"):
            queued = f"movies like {example}"

    for message in st.session_state[_HISTORY_KEY]:
        _render_message(message)

    prompt = st.chat_input("Name a film, for example: movies like Toy Story") or queued
    if not prompt:
        return

    user_message = {"role": "user", "content": prompt}
    st.session_state[_HISTORY_KEY].append(user_message)
    _render_message(user_message)

    with st.spinner("Scoring the catalogue…"):
        reply = _answer(backend, prompt)
    st.session_state[_HISTORY_KEY].append(reply)
    _render_message(reply)
