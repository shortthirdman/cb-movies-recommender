# Content-Based Movie Recommender

> **MDX Dubai — CST4275 Research Project**  
> *A production-grade Content-Based Filtering system for movie recommendations,
> combining TF-IDF vectorisation, cosine similarity, IMDB weighted rating,
> sentence-transformer embeddings, and a demographic popularity chart.*

---

## Resources

- [javierruanohdez/world-cup-2026-prediction](https://github.com/javierruanohdez/world-cup-2026-prediction)
- [Hicruben/world-cup-2026-prediction-model](https://github.com/Hicruben/world-cup-2026-prediction-model)
- [kautzarichramsyah/worldcup2026-prediction](https://github.com/kautzarichramsyah/worldcup2026-prediction)
- [Content-Based Recommender Systems with Python](https://pub.aimind.so/content-based-recommender-systems-with-python-4b314926c769)

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Datasets](#3-datasets)
4. [Installation](#4-installation)
5. [Quick Start](#5-quick-start)
6. [CLI Reference](#6-cli-reference)
7. [Streamlit Web App](#7-streamlit-web-app)
8. [Jupyter Notebook](#8-jupyter-notebook)
9. [Recommendation Strategies](#9-recommendation-strategies)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Project Structure](#11-project-structure)
12. [Running Tests](#12-running-tests)
13. [Configuration](#13-configuration)

---

## 1. Overview

This system implements four complementary recommendation modules, directly addressing the MSc research proposal objectives:

| Module | Proposal §| Description |
|---|---|---|
| **Combined CB Recommender** | §1 | TF-IDF on full tag soup (genres + keywords + cast + director + overview + genome tags) |
| **Demographic Chart** | §2 | IMDB weighted-rating formula for cold-start / anonymous users |
| **Plot-Description Recommender** | §3 | TF-IDF on overview text only — pure narrative similarity |
| **Metadata Recommender** | §4 | TF-IDF on cast + director + genres + keywords — no overview text |
| **SBERT Semantic Encoder** | §5 | `all-MiniLM-L6-v2` sentence embeddings — captures paraphrase & semantic similarity |
| **Cold-Start Evaluation** | §7 | Metrics for movies with zero or very few votes |

**Evaluation metrics**: Precision@k, Recall@k, NDCG@k, Intra-List Diversity, Novelty, Genre Coverage, Catalogue Coverage.

---

## 2. System Architecture

```
Raw CSVs (5 sources)
      │
      ▼
┌─────────────────────────────┐
│  ingest/loaders.py          │  8 loaders — TMDB + MovieLens datasets
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  features/engineers.py      │  tag_soup / plot_soup / metadata_soup
│  features/merger.py         │  build_master() → movies_master.csv
└────────────┬────────────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
┌────────────┐  ┌─────────────────────┐
│CBRecom-    │  │DemographicRecommend-│
│mender      │  │er (IMDB WR formula) │
│3 strategies│  └─────────────────────┘
└────────────┘
      │
      ▼
┌─────────────────────────────┐
│  io/artefacts.py            │  joblib → models/*.pkl
└────────────┬────────────────┘
             │
    ┌────────┴───────────┐
    ▼                    ▼
┌───────┐         ┌────────────────┐
│  CLI  │         │ Streamlit App  │
│typer  │         │ 4-tab web GUI  │
└───────┘         └────────────────┘
```

---

## 3. Datasets

Five datasets are used, all sourced from Kaggle. They are **not** committed to the repository — download them with `make datasets` (requires a [Kaggle API key](https://www.kaggle.com/docs/api)).

| Dataset | Source | Used for |
|---|---|---|
| TMDB 5000 Movies | `tmdb/tmdb-movie-metadata` | Primary movie metadata, genres, keywords |
| TMDB 5000 Credits | `tmdb/tmdb-movie-metadata` | Cast and crew (director extraction) |
| The Movies Dataset | `rounakbanik/the-movies-dataset` | Extended metadata, `imdb_id`, `poster_path` |
| MovieLens 20M | `grouplens/movielens-20m-dataset` | Genome tags (content tags rated by users) |
| MovieLens Latest | auto-included | `links.csv` for TMDB ↔ MovieLens ID mapping |

```sh
make datasets   # downloads all three Kaggle packages into data/
```

---

## 4. Installation

**Requirements:** Python 3.12, [`uv`](https://docs.astral.sh/uv/) package manager.

```sh
# 1. Clone / open the project
cd workbench

# 2. Install all dependencies (production + dev)
uv sync

# 3. Verify the install
uv run recommender --help
```

> **Windows note:** prefix commands that print Unicode with  
> `$env:PYTHONIOENCODING="utf-8"; uv run ...`

---

## 5. Quick Start

### Step 1 — Train the model

```sh
# First run: builds movies_master.csv AND fits + saves all artefacts (~2–4 min)
uv run recommender train --force

# Subsequent runs: skip CSV rebuild, only refit models
uv run recommender train
```

This saves three artefacts to `models/`:
- `recommender.pkl` — fitted `CBRecommender` (combined + plot + metadata strategies)
- `demographic.pkl` — fitted `DemographicRecommender`
- `sbert_encoder.pkl` — fitted `SBERTEncoder` *(only if `--fit-sbert` is passed)*

### Step 2 — Get recommendations

```sh
uv run recommender recommend "Inception"
```

### Step 3 — Launch the web app

```sh
make app
# or equivalently:
uv run streamlit run app/streamlit_app.py --server.port 8501
```

---

## 6. CLI Reference

The CLI entry point is registered as `recommender` in [`pyproject.toml`](pyproject.toml).  
All subcommands support `--verbose` / `--help`.

---

### `train` — build master CSV and fit all models

```sh
uv run recommender train [OPTIONS]

Options:
  -f, --force        Force rebuild of movies_master.csv even if it exists
  -v, --vectoriser   Vectoriser type: 'tfidf' (default) or 'count'
  --fit-sbert        Also fit and save a SBERTEncoder (slow — ~5 min first run)
  --verbose          Enable DEBUG logging
```

**Examples:**

```sh
# Standard train (idempotent — skips CSV rebuild if it exists)
uv run recommender train

# Force full rebuild from raw CSVs
uv run recommender train --force

# Full rebuild + SBERT encoder
uv run recommender train --force --fit-sbert

# Use Count vectoriser instead of TF-IDF
uv run recommender train --vectoriser count
```

---

### `recommend` — get similar movies for a title

```sh
uv run recommender recommend TITLE [OPTIONS]

Arguments:
  TITLE              Movie title (fuzzy-matched, case-insensitive)

Options:
  -n, --n            Number of recommendations (default: 10)
  -s, --strategy     Recommendation strategy: combined (default), plot, metadata
  --verbose
```

**Examples:**

```sh
# Default — combined strategy, top 10
uv run recommender recommend "Inception"

# Plot-description strategy (overview text only)
uv run recommender recommend "Inception" --strategy plot

# Metadata strategy (cast/director/genres — no overview)
uv run recommender recommend "Inception" --strategy metadata

# Top 20 with metadata strategy
uv run recommender recommend "The Dark Knight" -n 20 --strategy metadata

# Fuzzy matching — handles typos
uv run recommender recommend "incepshun"
```

**Output:** A Rich-formatted table with rank, title, similarity score, weighted rating, genres, and release year.

---

### `chart` — demographic popularity chart

```sh
uv run recommender chart [OPTIONS]

Options:
  -n, --n            Number of films to show (default: 20)
  -g, --genre        Filter by genre, e.g. 'Action', 'Drama'
  -l, --language     Filter by ISO language code, e.g. 'en', 'fr'
  --min-year         Minimum release year (inclusive)
  --max-year         Maximum release year (inclusive)
  --verbose
```

**Examples:**

```sh
# Global top 20 by weighted rating
uv run recommender chart

# Top 10 Action films
uv run recommender chart -n 10 --genre Action

# Top 15 English Drama films released 2000–2020
uv run recommender chart -n 15 --genre Drama --language en --min-year 2000 --max-year 2020

# Top 25 films from the 1990s
uv run recommender chart -n 25 --min-year 1990 --max-year 1999
```

**Output:** A Rich-formatted table with rank, title, weighted rating, vote count, average rating, genres, and year.

---

### `evaluate` — run evaluation metrics

```sh
uv run recommender evaluate [OPTIONS]

Options:
  -t, --titles       Query title (repeat for multiple: --titles Avatar --titles Inception)
  --k                Cutoff k for all metrics (default: 10)
  -o, --output       Path to save evaluation CSV report
  --verbose
```

**Examples:**

```sh
# Default evaluation on 5 representative titles at k=10
uv run recommender evaluate

# Custom query set
uv run recommender evaluate \
  --titles "Avatar" --titles "Inception" --titles "Toy Story" \
  --k 15

# Save results to CSV
uv run recommender evaluate --output results/eval_report.csv
```

---

## 7. Streamlit Web App

The Streamlit app provides a full interactive GUI with four tabs.

### Launch

```sh
make app
# or
uv run streamlit run app/streamlit_app.py --server.port 8501
```

Then open **http://localhost:8501** in your browser.

---

### Tab 1 — 🎯 Recommend

- **Autocomplete search** — select from all ~4,800 movie titles
- **Strategy selector** — choose `combined`, `plot`, or `metadata`
- **Number of results** slider (5–20)
- Results displayed as a Netflix-style card grid with TMDB posters
- Expandable full results table

### Tab 2 — 📈 Demographic Charts

- IMDB weighted-rating popularity chart
- Filterable by **genre**, **language** (ISO code), **min/max release year**
- Interactive Plotly bar chart + sortable table
- CSV download button

### Tab 3 — 📊 Evaluate

- Enter query titles (one per line) and set cutoff k
- Computes all 7 metrics: P@k, R@k, NDCG@k, ILD, Novelty, Genre Coverage, Catalogue Coverage
- Results table with blue gradient highlighting
- Radar (spider) chart of aggregate metrics
- CSV download of full report

### Tab 4 — 🔍 EDA

- Summary statistics (movie count, mean weighted score, language count)
- Top-20 genre distribution bar chart
- IMDB weighted-score distribution histogram
- Movies-per-year release trend line
- Top 20 movies by weighted score
- Tag-soup word cloud

### Sidebar Controls

- **Model status indicator** — shows whether artefacts are trained
- **Train / Rebuild Model** button — runs the full pipeline from the browser
- **Force rebuild master CSV** checkbox
- **Also fit SBERT encoder** checkbox (slow — adds ~5 min)

---

## 8. Jupyter Notebook

A full demo notebook is available at [`notebooks/CB_Recommender_Demo.ipynb`](notebooks/CB_Recommender_Demo.ipynb).

### Launch JupyterLab

```sh
make lab
# or
uv run jupyter lab --notebook-dir=./notebooks --no-browser
```

Then open the URL printed in the terminal (e.g. `http://localhost:8888/lab`).

### Notebook Sections

| Section | Content |
|---|---|
| 1 | Setup — imports, display config, global flags |
| 2 | Pipeline execution (`run_pipeline`) |
| 3 | Data overview — shape, dtypes, missing values |
| 4 | EDA — genre distribution, vote distribution, release year trend |
| 5 | Feature engineering — tag soup inspection, word cloud |
| 6 | Model mechanics — vectoriser stats, similarity matrix distribution |
| 7 | Sample recommendations — combined strategy |
| 7b | **Three-strategy comparison** — side-by-side combined / plot / metadata |
| 7c | **Demographic chart** — genre-filtered popularity charts |
| 7d | **SBERT semantic comparison** — TF-IDF vs sentence-transformer results |
| 8 | Evaluation — full metrics report + radar chart |
| 8b | **Cold-start experiment** — evaluation on zero-vote films |
| 9 | Discussion & limitations |

---

## 9. Recommendation Strategies

All three strategies are fitted in a single `CBRecommender.fit()` call and served from independent TF-IDF + cosine similarity matrices.

| Strategy | `--strategy` value | Soup column | What it encodes |
|---|---|---|---|
| **Combined** *(default)* | `combined` | `tag_soup` | Genres + keywords + cast + director + overview stems + genome tags |
| **Plot-Description** | `plot` | `plot_soup` | Overview text only (narrative similarity) |
| **Metadata** | `metadata` | `metadata_soup` | Cast + director + genres + keywords (no overview) |

### Demographic Recommender (separate module)

Uses the IMDB weighted-rating formula:

```
WR = (v / (v + m)) × R  +  (m / (v + m)) × C
```

where `v` = movie vote count, `m` = 90th-percentile vote threshold,
`R` = movie average rating, `C` = corpus mean rating.

### SBERT Encoder (optional)

Uses [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384-dimensional dense embeddings).  
Requires `uv run recommender train --fit-sbert` or an explicit `SBERTEncoder().fit(master_df)` call.

---

## 10. Evaluation Metrics

| Metric | Symbol | Description |
|---|---|---|
| Precision@k | P@k | Fraction of top-k recommendations sharing a genre with the query |
| Recall@k | R@k | Fraction of all genre-relevant corpus movies retrieved in top-k |
| NDCG@k | — | Normalised Discounted Cumulative Gain (graded by cosine similarity score) |
| Intra-List Diversity | ILD | Average pairwise dissimilarity within a recommendation list |
| Novelty | — | Mean log-popularity rank (higher = more niche) |
| Genre Coverage | — | Fraction of all corpus genres represented in the recommendation list |
| Catalogue Coverage | — | Fraction of all movies appearing in at least one recommendation list |

> **Relevance proxy:** genre-overlap is used as the relevance signal since no explicit user ratings are available in a CB-only system. This is the standard academic convention — see Shani & Gunawardana (2011).

---

## 11. Project Structure

```
workbench/
├── app/
│   └── streamlit_app.py          # 4-tab Streamlit web GUI
├── data/
│   ├── tmdb-movie-metadata/      # TMDB 5000 movies + credits CSVs
│   ├── the-movies-dataset/       # Extended metadata, keywords, links
│   ├── movielens-20m-dataset/    # Genome scores + tags
│   ├── movielens-latest/         # links.csv (ID mapping)
│   └── processed/
│       └── movies_master.csv     # Built by run_pipeline() — not committed
├── models/
│   ├── recommender.pkl           # Fitted CBRecommender — not committed
│   ├── demographic.pkl           # Fitted DemographicRecommender — not committed
│   └── sbert_encoder.pkl         # Fitted SBERTEncoder (optional) — not committed
├── notebooks/
│   ├── CB_Recommender_Demo.ipynb # Full end-to-end demo notebook
│   └── Main.ipynb                # Primary research workbench notebook
├── src/
│   └── recommender/
│       ├── __init__.py           # MovieNotFoundError, __version__
│       ├── config.py             # Centralised paths + hyperparameters
│       ├── cli.py                # Typer CLI (train/recommend/chart/evaluate)
│       ├── ingest/
│       │   ├── loaders.py        # 8 CSV loaders
│       │   └── schemas.py        # Column name constants
│       ├── features/
│       │   ├── parsers.py        # safe_literal_eval, extract_names, extract_director
│       │   ├── engineers.py      # tag_soup / plot_soup / metadata_soup builders
│       │   └── merger.py         # build_master() → movies_master.csv
│       ├── model/
│       │   ├── vectoriser.py     # TF-IDF + Count matrix builders
│       │   ├── similarity.py     # linear_kernel cosine similarity
│       │   ├── recommender.py    # CBRecommender (3 strategies)
│       │   ├── demographic.py    # DemographicRecommender (IMDB WR formula)
│       │   └── encoder.py        # SBERTEncoder (sentence-transformers)
│       ├── pipeline/
│       │   └── train.py          # run_pipeline() orchestrator
│       ├── evaluation/
│       │   ├── metrics.py        # 7 metrics + cold_start_experiment()
│       │   └── report.py         # CSV + radar chart generation
│       └── io/
│           └── artefacts.py      # joblib save/load wrappers
├── tests/
│   ├── test_features.py          # 30 tests — parsers & engineers
│   ├── test_loaders.py           # 10 tests — CSV loaders
│   ├── test_metrics.py           # 35 tests — evaluation metrics
│   ├── test_pipeline.py          #  8 tests — artefact I/O + pipeline
│   └── test_recommender.py       # 37 tests — CBRecommender + DemographicRecommender
├── Makefile                      # make sync / lab / app / datasets
├── pyproject.toml                # uv project manifest + dependencies
└── uv.lock                       # locked dependency graph
```

---

## 12. Running Tests

```sh
# Run the full test suite (120 tests)
$env:PYTHONIOENCODING="utf-8"; uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_recommender.py -v

# Run a single test class
uv run pytest tests/test_recommender.py::TestStrategies -v

# Run a single test by name
uv run pytest tests/test_recommender.py::TestStrategies::test_strategy_plot_returns_dataframe -v

# Run with short tracebacks
uv run pytest tests/ --tb=short -q
```

**Expected result:** `120 passed` — no real data files are read; all tests use synthetic in-memory DataFrames.

---

## 13. Configuration

All paths and hyperparameters live in [`src/recommender/config.py`](src/recommender/config.py) — never hardcoded elsewhere.

| Constant | Default | Description |
|---|---|---|
| `TFIDF_MAX_FEATURES` | `50_000` | Maximum TF-IDF vocabulary size |
| `TOP_CAST_N` | `3` | Number of cast members included in tag soup |
| `WEIGHTED_RATING_PERCENTILE` | `0.90` | Vote-count percentile threshold for demographic chart |
| `GENOME_TOP_N_TAGS` | `10` | Number of MovieLens genome tags per movie |
| `SBERT_MODEL_NAME` | `"all-MiniLM-L6-v2"` | Sentence-transformer model identifier |
| `DEMOGRAPHIC_CHART_SIZE` | `250` | Maximum size of the pre-built weighted-rating chart |

Paths resolve relative to the package location (`src/recommender/config.py → project root`), so the package works correctly regardless of the current working directory.

---

## Makefile Targets

```sh
make sync       # uv sync — install all deps
make lab        # start JupyterLab at ./notebooks (no browser)
make app        # launch Streamlit on port 8501
make datasets   # download all Kaggle datasets (requires Kaggle API key)
```

---

*MDX Dubai — CST4275 Research Project — M01087647*
