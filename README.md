# Content-Based Movie Recommender

> **MDX Dubai — CST4275 Research Project**  
> *A production-grade Content-Based Filtering system for movie recommendations,
> combining TF-IDF vectorisation, cosine similarity, IMDB weighted rating,
> sentence-transformer embeddings, and a demographic popularity chart.*

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


*MDX Dubai — CST4275 Research Project — M01087647*