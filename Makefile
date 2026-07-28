.PHONY: sync lab marimo dev-deps deps datasets app

# Sync the project and install dependencies from pyproject.toml
sync:
	uv sync

dev-deps:
	uv add --dev ipywidgets jedi-language-server jupyter jupyterlab jupyter-contrib-nbextensions jupyter-lsp jupyter-nbextensions-configurator jupyterlab-lsp nbconvert notebook

deps:
	uv add python-dotenv numpy pandas polars scikit-learn torch statsmodels wordcloud tensorflow hmmlearn matplotlib seaborn scipy plotly nltk spacy stanza sentence-transformers

datasets:
	kaggle datasets download grouplens/movielens-20m-dataset
	kaggle datasets download tmdb/tmdb-movie-metadata
	kaggle datasets download asaniczka/tmdb-movies-dataset-2023-930k-movies

# Run the Jupyter Lab server using the activated virtual environment
lab:
	uv run jupyter lab --notebook-dir=./notebooks --no-browser

# Run the Marimo editor (if you opt for the Marimo stack)
marimo:
	uv run marimo edit ./notebooks

# Run the Streamlit web GUI (requires trained model artefacts)
app:
	uv run streamlit run app/streamlit_app.py --server.port 8501