.PHONY: sync lab marimo dev-deps deps app api profiling-ext test lint

# Sync the project and install dependencies from pyproject.toml
sync:
	uv sync

dev-deps:
	uv add --dev ipywidgets jedi-language-server jupyter jupyterlab jupyter-contrib-nbextensions jupyter-lsp jupyter-nbextensions-configurator jupyterlab-lsp nbconvert notebook

deps:
	uv add python-dotenv numpy pandas polars scikit-learn torch statsmodels wordcloud tensorflow hmmlearn matplotlib seaborn scipy plotly nltk spacy stanza sentence-transformers 

# Run the Jupyter Lab server using the activated virtual environment
lab:
	jupyter lab --notebook-dir=./notebooks --no-browser

# Run the Marimo editor (if you opt for the Marimo stack)
marimo:
	marimo edit ./notebooks

# Run the Streamlit web GUI (requires trained model artefacts)
app: app/streamlit_app.py
	streamlit run app/streamlit_app.py --server.port 8501

api: app/api.py
	uvicorn app.api:app --reload

# Install the Streamlit profiling component. Kept out of requirements.txt because
# it declares Requires-Python <3.12 and pins watchdog<4; --no-deps leaves the
# project's own streamlit and ydata-profiling pins untouched.
profiling-ext:
	pip install --no-deps --ignore-requires-python streamlit-ydata-profiling==0.2.1
	uv pip install --no-deps --ignore-requires-python streamlit-ydata-profiling==0.2.1

test:
	pytest tests -q

lint:
	ruff check app tests