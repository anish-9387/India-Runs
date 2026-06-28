CANDIDATES ?= ./candidates.jsonl
OUT ?= submission.csv

.PHONY: install rank validate test eda sandbox embeddings clean

install:
	poetry install --no-root

rank:
	poetry run python rank.py --candidates $(CANDIDATES) --out $(OUT)

validate:
	poetry run python validate_submission.py $(OUT)

test:
	poetry run python -m pytest tests/ -q

eda:
	poetry run python scripts/eda.py --candidates $(CANDIDATES)

embeddings:
	poetry run python scripts/precompute_embeddings.py --candidates $(CANDIDATES) \
		--out artifacts/dense_embeddings.npz

sandbox:
	poetry run streamlit run sandbox/app.py

clean:
	poetry run python -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
	rm -f $(OUT)
