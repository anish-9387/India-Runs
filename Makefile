CANDIDATES ?= ./candidates.jsonl
OUT ?= submission.csv

.PHONY: install rank validate test eda sandbox embeddings clean

install:
	pip install -r requirements.txt

rank:
	python rank.py --candidates $(CANDIDATES) --out $(OUT)

validate:
	python validate_submission.py $(OUT)

test:
	python -m pytest tests/ -q

eda:
	python scripts/eda.py --candidates $(CANDIDATES)

embeddings:
	python scripts/precompute_embeddings.py --candidates $(CANDIDATES) \
		--out artifacts/dense_embeddings.npz

sandbox:
	streamlit run sandbox/app.py

clean:
	rm -f $(OUT)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
