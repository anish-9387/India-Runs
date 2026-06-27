# Redrob Candidate Ranker — India Runs Data & AI Challenge

A CPU-only, fully-offline system that ranks the **top 100** of **100,000**
candidates against the *Senior AI Engineer — Founding Team* job description, with
a 1–2 sentence **grounded, hallucination-free** justification per candidate.

> **Full 100k run: ~52 s wall-clock, ~1.5 GB RAM, 0 honeypots in the top-100.**
> Well inside the challenge budget (≤ 5 min, ≤ 16 GB, CPU-only, no network).

---

## Reproduce the submission (single command)

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

`--candidates` accepts `candidates.jsonl` or `candidates.jsonl.gz`. Validate the
output with the organizer's validator:

```bash
python validate_submission.py submission.csv      # -> "Submission is valid."
```

Run the tests:

```bash
python -m pytest tests/ -q        # or: python tests/test_smoke.py
```

---

## The problem (and the trap)

The dataset is **adversarial by design**. The biggest trap: candidates who stuff
their *skills list* with AI keywords ("RAG", "Pinecone", "Embeddings") while
their actual job is Marketing Manager / HR / Accountant. The JD is explicit:

> *"A candidate who has all the AI keywords listed as skills but whose title is
> 'Marketing Manager' is not a fit, no matter how perfect their skill list looks."*

There are **no relevance labels** in the data, so this is an **unsupervised,
rules-encoded ranking** problem — the "model" is a transparent, defensible
scoring function, not a trained ranker. The design choices below are what we
defend at the Stage-5 interview.

## Approach (how we beat the trap)

A **hybrid, multi-signal scorer**:

```
final = (0.62·structured_fit + 0.38·semantic) · behavioral_modifier · honeypot_gate
```

| Channel | What it does | Why |
|---|---|---|
| **structured_fit** | reads the **title** and the **free-text career descriptions** (not the skills array) | an HR Manager with 9 AI skills scores ~0; only real retrieval/ranking work in the *descriptions* earns the domain score |
| **semantic** | TF-IDF cosine of the candidate text to a JD "ideal-profile" query, optionally RRF-fused with dense embeddings | catches "plain-language Tier-5s" who built a recommender without using buzzwords |
| **behavioral_modifier** | multiplicative ∈ [0.55, 1.12] from the 23 `redrob_signals` | a perfect-on-paper candidate who is inactive / unresponsive is down-weighted, not zeroed |
| **honeypot_gate** | sinks internally-impossible profiles | keeps the top-100 honeypot rate at 0% (Stage-3 DQ is > 10%) |

### `structured_fit` components (weights in [config.py](src/redrob_ranker/config.py))

`role` 0.32 · `domain` 0.30 · `product` 0.12 · `experience` 0.10 · `location` 0.10 · `external` 0.06,
minus penalties for the JD's explicit *do-NOT-wants* (CV/speech/robotics-only,
entire-career-at-services-firms, off-role title with a stuffed skill list).

- **role** — title taxonomy (`core_ai` → `off_role`); decisive against keyword-stuffers.
- **domain** — weighted, saturated count of retrieval/ranking/recsys/NLP/eval terms found in **career descriptions + summary** (skills array is deliberately excluded here).
- **product** — fraction of career at product (non-services) companies.

### Honeypots — calibrated, false-positive-free

Detection thresholds were fit to the actual 100k distribution (see
[scripts/eda.py](scripts/eda.py)). For genuine candidates these quantities are
tightly bounded; the ~80 honeypots sit in a cleanly-separated tail:

- `years_of_experience − career_span > 1.5y` ("8 years at a 3-year-old company")
- `years_of_experience − Σ role_durations > 2.0y`
- `≥ 3` advanced/expert skills with `0` months of use ("expert in 10 skills, 0 years used")

### Reasoning — "rank, don't generate"

Every justification is **assembled from the candidate's real fields** — there is
**no free-text LLM step**, so it is hallucination-free by construction (exactly
what the Stage-4 manual review rewards). Phrasing varies by fit band and a
deterministic per-candidate index, and the tone matches the rank (concerns are
surfaced for weaker picks). See [reasoning.py](src/redrob_ranker/reasoning.py).

---

## Architecture

```
candidates.jsonl ──► [reference date scan] ──► [feature extraction] ──┐
                                                                      │  structured_fit
                                                                      │  behavioral_modifier
                                                                      │  honeypot flag
                                                                      ▼
                          [semantic channel: TF-IDF  (+ dense via RRF, optional)]
                                                                      │
                                                                      ▼
                       final = blend · modifier · gate ──► rank + grounded reasoning
                                                                      ▼
                                                              submission.csv (top-100)
```

## Layout

```
rank.py                         single-command entry point
src/redrob_ranker/
  config.py                     weights, role taxonomy, domain vocab, JD query
  loading.py                    jsonl/gz streaming + dataset reference date
  features.py                   structured fit, behavioral modifier, honeypot
  text_channel.py               TF-IDF + RRF dense fusion
  reasoning.py                  grounded, hallucination-free justifications
  pipeline.py                   orchestration + spec-compliant CSV writer
scripts/
  eda.py                        honeypot calibration + role landscape
  precompute_embeddings.py      OPTIONAL model2vec dense artifact (offline)
sandbox/app.py                  Streamlit demo (mandatory sandbox requirement)
tests/test_smoke.py             behavioural smoke tests
```

## Compute & reproducibility

- **CPU-only, no network** at ranking time. Default run uses TF-IDF — **no model
  downloads**.
- **Optional dense channel:** run [scripts/precompute_embeddings.py](scripts/precompute_embeddings.py)
  once offline (it may download a ~30 MB static `model2vec` model and may exceed
  5 minutes — allowed for pre-computation). It writes
  `artifacts/dense_embeddings.npz`, which `rank.py` auto-detects; the timed
  ranking step only *loads* the vectors. Falls back to TF-IDF if absent.
- The reference "now" for recency is the dataset's max `last_active_date`
  (reproducible — no wall-clock dependency).

## Sandbox

```bash
streamlit run sandbox/app.py
```

Upload `sample_candidates.json` (or any ≤100-candidate `.jsonl`/`.json`) and the
demo runs the identical pipeline end-to-end. Deployable as-is to HuggingFace
Spaces or Streamlit Cloud (see [sandbox/requirements.txt](sandbox/requirements.txt)).

## Results (this run)

- Top-100 is **100% genuine AI/ML/retrieval roles** (Recommendation Systems /
  ML / Applied ML / AI / Search / NLP Engineers, Applied Scientists) — **zero
  keyword-stuffers**.
- **0 honeypots** in the top-100 (46 flagged across the pool).
- Validator: **"Submission is valid."**
