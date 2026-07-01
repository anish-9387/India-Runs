"""
Sandbox demo (Streamlit) for the candidate ranker.

Satisfies the hackathon's mandatory sandbox requirement: accepts a small
candidate sample (<=100 records, .jsonl or a .json array), runs the SAME ranking
pipeline used for the full submission, and shows the ranked output with grounded
reasoning. Runs CPU-only, well within the 5-minute budget on a small sample.

Deploy on HuggingFace Spaces / Streamlit Cloud:
    streamlit run sandbox/app.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pipeline import rank  # noqa: E402

st.set_page_config(page_title="Candidate Ranker", layout="wide")
st.title("Candidate Ranker")
st.caption("Hybrid structured + semantic ranking with grounded, "
           "hallucination-free reasoning. CPU-only, no network.")

uploaded = st.file_uploader(
    "Upload a candidate sample (.jsonl, or a .json array of <=100 candidates)",
    type=["jsonl", "json"],
)
top_n = st.slider("How many to rank", 5, 100, 25)

if uploaded is not None:
    raw = uploaded.read().decode("utf-8")
    # accept either JSONL or a JSON array
    if uploaded.name.endswith(".json"):
        records = json.loads(raw)
    else:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    st.write(f"Loaded **{len(records)}** candidates.")

    if st.button("Rank candidates"):
        with tempfile.TemporaryDirectory() as d:
            in_path = os.path.join(d, "sample.jsonl")
            out_path = os.path.join(d, "submission.csv")
            with open(in_path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            stats = rank(in_path, out_path, dense_path=None,
                         top_n=min(top_n, len(records)))
            import csv
            with open(out_path, encoding="utf-8") as f:
                csv_text = f.read()
            rows = list(csv.DictReader(csv_text.splitlines()))

        st.success(f"Ranked {len(rows)} candidates in {stats['elapsed_sec']}s "
                   f"— {stats['honeypots_in_top']} honeypots in top-{len(rows)}.")
        st.dataframe(rows, use_container_width=True)
        st.download_button("Download submission.csv", data=csv_text,
                           file_name="submission.csv", mime="text/csv")
else:
    st.info("Upload a sample to begin. The bundled `sample_candidates.json` "
            "(50 candidates) works out of the box.")
