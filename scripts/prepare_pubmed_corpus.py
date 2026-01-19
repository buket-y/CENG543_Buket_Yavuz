import json
import csv
from pathlib import Path


# Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "pubmedqa.json"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "pubmed_corpus.tsv"

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def normalize(text: str) -> str:
    return " ".join(text.strip().split())

print(f"Loading raw data from: {RAW_PATH}")

with open(RAW_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
skipped = 0

for idx, (qid, item) in enumerate(data.items()):
    context_obj = item.get("context")

    
    if not isinstance(context_obj, dict):
        skipped += 1
        continue

    sentences = context_obj.get("contexts")
    if not sentences or not isinstance(sentences, list):
        skipped += 1        
        continue

    text = normalize(" ".join(sentences))
    doc_id = f"PMQA_DOC_{idx}"

    rows.append((doc_id, text))

print(f"Prepared {len(rows)} documents (skipped {skipped})")

with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    for row in rows:
        writer.writerow(row)

print(f"SUCCESS: Corpus written to {OUT_PATH}")
