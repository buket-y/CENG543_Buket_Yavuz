from datasets import load_dataset
import json
from pathlib import Path


# PROJECT ROOT (TERM_PROJECT)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = RAW_DIR / "pubmedqa.json"

print("Downloading PubMedQA...")
print(f"Output path: {OUT_PATH}")

# HuggingFace PubMedQA (labeled)
dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")

data = {}

for idx, item in enumerate(dataset):
    data[f"PMQA_{idx}"] = {
        "pubmed_id": item.get("pubmed_id"),
        "context": item.get("context"),
        "question": item.get("question"),
        "answer": item.get("final_decision")
    }

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"SUCCESS: {len(data)} samples written.")
