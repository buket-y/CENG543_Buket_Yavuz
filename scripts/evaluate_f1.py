import json
import re
from sklearn.metrics import f1_score, accuracy_score



# CONFIG

LLM_ONLY_PATH = "llm_only_batch.jsonl"
RAG_PATH = "rag_batch.jsonl"

LABELS = ["yes", "no", "maybe"]



# NORMALIZATION

def normalize_answer(text: str) -> str:
    if text is None:
        return "unknown"

    text = text.lower()

    if re.search(r"\byes\b|support|affirmative", text):
        return "yes"
    if re.search(r"\bno\b|does not|not support|negative", text):
        return "no"
    if re.search(r"maybe|unclear|mixed|partial", text):
        return "maybe"
    if re.search(r"insufficient|unknown|cannot determine|not enough", text):
        return "unknown"

    return "unknown"



# LOAD JSONL

def load_predictions(path):
    gold = []
    pred = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            gold_label = item["gold_answer"]
            pred_label = normalize_answer(item["generated_answer"])

            if gold_label not in LABELS:
                continue  # safety

            gold.append(gold_label)
            pred.append(pred_label)

    return gold, pred



# EVALUATION

def evaluate(path, name):
    gold, pred = load_predictions(path)

    f1 = f1_score(gold, pred, labels=LABELS, average="macro")
    acc = accuracy_score(gold, pred)

    print(f"\n=== {name} ===")
    print(f"Samples: {len(gold)}")
    print(f"Macro-F1: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")



# RUN

if __name__ == "__main__":
    evaluate(LLM_ONLY_PATH, "LLM-only")
    evaluate(RAG_PATH, "DPR-RAG")
