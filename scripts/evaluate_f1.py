import json
import re
import glob
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score

############################################
# CONFIG
############################################

LABELS = ["yes", "no", "maybe"]

############################################
# NORMALIZATION
############################################

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

############################################
# LOAD + GROUP
############################################

def load_predictions(path):
    gold, pred = [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            gold_label = item["gold_answer"]
            pred_label = normalize_answer(item["generated_answer"])

            if gold_label not in LABELS:
                continue

            gold.append(gold_label)
            pred.append(pred_label)

    return gold, pred

############################################
# EVALUATION
############################################

def evaluate(path):
    gold, pred = load_predictions(path)

    if not gold:
        return None

    return {
        "samples": len(gold),
        "macro_f1": f1_score(gold, pred, labels=LABELS, average="macro"),
        "accuracy": accuracy_score(gold, pred)
    }

############################################
# RUN
############################################

if __name__ == "__main__":

    results = {}

    # ---- LLM-ONLY ----
    for path in glob.glob("llm_only_*.jsonl"):
        llm = path.replace("llm_only_", "").replace(".jsonl", "")
        results[f"LLM-only | {llm}"] = evaluate(path)

    # ---- RAG ----
    # ---- RAG ----
    for path in glob.glob("rag_*.jsonl"):
        try:
            parts = path.replace(".jsonl", "").split("_", 2)
            if len(parts) != 3:
                continue
            _, llm, retriever = parts
            key = f"RAG | {llm} + {retriever}"
            results[key] = evaluate(path)
        except ValueError:
            continue

    # ---- PRINT ----
    for k, v in results.items():
        if v is None:
            continue
        print(f"\n=== {k} ===")
        print(f"Samples: {v['samples']}")
        print(f"Macro-F1: {v['macro_f1']:.4f}")
        print(f"Accuracy: {v['accuracy']:.4f}")
