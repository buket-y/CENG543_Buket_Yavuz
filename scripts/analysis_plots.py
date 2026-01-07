import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# PATHS
LLM_PATH = "llm_only_batch.jsonl"
RAG_PATH = "rag_batch.jsonl"

LABELS = ["yes", "no", "maybe"]


# HELPERS

def normalize_label(x):
    if x is None:
        return "unknown"
    x = x.lower()
    if "yes" in x:
        return "yes"
    if "no" in x:
        return "no"
    if "maybe" in x:
        return "maybe"
    return "unknown"


def load_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data



# LOAD DATA

llm_data = load_jsonl(LLM_PATH)
rag_data = load_jsonl(RAG_PATH)

assert len(llm_data) == len(rag_data)

gold = [normalize_label(x["gold_answer"]) for x in llm_data]
llm_pred = [normalize_label(x["generated_answer"]) for x in llm_data]
rag_pred = [normalize_label(x["generated_answer"]) for x in rag_data]


# 1CONFUSION MATRICES

def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=LABELS,
                yticklabels=LABELS,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Gold")
    plt.title(title)
    plt.tight_layout()
    plt.show()


plot_confusion(gold, llm_pred, "Confusion Matrix – LLM-only")
plot_confusion(gold, rag_pred, "Confusion Matrix – DPR-RAG")


# PER-QUESTION CORRECTNESS

llm_correct = [int(p == g) for p, g in zip(llm_pred, gold)]
rag_correct = [int(p == g) for p, g in zip(rag_pred, gold)]

x = np.arange(len(llm_correct))

plt.figure(figsize=(10, 3))
plt.scatter(x, llm_correct, label="LLM-only", alpha=0.6)
plt.scatter(x, rag_correct, label="DPR-RAG", alpha=0.6)
plt.yticks([0, 1], ["Wrong", "Correct"])
plt.xlabel("Question Index")
plt.title("Per-question Answer Correctness")
plt.legend()
plt.tight_layout()
plt.show()


#  CITATION OVERLAP DISTRIBUTION (RAG)

citation_scores = []

for item in rag_data:
    docs = item["retrieved_docs"]
    if len(docs) == 0:
        citation_scores.append(0.0)
    else:
        citation_scores.append(np.mean([d["score"] for d in docs]))

plt.figure(figsize=(6, 4))
plt.hist(citation_scores, bins=10, edgecolor="black")
plt.xlabel("Average Retrieval Score")
plt.ylabel("Frequency")
plt.title("Citation / Retrieval Score Distribution (DPR-RAG)")
plt.tight_layout()
plt.show()


#  FAITHFULNESS vs CORRECTNESS

plt.figure(figsize=(6, 4))
plt.scatter(citation_scores, rag_correct, alpha=0.7)
plt.yticks([0, 1], ["Wrong", "Correct"])
plt.xlabel("Avg Retrieval Score")
plt.ylabel("Answer Correctness")
plt.title("Faithfulness vs Correctness (DPR-RAG)")
plt.tight_layout()
plt.show()

print("✔ Analysis plots generated successfully.")
