#!/usr/bin/env python3
"""
Analysis and Visualization Script for RAG vs LLM-Only Experiments.
Generates plots and saves them to a 'results' folder.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

# ============================================
# CONFIG
# ============================================

RESULTS_DIR = "results"
LABELS = ["yes", "no", "maybe"]

os.makedirs(RESULTS_DIR, exist_ok=True)

# Use non-interactive backend to save files without display
plt.switch_backend('Agg')

import re

# ============================================
# HELPERS
# ============================================

def normalize_label(text):
    if text is None:
        return "unknown"
    
    text = text.lower()
    
    # Check for direct matches first
    if text in ["yes", "no", "maybe"]:
        return text
        
    # Regex for more complex answers
    if re.search(r"\byes\b|support|affirmative", text):
        return "yes"
    if re.search(r"\bno\b|does not|not support|negative", text):
        return "no"
    if re.search(r"maybe|unclear|mixed|partial|uncertain", text):
        return "maybe"
    if re.search(r"insufficient|unknown|cannot determine|not enough", text):
        return "unknown"
        
    return "unknown"


def load_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# ============================================
# LOAD ALL RESULT FILES
# ============================================

llm_files = sorted(glob.glob("llm_only_*.jsonl"))
rag_files = sorted(glob.glob("rag_*.jsonl"))

print(f"Found {len(llm_files)} LLM-only files: {llm_files}")
print(f"Found {len(rag_files)} RAG files: {rag_files}")

if not llm_files and not rag_files:
    print("No result files found!")
    exit(1)

# ============================================
# 1. AGGREGATE METRICS TABLE
# ============================================

metrics_data = []

for path in llm_files:
    llm_name = path.replace("llm_only_", "").replace(".jsonl", "")
    data = load_jsonl(path)
    
    gold = [normalize_label(x.get("gold_answer")) for x in data]
    pred = [normalize_label(x.get("generated_answer")) for x in data]
    
    # Filter valid labels
    valid_pairs = [(g, p) for g, p in zip(gold, pred) if g in LABELS]
    if valid_pairs:
        gold_valid, pred_valid = zip(*valid_pairs)
        macro_f1 = f1_score(gold_valid, pred_valid, labels=LABELS, average="macro", zero_division=0)
        accuracy = accuracy_score(gold_valid, pred_valid)
        metrics_data.append({
            "Model": f"LLM-only ({llm_name})",
            "Samples": len(valid_pairs),
            "Macro-F1": macro_f1,
            "Accuracy": accuracy
        })

for path in rag_files:
    parts = path.replace(".jsonl", "").split("_")
    if len(parts) >= 3:
        llm_name = parts[1]
        retriever_name = parts[2]
    else:
        continue
    
    data = load_jsonl(path)
    
    gold = [normalize_label(x.get("gold_answer")) for x in data]
    pred = [normalize_label(x.get("generated_answer")) for x in data]
    
    valid_pairs = [(g, p) for g, p in zip(gold, pred) if g in LABELS]
    if valid_pairs:
        gold_valid, pred_valid = zip(*valid_pairs)
        macro_f1 = f1_score(gold_valid, pred_valid, labels=LABELS, average="macro", zero_division=0)
        accuracy = accuracy_score(gold_valid, pred_valid)
        metrics_data.append({
            "Model": f"RAG ({llm_name} + {retriever_name})",
            "Samples": len(valid_pairs),
            "Macro-F1": macro_f1,
            "Accuracy": accuracy
        })

metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv(os.path.join(RESULTS_DIR, "metrics_summary.csv"), index=False)
print(f"\n✔ Metrics saved to {RESULTS_DIR}/metrics_summary.csv")
print(metrics_df.to_string(index=False))

# ============================================
# 2. CONFUSION MATRICES
# ============================================

def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=LABELS,
                yticklabels=LABELS,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Gold")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close()
    print(f"✔ Saved {filename}")

# Generate confusion matrices for all models
for path in llm_files:
    llm_name = path.replace("llm_only_", "").replace(".jsonl", "")
    data = load_jsonl(path)
    gold = [normalize_label(x.get("gold_answer")) for x in data]
    pred = [normalize_label(x.get("generated_answer")) for x in data]
    valid_pairs = [(g, p) for g, p in zip(gold, pred) if g in LABELS]
    if valid_pairs:
        gold_valid, pred_valid = zip(*valid_pairs)
        save_confusion_matrix(gold_valid, pred_valid,
                              f"Confusion Matrix – LLM-only ({llm_name})",
                              f"confusion_llm_only_{llm_name}.png")

for path in rag_files:
    parts = path.replace(".jsonl", "").split("_")
    if len(parts) >= 3:
        llm_name = parts[1]
        retriever_name = parts[2]
    else:
        continue
    data = load_jsonl(path)
    gold = [normalize_label(x.get("gold_answer")) for x in data]
    pred = [normalize_label(x.get("generated_answer")) for x in data]
    valid_pairs = [(g, p) for g, p in zip(gold, pred) if g in LABELS]
    if valid_pairs:
        gold_valid, pred_valid = zip(*valid_pairs)
        save_confusion_matrix(gold_valid, pred_valid,
                              f"Confusion Matrix – RAG ({llm_name} + {retriever_name})",
                              f"confusion_rag_{llm_name}_{retriever_name}.png")

# ============================================
# 3. F1 SCORE COMPARISON BAR CHART
# ============================================

if len(metrics_df) > 0:
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics_df))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, metrics_df["Macro-F1"], width, label="Macro-F1", color="steelblue")
    bars2 = plt.bar(x + width/2, metrics_df["Accuracy"], width, label="Accuracy", color="coral")
    
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Macro-F1 and Accuracy Comparison")
    plt.xticks(x, metrics_df["Model"], rotation=45, ha="right")
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "f1_accuracy_comparison.png"), dpi=150)
    plt.close()
    print("✔ Saved f1_accuracy_comparison.png")

# ============================================
# 4. RETRIEVAL SCORE DISTRIBUTION (RAG only)
# ============================================

for path in rag_files:
    parts = path.replace(".jsonl", "").split("_")
    if len(parts) >= 3:
        llm_name = parts[1]
        retriever_name = parts[2]
    else:
        continue
    
    data = load_jsonl(path)
    
    citation_scores = []
    for item in data:
        docs = item.get("retrieved_docs", [])
        if len(docs) > 0:
            citation_scores.append(np.mean([d.get("score", 0) for d in docs]))
        else:
            citation_scores.append(0.0)
    
    plt.figure(figsize=(8, 5))
    plt.hist(citation_scores, bins=20, edgecolor="black", alpha=0.7)
    plt.xlabel("Average Retrieval Score")
    plt.ylabel("Frequency")
    plt.title(f"Retrieval Score Distribution ({llm_name} + {retriever_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"retrieval_dist_{llm_name}_{retriever_name}.png"), dpi=150)
    plt.close()
    print(f"✔ Saved retrieval_dist_{llm_name}_{retriever_name}.png")

# ============================================
# 5. FAITHFULNESS METRICS (if available)
# ============================================

faithfulness_data = []

for path in rag_files:
    parts = path.replace(".jsonl", "").split("_")
    if len(parts) >= 3:
        llm_name = parts[1]
        retriever_name = parts[2]
    else:
        continue
    
    data = load_jsonl(path)
    
    citation_overlaps = []
    for item in data:
        docs = item.get("retrieved_docs", [])
        answer = item.get("generated_answer", "")
        
        if docs and answer:
            doc_text = " ".join([d.get("text", "") for d in docs]).lower()
            answer_tokens = set(answer.lower().split())
            doc_tokens = set(doc_text.split())
            
            if answer_tokens:
                overlap = len(answer_tokens & doc_tokens) / len(answer_tokens)
                citation_overlaps.append(overlap)
    
    if citation_overlaps:
        avg_overlap = np.mean(citation_overlaps)
        faithfulness_data.append({
            "Model": f"RAG ({llm_name} + {retriever_name})",
            "Avg Citation Overlap": avg_overlap
        })

if faithfulness_data:
    faith_df = pd.DataFrame(faithfulness_data)
    faith_df.to_csv(os.path.join(RESULTS_DIR, "faithfulness_summary.csv"), index=False)
    print(f"\n✔ Faithfulness metrics saved to {RESULTS_DIR}/faithfulness_summary.csv")
    print(faith_df.to_string(index=False))

# ============================================
# 6. TEXT REPORT GENERATION
# ============================================

report_path = os.path.join(RESULTS_DIR, "summary_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("========================================================\n")
    f.write("              EXPERIMENT SUMMARY REPORT                 \n")
    f.write("========================================================\n\n")
    
    # 1. Classification Metrics
    f.write("--------------------------------------------------------\n")
    f.write("1. CLASSIFICATION METRICS (Macro-F1 & Accuracy)\n")
    f.write("--------------------------------------------------------\n")
    if len(metrics_df) > 0:
        f.write(metrics_df.to_string(index=False))
    else:
        f.write("No metrics computed.")
    f.write("\n\n")

    # 2. Faithfulness Metrics
    f.write("--------------------------------------------------------\n")
    f.write("2. FAITHFULNESS METRICS (RAG Only)\n")
    f.write("--------------------------------------------------------\n")
    if faithfulness_data:
        f_df = pd.DataFrame(faithfulness_data)
        f.write(f_df.to_string(index=False))
    else:
        f.write("No faithfulness metrics computed.")
    f.write("\n\n")
    
    # 3. Model Observations
    f.write("--------------------------------------------------------\n")
    f.write("3. OBSERVATIONS\n")
    f.write("--------------------------------------------------------\n")
    
    # Find best model
    if len(metrics_df) > 0:
        best_acc = metrics_df.loc[metrics_df["Accuracy"].idxmax()]
        best_f1 = metrics_df.loc[metrics_df["Macro-F1"].idxmax()]
        
        f.write(f"Best Accuracy: {best_acc['Accuracy']:.4f} ({best_acc['Model']})\n")
        f.write(f"Best Macro-F1: {best_f1['Macro-F1']:.4f} ({best_f1['Model']})\n")
    
    f.write("\n")
    
    # Compare with RAG
    # This is a simple heuristic: check if RAG consistently outperforms LLM-only
    rag_wins = 0
    total_comparisons = 0
    
    # Helper to find matching rows
    def get_metric(model_str, metric_col):
        row = metrics_df[metrics_df["Model"] == model_str]
        if not row.empty:
            return row.iloc[0][metric_col]
        return None

    # Iterate unique LLMs
    unique_llms = set()
    for m in metrics_df["Model"]:
        if "LLM-only" in m:
            unique_llms.add(m.replace("LLM-only (", "").replace(")", ""))
            
    for llm in unique_llms:
        llm_acc = get_metric(f"LLM-only ({llm})", "Accuracy")
        
        # Check DPR
        dpr_acc = get_metric(f"RAG ({llm} + DPR)", "Accuracy")
        if llm_acc is not None and dpr_acc is not None:
            total_comparisons += 1
            if dpr_acc > llm_acc:
                rag_wins += 1
                
        # Check SBERT
        sbert_acc = get_metric(f"RAG ({llm} + SBERT)", "Accuracy")
        if llm_acc is not None and sbert_acc is not None:
            total_comparisons += 1
            if sbert_acc > llm_acc:
                rag_wins += 1
                
    if total_comparisons > 0:
        win_rate = (rag_wins / total_comparisons) * 100
        f.write(f"RAG outperformed LLM-only in {rag_wins}/{total_comparisons} comparisons ({win_rate:.0f}%).\n")

print(f"\n✔ Summary report saved to {report_path}")

# ============================================
# DONE
# ============================================

print(f"\n✔ All analysis complete! Results saved to '{RESULTS_DIR}/' folder.")
