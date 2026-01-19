import json
import re
import glob
from tqdm import tqdm
from collections import defaultdict

############################################
# CONFIG
############################################

STOPWORDS = set([
    "the", "is", "are", "was", "were", "a", "an", "of", "to", "in",
    "and", "or", "for", "with", "on", "by", "as", "that", "this",
    "it", "be", "from", "at"
])

############################################
# TOKENIZATION
############################################

def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z]+", text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

############################################
# METRICS
############################################

def citation_overlap(answer, docs):
    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return 0.0

    doc_tokens = set()
    for d in docs:
        doc_tokens.update(tokenize(d["text"]))

    overlap = answer_tokens & doc_tokens
    return len(overlap) / len(answer_tokens)

def attribution_score(answer, docs):
    sentences = re.split(r"[.!?]", answer)
    supported, total = 0, 0

    doc_text = " ".join(d["text"].lower() for d in docs)

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 5:
            continue

        total += 1
        tokens = tokenize(sent)
        if not tokens:
            continue

        hits = sum(1 for t in tokens if t in doc_text)
        if hits / len(tokens) >= 0.3:
            supported += 1

    return supported / total if total > 0 else 0.0

############################################
# RUN
############################################

if __name__ == "__main__":

    stats = defaultdict(lambda: {
        "citation": [],
        "attribution": []
    })

    # ---- RAG ----
    for path in glob.glob("rag_*.jsonl"):
        try:
            parts = path.replace(".jsonl", "").split("_", 2)
            if len(parts) != 3:
                continue
            _, llm, retriever = parts
            key = f"RAG | {llm} + {retriever}"
        except ValueError:
            continue

        with open(path, encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Evaluating {key}"):
                item = json.loads(line)

                stats[key]["citation"].append(
                    citation_overlap(
                        item["generated_answer"],
                        item["retrieved_docs"]
                    )
                )
                stats[key]["attribution"].append(
                    attribution_score(
                        item["generated_answer"],
                        item["retrieved_docs"]
                    )
                )

    # ---- REPORT ----
    for k, v in stats.items():
        print(f"\n=== Faithfulness | {k} ===")
        print(f"Avg Citation Overlap: {sum(v['citation']) / len(v['citation']):.4f}")
        print(f"Avg Attribution Score: {sum(v['attribution']) / len(v['attribution']):.4f}")
