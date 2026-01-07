import json
import re
from tqdm import tqdm



# CONFIG

RAG_PATH = "rag_batch.jsonl"

STOPWORDS = set([
    "the", "is", "are", "was", "were", "a", "an", "of", "to", "in",
    "and", "or", "for", "with", "on", "by", "as", "that", "this",
    "it", "be", "from", "at"
])



# TOKENIZATION

def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z]+", text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]



# CITATION OVERLAP

def citation_overlap(answer, docs):
    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return 0.0

    doc_tokens = set()
    for d in docs:
        doc_tokens.update(tokenize(d["text"]))

    overlap = answer_tokens & doc_tokens
    return len(overlap) / len(answer_tokens)



# ATTRIBUTION SCORE

def attribution_score(answer, docs):
    sentences = re.split(r"[.!?]", answer)
    supported = 0
    total = 0

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

    if total == 0:
        return 0.0

    return supported / total



# RUN EVALUATION

citation_scores = []
attribution_scores = []

with open(RAG_PATH, encoding="utf-8") as f:
    for line in tqdm(f, desc="Evaluating faithfulness"):
        item = json.loads(line)

        answer = item["generated_answer"]
        docs = item["retrieved_docs"]

        citation_scores.append(
            citation_overlap(answer, docs)
        )
        attribution_scores.append(
            attribution_score(answer, docs)
        )



# RESULTS

avg_citation = sum(citation_scores) / len(citation_scores)
avg_attribution = sum(attribution_scores) / len(attribution_scores)

print("\n=== Faithfulness Metrics (DPR-RAG) ===")
print(f"Avg Citation Overlap: {avg_citation:.4f}")
print(f"Avg Attribution Score: {avg_attribution:.4f}")
