import torch
import faiss
import numpy as np
from transformers import (
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer
)


# PATHS

CORPUS_PATH = "data/processed/pubmed_corpus_fixed.tsv"
INDEX_PATH = "dpr_faiss.index"


# DEVICE

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# LOAD QUESTION MODEL

q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)
q_model = DPRQuestionEncoder.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
).to(device)
q_model.eval()


# LOAD FAISS INDEX

index = faiss.read_index(INDEX_PATH)


# LOAD CORPUS TEXTS

texts = []
with open(CORPUS_PATH, encoding="utf-8") as f:
    for line in f:
        _, text = line.rstrip("\n").split("\t", 1)
        texts.append(text)

print(f"Loaded {len(texts)} documents")



# RETRIEVE FUNCTION (THIS IS THE KEY FIX)


def retrieve(query, top_k=5):
    """
    Dense retrieval using DPR + FAISS.

    Args:
        query (str): input question
        top_k (int): number of passages to retrieve

    Returns:
        List[dict]: retrieved passages with text and score
    """

    inputs = q_tokenizer(
        query,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        q_vec = q_model(**inputs).pooler_output
        q_vec = torch.nn.functional.normalize(q_vec, dim=1).cpu().numpy()

    scores, indices = index.search(q_vec, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "rank": rank + 1,
            "score": float(scores[0][rank]),
            "text": texts[idx]
        })

    return results


# ==========================================================
#  STANDALONE TEST (OLD BEHAVIOR PRESERVED)
# ==========================================================

if __name__ == "__main__":
    query = "What is programmed cell death?"
    top_k = 5

    results = retrieve(query, top_k)

    print("\nQuery:", query)
    print("\nTop-k retrieved passages:\n")

    for r in results:
        print(f"[{r['rank']}] (score={r['score']:.4f})")
        print(r["text"][:500])
        print("-" * 80)
