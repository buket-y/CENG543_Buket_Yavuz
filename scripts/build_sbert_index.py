# scripts/build_sbert_index.py

import os
import faiss
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


############################################
# CONFIG
############################################

CORPUS_PATH = "data/processed/pubmed_corpus_fixed.tsv"
OUT_INDEX_PATH = "sbert_faiss.index"

SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


############################################
# LOAD CORPUS
############################################

def load_corpus(path):
    texts = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            _, text = line.rstrip("\n").split("\t", 1)
            texts.append(text)

    return texts


############################################
# BUILD SBERT INDEX
############################################

def build_sbert_index():
    print("=== Loading corpus ===")
    texts = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(texts)} documents")

    print("\n=== Loading SBERT model ===")
    model = SentenceTransformer(SBERT_MODEL_NAME, device=DEVICE)

    print("\n=== Encoding corpus ===")
    embeddings = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding"):
        batch = texts[i:i + BATCH_SIZE]
        emb = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)
    dim = embeddings.shape[1]

    print(f"\nEmbedding shape: {embeddings.shape}")

    print("\n=== Building FAISS index ===")
    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(embeddings)

    print(f"FAISS index size: {index.ntotal}")

    print("\n=== Saving index ===")
    faiss.write_index(index, OUT_INDEX_PATH)

    print(f"Index saved to: {OUT_INDEX_PATH}")


############################################
# MAIN
############################################

if __name__ == "__main__":
    build_sbert_index()
