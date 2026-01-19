import torch
import faiss
import numpy as np
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer
)
from tqdm import tqdm


# PATHS

CORPUS_PATH = "data/processed/pubmed_corpus_fixed.tsv"
INDEX_PATH = "dpr_faiss.index"
EMB_PATH = "dpr_ctx_embeddings.npy"


# DEVICE

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# LOAD MODEL

tokenizer = DPRContextEncoderTokenizer.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
)
model = DPRContextEncoder.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
).to(device)
model.eval()


# LOAD CORPUS

texts = []
with open(CORPUS_PATH, encoding="utf-8") as f:
    for line in f:
        _, text = line.rstrip("\n").split("\t", 1)
        texts.append(text)

print(f"Loaded {len(texts)} documents")


# ENCODE DOCUMENTS

batch_size = 16   
embeddings = []

with torch.no_grad():
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        outputs = model(**inputs)
        vecs = outputs.pooler_output
        vecs = torch.nn.functional.normalize(vecs, dim=1)
        embeddings.append(vecs.cpu().numpy())

embeddings = np.vstack(embeddings)
np.save(EMB_PATH, embeddings)

print("Embeddings shape:", embeddings.shape)


# BUILD FAISS INDEX

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)   # cosine similarity
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)
print("FAISS index written to:", INDEX_PATH)
