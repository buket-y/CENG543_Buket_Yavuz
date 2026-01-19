# scripts/retrieve.py

import torch
import faiss
from transformers import (
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer
)
from sentence_transformers import SentenceTransformer
import numpy as np

############################################
# DEVICE
############################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


############################################
# BASE RETRIEVER (ABSTRACTION)
############################################

class BaseRetriever:
    """
    Abstract base class for dense retrievers.
    """

    def retrieve(self, query: str, top_k: int):
        raise NotImplementedError


############################################
# DPR RETRIEVER
############################################

class DPRRetriever(BaseRetriever):
    """
    Dense Passage Retriever using DPR + FAISS.
    All heavy resources are loaded ONLY at initialization time.
    """

    def __init__(
        self,
        corpus_path: str,
        index_path: str,
        question_encoder_name: str = "facebook/dpr-question_encoder-single-nq-base"
    ):
        self.corpus_path = corpus_path
        self.index_path = index_path
        self.question_encoder_name = question_encoder_name

        self.device = DEVICE

        self._load_models()
        self._load_index()
        self._load_corpus()

    def _load_models(self):
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            self.question_encoder_name
        )
        self.q_model = DPRQuestionEncoder.from_pretrained(
            self.question_encoder_name
        ).to(self.device)
        self.q_model.eval()

    def _load_index(self):
        self.index = faiss.read_index(self.index_path)

    def _load_corpus(self):
        self.texts = []
        with open(self.corpus_path, encoding="utf-8") as f:
            for line in f:
                _, text = line.rstrip("\n").split("\t", 1)
                self.texts.append(text)

    def retrieve(self, query: str, top_k: int = 5):
        inputs = self.q_tokenizer(
            query,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            q_vec = self.q_model(**inputs).pooler_output
            q_vec = torch.nn.functional.normalize(q_vec, dim=1)
            q_vec = q_vec.cpu().numpy()

        scores, indices = self.index.search(q_vec, top_k)

        results = []
        for rank, idx in enumerate(indices[0]):
            results.append({
                "rank": rank + 1,
                "score": float(scores[0][rank]),
                "text": self.texts[idx]
            })

        return results


############################################
# SBERT RETRIEVER (NEW)
############################################

class SBERTRetriever(BaseRetriever):
    """
    Dense Retriever using Sentence-BERT (MiniLM) + FAISS.
    Much faster than DPR, very strong baseline.
    """

    def __init__(
        self,
        corpus_path: str,
        index_path: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.corpus_path = corpus_path
        self.index_path = index_path
        self.model_name = model_name

        self.device = DEVICE

        self._load_model()
        self._load_index()
        self._load_corpus()

    def _load_model(self):
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def _load_index(self):
        self.index = faiss.read_index(self.index_path)

    def _load_corpus(self):
        self.texts = []
        with open(self.corpus_path, encoding="utf-8") as f:
            for line in f:
                _, text = line.rstrip("\n").split("\t", 1)
                self.texts.append(text)

    def retrieve(self, query: str, top_k: int = 5):
        with torch.no_grad():
            q_vec = self.model.encode(
                query,
                normalize_embeddings=True,
                convert_to_numpy=True
            )

        q_vec = np.expand_dims(q_vec, axis=0)
        scores, indices = self.index.search(q_vec, top_k)

        results = []
        for rank, idx in enumerate(indices[0]):
            results.append({
                "rank": rank + 1,
                "score": float(scores[0][rank]),
                "text": self.texts[idx]
            })

        return results


############################################
# STANDALONE TEST
############################################

if __name__ == "__main__":
    retriever = SBERTRetriever(
        corpus_path="data/processed/pubmed_corpus_fixed.tsv",
        index_path="sbert_faiss.index"
    )

    query = "What is programmed cell death?"
    results = retriever.retrieve(query, top_k=5)

    print("Query:", query)
    print("\nTop-k retrieved passages:\n")

    for r in results:
        print(f"[{r['rank']}] (score={r['score']:.4f})")
        print(r["text"][:500])
        print("-" * 80)
