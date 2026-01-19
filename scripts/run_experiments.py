# scripts/run_experiments.py

import json
from tqdm import tqdm

from generate import load_llm
from retrieve import DPRRetriever
from run_rag import run_rag
from run_llm_only import run_llm_only


# =====================
# CONFIG
# =====================

DATA_PATH = "data/raw/pubmedqa.json"

MAX_QUESTIONS = 1000
TOP_K = 5

# ---- LLM CONFIGS ----
LLMS = {
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.2",
    "Phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "Gemma-2B": "google/gemma-2b-it",
}

# ---- RETRIEVER CONFIG ----
RETRIEVER_CONFIGS = {
    "DPR": {
        "corpus_path": "data/processed/pubmed_corpus_fixed.tsv",
        "index_path": "dpr_faiss.index"
    },
    "SBERT": {
        "type": "sbert",
        "corpus_path": "data/processed/pubmed_corpus_fixed.tsv",
        "index_path": "sbert_faiss.index",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2"
    }
}


# =====================
# LOAD DATASET
# =====================

with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)

questions = []
for qid, item in data.items():
    questions.append({
        "id": qid,
        "question": item["question"],
        "gold_answer": item.get("answer", None)
    })

questions = questions[:MAX_QUESTIONS]
print(f"Loaded {len(questions)} questions")


# =====================
# RUN EXPERIMENTS
# =====================

for llm_name, llm_model in LLMS.items():

    print(f"\n=== Loading LLM: {llm_name} ===")
    tokenizer, model = load_llm(llm_model)

    # -------------------------
    # LLM-ONLY
    # -------------------------
    out_llm_only = f"llm_only_{llm_name}.jsonl"
    open(out_llm_only, "w").close()

    for item in tqdm(questions, desc=f"LLM-only | {llm_name}"):

        record = run_llm_only(
            question=item["question"],
            gold_answer=item["gold_answer"],
            tokenizer=tokenizer,
            model=model,
            generator_name=llm_name
        )

        record["id"] = item["id"]

        with open(out_llm_only, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


    # -------------------------
    # RAG (per retriever)
    # -------------------------
    for retriever_name, cfg in RETRIEVER_CONFIGS.items():

        print(f"\n--- Initializing Retriever: {retriever_name} ---")

        if cfg.get("type") == "sbert":
            from retrieve import SBERTRetriever
            retriever = SBERTRetriever(
                corpus_path=cfg["corpus_path"],
                index_path=cfg["index_path"],
                model_name=cfg["model_name"]
            )
        else:
            retriever = DPRRetriever(
                corpus_path=cfg["corpus_path"],
                index_path=cfg["index_path"]
            )

        out_rag = f"rag_{llm_name}_{retriever_name}.jsonl"
        open(out_rag, "w").close()

        for item in tqdm(
            questions,
            desc=f"RAG | {llm_name} + {retriever_name}"
        ):

            record = run_rag(
                question=item["question"],
                gold_answer=item["gold_answer"],
                k=TOP_K,
                retriever=retriever,
                tokenizer=tokenizer,
                model=model,
                retriever_name=retriever_name,
                generator_name=llm_name
            )

            record["id"] = item["id"]

            with open(out_rag, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


print("\nAll batch experiments completed.")
