import json
from tqdm import tqdm

from scripts.generate import generate_answer
from scripts.retrieve import retrieve


# =====================
# CONFIG
# =====================
DATA_PATH = "data/raw/pubmedqa.json"

OUT_LLM_ONLY = "llm_only_batch.jsonl"
OUT_RAG = "rag_batch.jsonl"

MAX_QUESTIONS = 50      # deney için ilk N soru
TOP_K = 5


# =====================
# LOAD DATASET
# =====================
with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)

questions = []

for qid, item in data.items():
    # PubMedQA schema (CONFIRMED)
    question = item["question"]                 # ✅ doğru key
    gold_answer = item.get("answer", None)      # yes / no / maybe

    questions.append({
        "id": qid,
        "question": question,
        "gold_answer": gold_answer
    })

questions = questions[:MAX_QUESTIONS]

print(f"Loaded {len(questions)} questions")


# =====================
# CLEAR OLD OUTPUTS
# =====================
open(OUT_LLM_ONLY, "w").close()
open(OUT_RAG, "w").close()


# =====================
# RUN EXPERIMENTS
# =====================
for item in tqdm(questions, desc="Running batch experiments"):

    qid = item["id"]
    question = item["question"]
    gold_answer = item["gold_answer"]

    # ==================================================
    # LLM-ONLY
    # ==================================================
    llm_answer = generate_answer(
        question=question,
        docs=[]
    )

    llm_record = {
        "id": qid,
        "question": question,
        "gold_answer": gold_answer,
        "retrieved_docs": [],
        "generated_answer": llm_answer,
        "setting": "LLM-only"
    }

    with open(OUT_LLM_ONLY, "a", encoding="utf-8") as f:
        f.write(json.dumps(llm_record, ensure_ascii=False) + "\n")


    # ==================================================
    # DPR-RAG
    # ==================================================
    retrieved_docs = retrieve(
        query=question,
        top_k=TOP_K
    )

    rag_answer = generate_answer(
        question=question,
        docs=retrieved_docs
    )

    rag_record = {
        "id": qid,
        "question": question,
        "gold_answer": gold_answer,
        "retrieved_docs": retrieved_docs,
        "generated_answer": rag_answer,
        "setting": "DPR-RAG",
        "top_k": TOP_K
    }

    with open(OUT_RAG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rag_record, ensure_ascii=False) + "\n")


print("Batch experiments completed.")
