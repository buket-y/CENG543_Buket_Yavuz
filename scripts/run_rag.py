from retrieve import retrieve
from generate import generate_answer
import json

def run_rag(question, gold_answer=None, k=5):
    docs = retrieve(question, top_k=k)
    answer = generate_answer(question, docs)

    result = {
        "question": question,
        "gold_answer": gold_answer,
        "retrieved_docs": docs,
        "generated_answer": answer,
        "retriever": "DPR",
        "generator": "Mistral-7B-Instruct",
        "k": k
    }

    return result


if __name__ == "__main__":
    q = "What is the role of BRCA1 in breast cancer?"
    out = run_rag(q)

    print("Generated Answer:\n", out["generated_answer"])

    with open("rag_outputs.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")
