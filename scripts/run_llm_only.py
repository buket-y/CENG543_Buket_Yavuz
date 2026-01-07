import json
from generate import generate_answer

def run_llm_only(question, gold_answer=None):
    answer = generate_answer(question, docs=[])

    result = {
        "question": question,
        "gold_answer": gold_answer,
        "retrieved_docs": [],
        "generated_answer": answer,
        "setting": "LLM-only"
    }
    return result


if __name__ == "__main__":
    q = "What is the role of BRCA1 in breast cancer?"

    out = run_llm_only(q)

    print("LLM-only Answer:\n", out["generated_answer"])

    with open("llm_only_outputs.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")
