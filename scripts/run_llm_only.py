# scripts/run_llm_only.py

from typing import Optional

from generate import generate_answer


############################################
# CORE LLM-ONLY FUNCTION (STATELESS)
############################################

def run_llm_only(
    question: str,
    gold_answer: Optional[str],
    tokenizer,
    model,
    generator_name: str
):
    """
    Runs a single LLM-only inference.
    All dependencies are injected explicitly.
    """

    answer = generate_answer(
        question=question,
        docs=[],
        tokenizer=tokenizer,
        model=model
    )

    result = {
        "question": question,
        "gold_answer": gold_answer,
        "retrieved_docs": [],
        "generated_answer": answer,
        "retriever": None,
        "generator": generator_name,
        "setting": "LLM-only"
    }

    return result


############################################
# OPTIONAL STANDALONE TEST
############################################

if __name__ == "__main__":
    print(
        "run_llm_only.py is a helper module.\n"
        "Please run experiments via run_experiments.py"
    )
