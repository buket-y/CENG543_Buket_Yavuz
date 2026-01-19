# scripts/run_rag.py

from typing import Optional

from generate import generate_answer


############################################
# CORE RAG FUNCTION (STATELESS)
############################################

def run_rag(
    question: str,
    gold_answer: Optional[str],
    k: int,
    retriever,
    tokenizer,
    model,
    retriever_name: str,
    generator_name: str
):
    """
    Runs a single RAG inference.
    All dependencies are injected explicitly.
    """

    docs = retriever.retrieve(question, top_k=k)

    answer = generate_answer(
        question=question,
        docs=docs,
        tokenizer=tokenizer,
        model=model
    )

    result = {
        "question": question,
        "gold_answer": gold_answer,
        "retrieved_docs": docs,
        "generated_answer": answer,
        "retriever": retriever_name,
        "generator": generator_name,
        "k": k
    }

    return result


############################################
# OPTIONAL STANDALONE TEST
############################################

if __name__ == "__main__":
    print(
        "run_rag.py is a helper module.\n"
        "Please run experiments via run_experiments.py"
    )
