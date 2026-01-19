# scripts/generate.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

############################################
# DEVICE & DTYPE
############################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


############################################
# LLM LOADING (CALLED EXPLICITLY)
############################################

def load_llm(model_name: str):
    """
    Loads tokenizer and model for a given LLM.
    This function MUST be called explicitly (no import-time loading).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None
    )

    model.eval()
    return tokenizer, model


############################################
# PROMPT BUILDING
############################################

def build_prompt(question, docs):
    """
    Builds a prompt for:
    - RAG-based generation (docs provided)
    - LLM-only baseline (docs empty)
    """

    # -------- RAG MODE --------
    if docs is not None and len(docs) > 0:
        context = "\n\n".join([d["text"] for d in docs])

        prompt = f"""
You are a biomedical question answering assistant.

Answer the question ONLY using the information provided in the context.
Do NOT use any external knowledge.
If the answer cannot be derived from the context, say "Insufficient evidence."

Context:
{context}

Question:
{question}

Answer:
"""
    # -------- LLM-ONLY MODE --------
    else:
        prompt = f"""
You are a biomedical question answering assistant.

Answer the following question to the best of your knowledge.

Question:
{question}

Answer:
"""

    return prompt.strip()


############################################
# GENERATION FUNCTION (STATELESS)
############################################

def generate_answer(
    question,
    docs,
    tokenizer,
    model,
    max_new_tokens=128
):
    """
    Generates an answer using the given LLM.

    Args:
        question (str)
        docs (list): [] for LLM-only, retrieved docs for RAG
        tokenizer: HuggingFace tokenizer (REQUIRED)
        model: HuggingFace causal LM (REQUIRED)
        max_new_tokens (int)

    Returns:
        answer (str)
    """

    prompt = build_prompt(question, docs)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,      # deterministic
            temperature=0.0,
            top_p=1.0
        )

    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Remove prompt, return only the answer
    answer = decoded[len(prompt):].strip()
    return answer


############################################
# DEBUG / STANDALONE TEST (OPTIONAL)
############################################

if __name__ == "__main__":
    tokenizer, model = load_llm("mistralai/Mistral-7B-Instruct-v0.2")

    test_question = "What is the role of BRCA1 in breast cancer?"

    print("=== LLM-ONLY ===")
    print(generate_answer(
        question=test_question,
        docs=[],
        tokenizer=tokenizer,
        model=model
    ))

    print("\n=== RAG MODE (DUMMY CONTEXT) ===")
    dummy_docs = [
        {"text": "BRCA1 is a tumor suppressor gene involved in DNA repair."}
    ]
    print(generate_answer(
        question=test_question,
        docs=dummy_docs,
        tokenizer=tokenizer,
        model=model
    ))
