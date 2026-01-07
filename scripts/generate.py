# scripts/generate.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

############################################
# MODEL CONFIGURATION
############################################

# Şu an için SABİT bir model seçiyoruz.
# (İleride tek satırda değiştirilebilir)
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


############################################
# LOAD MODEL & TOKENIZER
############################################

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None
)

model.eval()


############################################
# PROMPT BUILDING
############################################

def build_prompt(question, docs):
    """
    Builds a prompt for either:
    - RAG-based generation (docs provided)
    - LLM-only baseline (docs empty)

    Args:
        question (str)
        docs (list): list of retrieved documents (each must have "text")

    Returns:
        prompt (str)
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
# GENERATION FUNCTION
############################################

def generate_answer(
    question,
    docs=None,
    max_new_tokens=128
):
    """
    Generates an answer using the configured LLM.

    Args:
        question (str)
        docs (list or None): retrieved documents for RAG, or [] for LLM-only
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
            do_sample=False,        # deterministik (çok önemli)
            temperature=0.0,
            top_p=1.0
        )

    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Prompt'u temizle, sadece cevabı döndür
    answer = decoded[len(prompt):].strip()

    return answer


############################################
# DEBUG / STANDALONE TEST
############################################

if __name__ == "__main__":
    test_question = "What is the role of BRCA1 in breast cancer?"

    print("=== LLM-ONLY ===")
    print(generate_answer(test_question, docs=[]))

    print("\n=== RAG MODE (DUMMY CONTEXT) ===")
    dummy_docs = [
        {
            "text": "BRCA1 is a tumor suppressor gene involved in DNA repair."
        }
    ]
    print(generate_answer(test_question, docs=dummy_docs))
