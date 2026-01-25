from transformers import pipeline
from .prompt import PROMPT


# Initialize generator (CPU-only, stable)
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1
)


def generate_answer(question, contexts):
    """
    Generate an answer using provided contexts.
    Enforces citation presence if the model drops them.
    """

    prompt = PROMPT.format(
        context="\n".join(contexts),
        question=question
    )

    output = generator(
        prompt,
        max_new_tokens=256,
        do_sample=False
    )[0]["generated_text"]

    # ---- Citation enforcement ----
    citations = [f"[c{i+1}]" for i in range(len(contexts))]

    if not any(c in output for c in citations):
        output = output.strip() + " " + "".join(citations)

    return output


# ---------------- DEMO RUN ----------------

if __name__ == "__main__":
    question = "What is asthma?"

    contexts = [
        "Asthma is a chronic lung disease that inflames and narrows the airways [c1].",
        "Common symptoms include wheezing, coughing, and shortness of breath [c2]."
    ]

    answer = generate_answer(question, contexts)

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)
