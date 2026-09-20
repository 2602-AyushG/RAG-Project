from query import (
    load_db,
    retrieve,
    build_prompt,
    call_llm,
    MODEL_NAME
)


# (question, expected source file)
QUESTIONS = [

    # In-corpus questions
    ("What is DenseNet?", "Dense_Net.pdf"),
    ("What is the main idea behind DenseNet?", "Dense_Net.pdf"),

    ("How does EfficientNet improve model efficiency?", "Efficient_Net.pdf"),
    ("What is the compound scaling method used in EfficientNet?", "Efficient_Net.pdf"),

    ("How does Grad-CAM work?", "Grad_CAM.pdf"),
    ("What does the Grad-CAM heatmap represent?", "Grad_CAM.pdf"),

    ("What is LIME?", "LIME.pdf"),

    ("How does Integrated Gradients work?",
     "integrated_axiomatic.pdf"),

    ("What is the purpose of Integrated Gradients?",
     "integrated_axiomatic.pdf"),

    ("What is the role of explainable AI in colorectal histology?",
     "NGNDAI-2026_Paper_625.pdf"),


    # Out-of-corpus questions
    ("What is the capital of France?", None),
    ("How does Bitcoin mining work?", None),
    ("Who won the 2022 FIFA World Cup?", None),
]


K = 5


def main():

    db = load_db()

    hits = 0
    in_total = 0

    refusals = 0
    out_total = 0


    # -----------------------------
    # Hit@K and refusal test
    # -----------------------------

    print("\n--- EVALUATION ---")

    for question, expected_source in QUESTIONS:

        chunks = retrieve(
            db,
            question,
            K
        )

        sources = [
            doc.metadata.get("source")
            for doc, score in chunks
        ]


        # In-corpus question
        if expected_source:

            in_total += 1

            if expected_source in sources:
                hits += 1
                print(f"PASS: {question}")
            else:
                print(f"FAIL: {question}")


        # Out-of-corpus question
        else:

            out_total += 1

            prompt = build_prompt(
                question,
                chunks
            )

            answer = call_llm(prompt)

            if answer.strip() == (
                "I don't know based on the provided documents."
            ):
                refusals += 1

            print(f"\nOUT: {question}")
            print("Answer:", answer)


    # -----------------------------
    # RAG vs no context
    # -----------------------------

    print("\n--- RAG VS NO CONTEXT ---")

    # First 3 in-corpus questions
    for question, expected_source in QUESTIONS[:3]:

        chunks = retrieve(
            db,
            question,
            K
        )

        # With context
        prompt = build_prompt(
            question,
            chunks
        )

        rag_answer = call_llm(prompt)


        # Without context
        prompt = f"""
Answer this question using your general knowledge:

{question}
"""

        no_context_answer = call_llm(
            prompt
        )


        print("\nQUESTION:", question)

        print("\nWITH CONTEXT:")
        print(rag_answer)

        print("\nWITHOUT CONTEXT:")
        print(no_context_answer)


    # -----------------------------
    # Final summary
    # -----------------------------

    hit_rate = hits / in_total
    refusal_rate = refusals / out_total

    print("\n--- FINAL SUMMARY ---")
    print("Number of questions:", len(QUESTIONS))
    print("K:", K)
    print(f"Hit@{K}: {hit_rate:.2%}")
    print(f"Refusal rate: {refusal_rate:.2%}")
    print("Model:", MODEL_NAME)


if __name__ == "__main__":
    main()