import json
from pathlib import Path

from query import (
    load_db,
    retrieve,
    build_prompt,
    call_llm,
    MODEL_NAME
)


# ============================================================
# EVALUATION QUESTIONS
#
# type:
#   direct
#   paraphrased
#   cross-paper
#   out-of-scope
#
# expected_sources:
#   list of acceptable source PDF names
#
# expected_pages:
#   list of expected PDF page numbers
# ============================================================

QUESTIONS = [

    # --------------------------------------------------------
    # DenseNet
    # --------------------------------------------------------

    {
        "id": "D01",
        "question": "What is the main connectivity idea introduced by DenseNet?",
        "expected_sources": ["Dense_Net.pdf"],
        "expected_pages": [1, 2],
        "type": "direct",
    },
    {
        "id": "D02",
        "question": "What advantages do DenseNets obtain from their dense connectivity pattern?",
        "expected_sources": ["Dense_Net.pdf"],
        "expected_pages": [1, 2],
        "type": "direct",
    },
    {
        "id": "D03",
        "question": "How does DenseNet combine feature maps from preceding layers?",
        "expected_sources": ["Dense_Net.pdf"],
        "expected_pages": [2],
        "type": "paraphrased",
    },
    {
        "id": "D04",
        "question": "Why can DenseNet require fewer parameters than traditional feed-forward architectures?",
        "expected_sources": ["Dense_Net.pdf"],
        "expected_pages": [2],
        "type": "paraphrased",
    },
    {
        "id": "D05",
        "question": "How do dense connections improve the flow of information and gradients during training?",
        "expected_sources": ["Dense_Net.pdf"],
        "expected_pages": [2],
        "type": "direct",
    },
    {
        "id": "D06",
        "question": "What is the purpose of transition layers between DenseNet dense blocks?",
        "expected_sources": ["Dense_Net.pdf"],
        "expected_pages": [3],
        "type": "direct",
    },
    {
        "id": "D07",
        "question": "What role do bottleneck and compression layers play in DenseNet-BC?",
        "expected_sources": ["Dense_Net.pdf"],
        "expected_pages": [4, 5],
        "type": "direct",
    },


    # --------------------------------------------------------
    # EfficientNet
    # --------------------------------------------------------

    {
        "id": "E01",
        "question": "What problem does EfficientNet identify with conventional ConvNet scaling?",
        "expected_sources": ["Efficient_Net.pdf"],
        "expected_pages": [1],
        "type": "direct",
    },
    {
        "id": "E02",
        "question": "What three dimensions are jointly scaled by EfficientNet's compound scaling method?",
        "expected_sources": ["Efficient_Net.pdf"],
        "expected_pages": [1, 2],
        "type": "direct",
    },
    {
        "id": "E03",
        "question": "How are the depth, width, and resolution scaling coefficients determined?",
        "expected_sources": ["Efficient_Net.pdf"],
        "expected_pages": [2],
        "type": "direct",
    },
    {
        "id": "E04",
        "question": "Why does EfficientNet argue that depth, width, and image resolution should be scaled together?",
        "expected_sources": ["Efficient_Net.pdf"],
        "expected_pages": [2],
        "type": "paraphrased",
    },
    {
        "id": "E05",
        "question": "What efficiency improvements does EfficientNet-B7 achieve compared with GPipe according to the paper?",
        "expected_sources": ["Efficient_Net.pdf"],
        "expected_pages": [1, 2],
        "type": "direct",
    },
    {
        "id": "E06",
        "question": "How does EfficientNet-B4 compare with ResNet-50 in ImageNet accuracy and FLOPS?",
        "expected_sources": ["Efficient_Net.pdf"],
        "expected_pages": [2],
        "type": "direct",
    },
    {
        "id": "E07",
        "question": "What is the relationship between the EfficientNet baseline network and the EfficientNet model family?",
        "expected_sources": ["Efficient_Net.pdf"],
        "expected_pages": [2, 3],
        "type": "paraphrased",
    },


    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    {
        "id": "G01",
        "question": "What is Grad-CAM designed to produce?",
        "expected_sources": ["Grad_CAM.pdf"],
        "expected_pages": [1],
        "type": "direct",
    },
    {
        "id": "G02",
        "question": "What information does Grad-CAM use to generate its localization map?",
        "expected_sources": ["Grad_CAM.pdf"],
        "expected_pages": [1],
        "type": "direct",
    },
    {
        "id": "G03",
        "question": "What does a Grad-CAM heatmap indicate about an image?",
        "expected_sources": ["Grad_CAM.pdf"],
        "expected_pages": [1, 2],
        "type": "paraphrased",
    },
    {
        "id": "G04",
        "question": "Why is Grad-CAM applicable to CNNs with fully connected layers?",
        "expected_sources": ["Grad_CAM.pdf"],
        "expected_pages": [1, 2],
        "type": "direct",
    },
    {
        "id": "G05",
        "question": "What are the two properties of a good visual explanation identified by the paper?",
        "expected_sources": ["Grad_CAM.pdf"],
        "expected_pages": [2],
        "type": "direct",
    },
    {
        "id": "G06",
        "question": "How does Guided Grad-CAM combine Grad-CAM with pixel-space visualizations?",
        "expected_sources": ["Grad_CAM.pdf"],
        "expected_pages": [2],
        "type": "paraphrased",
    },
    {
        "id": "G07",
        "question": "What does the paper report about Grad-CAM's localization performance compared with other visualization methods?",
        "expected_sources": ["Grad_CAM.pdf"],
        "expected_pages": [7],
        "type": "direct",
    },


    # --------------------------------------------------------
    # Integrated Gradients
    # --------------------------------------------------------

    {
        "id": "I01",
        "question": "What is the attribution problem studied by Integrated Gradients?",
        "expected_sources": ["integrated_axiomatic.pdf"],
        "expected_pages": [1],
        "type": "direct",
    },
    {
        "id": "I02",
        "question": "What are the two fundamental axioms proposed in the paper?",
        "expected_sources": ["integrated_axiomatic.pdf"],
        "expected_pages": [1, 2],
        "type": "direct",
    },
    {
        "id": "I03",
        "question": "What is the Sensitivity axiom?",
        "expected_sources": ["integrated_axiomatic.pdf"],
        "expected_pages": [2],
        "type": "direct",
    },
    {
        "id": "I04",
        "question": "Why can ordinary gradients violate Sensitivity?",
        "expected_sources": ["integrated_axiomatic.pdf"],
        "expected_pages": [2],
        "type": "paraphrased",
    },
    {
        "id": "I05",
        "question": "What does Implementation Invariance require?",
        "expected_sources": ["integrated_axiomatic.pdf"],
        "expected_pages": [2],
        "type": "direct",
    },
    {
        "id": "I06",
        "question": "Why is a baseline input needed for attribution?",
        "expected_sources": ["integrated_axiomatic.pdf"],
        "expected_pages": [1, 2],
        "type": "paraphrased",
    },
    {
        "id": "I07",
        "question": "What kinds of applications does the paper demonstrate for Integrated Gradients?",
        "expected_sources": ["integrated_axiomatic.pdf"],
        "expected_pages": [1, 6, 7],
        "type": "direct",
    },


    # --------------------------------------------------------
    # LIME
    # --------------------------------------------------------

    {
        "id": "L01",
        "question": "What problem is LIME designed to address?",
        "expected_sources": ["LIME.pdf"],
        "expected_pages": [1],
        "type": "direct",
    },
    {
        "id": "L02",
        "question": "How does LIME explain an individual prediction?",
        "expected_sources": ["LIME.pdf"],
        "expected_pages": [1, 2],
        "type": "paraphrased",
    },
    {
        "id": "L03",
        "question": "What does it mean for LIME to learn an interpretable model locally around a prediction?",
        "expected_sources": ["LIME.pdf"],
        "expected_pages": [1, 2],
        "type": "paraphrased",
    },
    {
        "id": "L04",
        "question": "What is SP-LIME and what problem does it address?",
        "expected_sources": ["LIME.pdf"],
        "expected_pages": [1],
        "type": "direct",
    },
    {
        "id": "L05",
        "question": "Why does the paper distinguish between trusting an individual prediction and trusting a model?",
        "expected_sources": ["LIME.pdf"],
        "expected_pages": [1, 2],
        "type": "direct",
    },
    {
        "id": "L06",
        "question": "How can explanations help identify problems such as data leakage or dataset shift?",
        "expected_sources": ["LIME.pdf"],
        "expected_pages": [2],
        "type": "paraphrased",
    },
    {
        "id": "L07",
        "question": "What does the paper report about using LIME to understand image-classification predictions?",
        "expected_sources": ["LIME.pdf"],
        "expected_pages": [1, 2],
        "type": "direct",
    },


    # --------------------------------------------------------
    # NGNDAI-2026 paper
    # --------------------------------------------------------

    {
        "id": "N01",
        "question": "What dataset is used in the NGNDAI-2026 study, and how many image patches does it contain?",
        "expected_sources": ["NGNDAI-2026_Paper_625.pdf"],
        "expected_pages": [4],
        "type": "direct",
    },
    {
        "id": "N02",
        "question": "What nine tissue classes are considered in the study?",
        "expected_sources": ["NGNDAI-2026_Paper_625.pdf"],
        "expected_pages": [4],
        "type": "direct",
    },
    {
        "id": "N03",
        "question": "What train/validation/test split is used, and what random seed is specified?",
        "expected_sources": ["NGNDAI-2026_Paper_625.pdf"],
        "expected_pages": [4],
        "type": "direct",
    },
    {
        "id": "N04",
        "question": "How are DenseNet-121 and EfficientNet-B5 initialized and trained in the two phases?",
        "expected_sources": ["NGNDAI-2026_Paper_625.pdf"],
        "expected_pages": [5],
        "type": "direct",
    },
    {
        "id": "N05",
        "question": "What are XCS, GLAS, and LIME Stability intended to measure?",
        "expected_sources": ["NGNDAI-2026_Paper_625.pdf"],
        "expected_pages": [3, 12],
        "type": "paraphrased",
    },
    {
        "id": "N06",
        "question": "What does the study report about DenseNet-121 and EfficientNet-B5 test accuracy?",
        "expected_sources": ["NGNDAI-2026_Paper_625.pdf"],
        "expected_pages": [1, 8, 9],
        "type": "direct",
    },
    {
        "id": "N07",
        "question": "What does the TUM true-positive, false-positive, and false-negative analysis show about explanation localization?",
        "expected_sources": ["NGNDAI-2026_Paper_625.pdf"],
        "expected_pages": [9, 10, 11],
        "type": "paraphrased",
    },


    # --------------------------------------------------------
    # Cross-paper
    # --------------------------------------------------------

    {
        "id": "X01",
        "question": "How does DenseNet's feature connectivity differ from EfficientNet's approach to improving model efficiency?",
        "expected_sources": [
            "Dense_Net.pdf",
            "Efficient_Net.pdf"
        ],
        "expected_pages": [1, 2],
        "type": "cross-paper",
    },
    {
        "id": "X02",
        "question": "How do Grad-CAM, LIME, and Integrated Gradients differ in the type of explanation they provide?",
        "expected_sources": [
            "Grad_CAM.pdf",
            "LIME.pdf",
            "integrated_axiomatic.pdf"
        ],
        "expected_pages": [1, 2],
        "type": "cross-paper",
    },
    {
        "id": "X03",
        "question": "How does the NGNDAI-2026 paper use DenseNet, EfficientNet, Grad-CAM, LIME, and Integrated Gradients together?",
        "expected_sources": [
            "NGNDAI-2026_Paper_625.pdf",
            "Dense_Net.pdf",
            "Efficient_Net.pdf",
            "Grad_CAM.pdf",
            "LIME.pdf",
            "integrated_axiomatic.pdf"
        ],
        "expected_pages": [1, 3, 4, 5],
        "type": "cross-paper",
    },
    {
        "id": "X04",
        "question": "What is the conceptual difference between Grad-CAM's spatial activation approach and LIME's local surrogate approach?",
        "expected_sources": [
            "Grad_CAM.pdf",
            "LIME.pdf"
        ],
        "expected_pages": [1, 2],
        "type": "cross-paper",
    },


    # --------------------------------------------------------
    # Out-of-scope
    # --------------------------------------------------------

    {
        "id": "O01",
        "question": "What is the capital of France?",
        "expected_sources": [],
        "expected_pages": [],
        "type": "out-of-scope",
    },
    {
        "id": "O02",
        "question": "How does Bitcoin mining work?",
        "expected_sources": [],
        "expected_pages": [],
        "type": "out-of-scope",
    },
    {
        "id": "O03",
        "question": "Who won the 2022 FIFA World Cup?",
        "expected_sources": [],
        "expected_pages": [],
        "type": "out-of-scope",
    },
    {
        "id": "O04",
        "question": "What is the current stock price of NVIDIA?",
        "expected_sources": [],
        "expected_pages": [],
        "type": "out-of-scope",
    },
]


K = 5

REFUSAL_TEXT = "I don't know based on the provided documents."

RESULTS_FILE = "eval_results.json"


# ============================================================
# Helper functions
# ============================================================

def normalize_source(source):
    """
    Normalize source names so small differences such as
    '(1)' in an uploaded filename do not break evaluation.
    """

    if not source:
        return ""

    source = Path(str(source)).name

    # Normalize uploaded copy of the user's paper
    if source.startswith("NGNDAI-2026_Paper_625"):
        return "NGNDAI-2026_Paper_625.pdf"

    return source


# def get_page(doc):
#     """
#     Try common metadata keys used by PDF loaders.

#     Returns None if the page number is unavailable.
#     """

#     metadata = doc.metadata or {}

#     for key in ["page", "page_number", "page_num"]:
#         value = metadata.get(key)

#         if value is not None:
#             try:
#                 # Most PDF libraries use zero-based page numbers.
#                 return int(value) + 1
#             except (TypeError, ValueError):
#                 pass

#     return None

def get_page(doc):
    page = doc.metadata.get("page")
    if page is None:
        return None
    return int(page)

def reciprocal_rank(sources, expected_sources):
    """
    MRR contribution for one question.

    For multi-paper questions, rank 1 is assigned when
    any expected paper is first encountered.
    """

    expected = {
        normalize_source(source)
        for source in expected_sources
    }

    for rank, source in enumerate(sources, start=1):
        if normalize_source(source) in expected:
            return 1.0 / rank

    return 0.0


def page_hit(chunks, question):
    """
    Check whether any retrieved chunk comes from an expected
    source AND an expected page.
    """

    expected_sources = {
        normalize_source(source)
        for source in question["expected_sources"]
    }

    expected_pages = set(question["expected_pages"])

    for doc, score in chunks:

        source = normalize_source(
            doc.metadata.get("source")
        )

        page = get_page(doc)

        if (
            source in expected_sources
            and page in expected_pages
        ):
            return True

    return False


# ============================================================
# Main evaluation
# ============================================================

def main():

    db = load_db()

    paper_hits = 0
    page_hits = 0
    mrr_total = 0.0

    in_total = 0
    out_total = 0

    correct_refusals = 0
    false_refusals = 0

    results = []

    print("\n" + "=" * 70)
    print("RAG EVALUATION")
    print("=" * 70)

    print(f"Questions: {len(QUESTIONS)}")
    print(f"K: {K}")
    print(f"Model: {MODEL_NAME}")

    # --------------------------------------------------------
    # Retrieval + refusal evaluation
    # --------------------------------------------------------

    print("\n--- RETRIEVAL / REFUSAL EVALUATION ---")

    for question in QUESTIONS:

        qid = question["id"]
        text = question["question"]
        qtype = question["type"]

        chunks = retrieve(
            db,
            text,
            K
        )

        sources = [
            normalize_source(
                doc.metadata.get("source")
            )
            for doc, score in chunks
        ]

        pages = [
            get_page(doc)
            for doc, score in chunks
        ]

        result = {
            "id": qid,
            "question": text,
            "type": qtype,
            "expected_sources": question["expected_sources"],
            "expected_pages": question["expected_pages"],
            "retrieved_sources": sources,
            "retrieved_pages": pages,
        }

        # ----------------------------------------------------
        # In-corpus
        # ----------------------------------------------------

        if qtype != "out-of-scope":

            in_total += 1

            expected_sources = {
                normalize_source(source)
                for source in question["expected_sources"]
            }

            paper_hit = any(
                source in expected_sources
                for source in sources
            )

            page_hit_result = page_hit(
                chunks,
                question
            )

            mrr = reciprocal_rank(
                sources,
                question["expected_sources"]
            )

            if paper_hit:
                paper_hits += 1

            if page_hit_result:
                page_hits += 1

            mrr_total += mrr

            result["paper_hit_at_k"] = paper_hit
            result["page_hit_at_k"] = page_hit_result
            result["reciprocal_rank"] = mrr

            status = "PASS" if paper_hit else "FAIL"

            print(
                f"{status}: {qid} | {text}"
            )

            print(
                f"     Sources: {sources}"
            )

            print(
                f"     Pages:   {pages}"
            )

        # ----------------------------------------------------
        # Out-of-scope
        # ----------------------------------------------------

        else:

            out_total += 1

            prompt = build_prompt(
                text,
                chunks
            )

            answer = call_llm(prompt)

            refused = (
                answer.strip() == REFUSAL_TEXT
            )

            if refused:
                correct_refusals += 1
            else:
                false_refusals += 1

            result["answer"] = answer
            result["refused"] = refused

            status = (
                "PASS"
                if refused
                else "FAIL"
            )

            print(
                f"{status}: {qid} | {text}"
            )

            print(
                f"     Answer: {answer}"
            )

        results.append(result)


    # --------------------------------------------------------
    # RAG VS NO CONTEXT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("--- RAG VS NO CONTEXT ---")
    print("=" * 70)

    # Use a small representative sample rather than
    # running two LLM calls for all 46 questions.

    comparison_ids = [
        "D01",
        "D02",
        "E02",
        "G01",
        "L01",
        "I02",
    ]

    rag_comparison = []

    for qid in comparison_ids:

        question = next(
            q for q in QUESTIONS
            if q["id"] == qid
        )

        text = question["question"]

        chunks = retrieve(
            db,
            text,
            K
        )

        # With RAG context
        rag_prompt = build_prompt(
            text,
            chunks
        )

        rag_answer = call_llm(
            rag_prompt
        )

        # Without RAG context
        no_context_prompt = f"""
Answer this question using your general knowledge:

{text}
"""

        no_context_answer = call_llm(
            no_context_prompt
        )

        print("\nQUESTION:", text)

        print("\nWITH CONTEXT:")
        print(rag_answer)

        print("\nWITHOUT CONTEXT:")
        print(no_context_answer)

        rag_comparison.append({
            "id": qid,
            "question": text,
            "with_context": rag_answer,
            "without_context": no_context_answer,
        })


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    paper_hit_rate = (
        paper_hits / in_total
        if in_total
        else 0.0
    )

    page_hit_rate = (
        page_hits / in_total
        if in_total
        else 0.0
    )

    mrr = (
        mrr_total / in_total
        if in_total
        else 0.0
    )

    refusal_rate = (
        correct_refusals / out_total
        if out_total
        else 0.0
    )

    false_refusal_rate = (
        false_refusals / in_total
        if in_total
        else 0.0
    )


    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    summary = {
        "number_of_questions": len(QUESTIONS),
        "in_corpus_questions": in_total,
        "out_of_scope_questions": out_total,
        "k": K,
        "paper_hit_at_5": paper_hit_rate,
        "page_hit_at_5": page_hit_rate,
        "mrr": mrr,
        "refusal_rate": refusal_rate,
        "false_refusal_rate": false_refusal_rate,
        "model": MODEL_NAME,
    }

    print("\n" + "=" * 70)
    print("--- FINAL SUMMARY ---")
    print("=" * 70)

    print(
        "Number of questions:",
        len(QUESTIONS)
    )

    print(
        "In-corpus questions:",
        in_total
    )

    print(
        "Out-of-scope questions:",
        out_total
    )

    print(
        "K:",
        K
    )

    print(
        f"Paper Hit@{K}: "
        f"{paper_hit_rate:.2%}"
    )

    print(
        f"Page Hit@{K}: "
        f"{page_hit_rate:.2%}"
    )

    print(
        f"MRR: "
        f"{mrr:.4f}"
    )

    print(
        f"Refusal rate: "
        f"{refusal_rate:.2%}"
    )

    print(
        f"False-refusal rate: "
        f"{false_refusal_rate:.2%}"
    )

    print(
        "Correct refusals:",
        correct_refusals,
        "/",
        out_total
    )

    print(
        "False refusals:",
        false_refusals,
        "/",
        in_total
    )

    print(
        "Model:",
        MODEL_NAME
    )


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output = {
        "summary": summary,
        "results": results,
        "rag_vs_no_context": rag_comparison,
    }

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nDetailed results saved to: "
        f"{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()