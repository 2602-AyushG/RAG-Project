## Evaluation

The RAG system is evaluated using a 50-question benchmark grounded in the six research papers stored in the vector database.

### Evaluation Dataset

The benchmark contains:

* **46 in-corpus questions** based on the six research papers
* **4 out-of-scope questions** that should be refused
* Questions covering:

  * Direct factual retrieval
  * Paraphrased questions
  * Cross-paper questions
  * Out-of-scope but potentially confusing questions

Each in-corpus question is associated with an expected source paper and expected page range.

### Evaluation Metrics

The evaluation reports the following metrics:

* **Paper Hit@5** — whether the expected paper appears among the top 5 retrieved chunks.
* **Page Hit@5** — whether a chunk from the expected page range appears among the top 5 retrieved chunks.
* **MRR (Mean Reciprocal Rank)** — measures how highly the expected source appears in the retrieval results.
* **Refusal Rate** — percentage of questions for which the model returns the required refusal response.
* **False-Refusal Rate** — percentage of valid in-corpus questions incorrectly refused by the model.

The evaluation also compares selected questions with and without retrieved context to examine the effect of RAG on the generated answers.

### Running the Evaluation

After the vector database has been created, run:

```bash
python eval.py
```

The evaluation prints retrieval results, generated answers, refusal statistics, and aggregate metrics to the terminal.

A detailed machine-readable copy of the evaluation results is also saved as:

```text
eval_results.json
```

### Current Evaluation Scope

The benchmark currently evaluates retrieval and answer generation across:

1. DenseNet
2. EfficientNet
3. Grad-CAM
4. Integrated Gradients
5. LIME
6. The project's comparative XAI study

The evaluation is intended to measure both **retrieval quality** and **grounded answer behavior**, rather than relying only on whether the language model produces a plausible answer.
