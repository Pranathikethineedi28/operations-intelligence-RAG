# Operations Intelligence RAG Assistant

This project is a Retrieval-Augmented Generation (RAG) based operations analytics assistant built on a customer support ticket dataset. It enables users to ask natural language questions about operational issues such as SLA breaches, escalations, customer dissatisfaction, recurring product issues, and resolution delays.

## Project Objective

The goal of this project is to make support operations data easier to query, analyze, and summarize. Instead of manually filtering thousands of support tickets, users can ask questions in plain English and retrieve relevant ticket records along with structured business insights.

## Tech Stack

- Python
- pandas
- sentence-transformers
- FAISS
- NLP
- Retrieval-Augmented Generation
- Claude API (answer generation and filter extraction)
- pytest and GitHub Actions

## Key Features

- Natural language querying over support ticket data
- Semantic retrieval using FAISS
- Root cause style summaries for operational issues
- Evidence-based ticket retrieval
- SLA breach and escalation analysis
- Customer dissatisfaction pattern identification

## Business Value

This project demonstrates how RAG can support operations teams by reducing manual ticket review, surfacing recurring issues, and improving access to root cause insights from unstructured support data.

## How It Works

1. **Documents:** each support ticket becomes one structured text document (product, category, priority, channel, region, SLA and escalation flags, response and resolution times, satisfaction score).
2. **Embeddings:** documents are embedded with `sentence-transformers/all-MiniLM-L6-v2` using normalized vectors.
3. **Index:** vectors are stored in a FAISS `IndexFlatIP` index, so inner-product search equals cosine similarity.
4. **Retrieval:** a plain-English question is embedded the same way and the top matching tickets are returned as evidence.
5. **Summary:** the retrieved tickets are summarized with counts and averages (SLA breaches, escalations, response and resolution times, satisfaction) and their most common products, categories, channels and regions.

## How to Run

```
pip install -r requirements.txt
```

The original exploration is in `operation_intelligence_RAG_demo.ipynb`. Its first cells build the 10K-ticket sample from a 200K-ticket dataset that is not included because of its size; the prepared sample, `support_tickets_rag_sample_10k.csv`, is included.

The reusable pipeline lives in the `rag/` package:

```
python -m eval.run_eval                  # retrieval evaluation: TF-IDF vs. filtered retrieval (no downloads)
python -m eval.run_eval --dense          # add Sentence Transformers + FAISS (downloads all-MiniLM-L6-v2)
python -m eval.run_eval --llm-parser     # add Claude-extracted filters (needs ANTHROPIC_API_KEY)

export ANTHROPIC_API_KEY=...
python -m rag.answer "Why are urgent Payment Gateway tickets breaching SLA?"
pytest -q
```

## Answer Generation with Grounded Citations

`rag/answer.py` completes the RAG pipeline: it retrieves the most relevant tickets, sends them to Claude with instructions to answer **only** from those tickets, and requires a citation such as `[T119738]` for every claim.

A citation check then runs on the answer. Every cited ID is compared with the IDs that were actually retrieved: citations to tickets the model was never shown are flagged, and an answer without valid citations is marked as not grounded.

## Data Audit

Before evaluating retrieval, I checked what the text fields actually contain. The dataset has only **10 unique issue descriptions**, assigned to categories essentially at random: for a given description, the most common category covers just 11.5% of its tickets (random assignment across 10 categories would give 10%). The free text therefore carries almost no information about a ticket's category, product or priority. Questions about those attributes have to be answered from the **structured fields**, which shaped the design below.

## Retrieval Evaluation

`eval/build_eval_set.py` creates 90 questions with known answers. Each question constrains 2 to 4 structured fields (for example product, priority and SLA status), and a retrieved ticket counts as relevant only if it matches **all** of them, so the labels come directly from the data.

- **Literal questions** use the exact field values ("Urgent", "Payment Gateway").
- **Paraphrased questions** use everyday wording ("critical", "called-in", "US and Canada").

Three approaches are compared at k = 5:

| Retriever | Literal precision@5 | Literal MRR | Paraphrased precision@5 | Paraphrased MRR |
|---|---:|---:|---:|---:|
| Random (expected) | 0.026 | | 0.016 | |
| TF-IDF | 0.563 | 0.762 | 0.353 | 0.502 |
| Filtered (rule-based parser) + TF-IDF | **1.000** | **1.000** | 0.353 | 0.502 |

Full results, including hit@5, are in [`results/retrieval_eval.md`](results/retrieval_eval.md).

**What the numbers show**

1. **Similarity search alone mixes constraints.** TF-IDF finds tickets that match *some* of the question's conditions, so only 56% of its top 5 match all of them.
2. **Metadata filtering fixes that when the wording matches the data.** Extracting filters from the question first and ranking only the matching tickets gives perfect precision on literal questions. This is the "hybrid search" pattern used in production retrieval systems.
3. **Rule-based parsing breaks on paraphrases.** "Critical" is not recognized as "Urgent", so filtering stops helping and precision drops to 0.35. This is the motivation for the `--llm-parser` option, where Claude maps everyday wording onto the allowed values. Its output is validated against the real vocabulary, so a hallucinated value can never become a filter.

**Caveat:** the literal score is an upper bound, because those questions use exactly the vocabulary the rule-based parser looks for. The paraphrased set is the more realistic test.

## Next Steps

- Run `--dense` and `--llm-parser` and add their rows to the table above.
- Evaluate answer quality, not only retrieval: check citation validity and numeric accuracy over the evaluation questions.
- Expand the paraphrased set with real user phrasings.
