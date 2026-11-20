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

Then open `operation_intelligence_RAG_demo.ipynb` in Jupyter or VS Code.

The first notebook cells build a 10K-ticket sample from the full 200K-ticket dataset, which is not included in this repository because of its size. The prepared sample, `support_tickets_rag_sample_10k.csv`, is included, so you can start from the cell that loads it and run the embedding, indexing and search steps from there.

## Current Scope and Next Steps

- **Current:** semantic retrieval plus a structured, evidence-based summary computed from the retrieved tickets.
- **Next:** pass the retrieved tickets to an LLM to generate a natural-language answer that cites ticket IDs, completing the generation step of the RAG pipeline.
- **Next:** add a small evaluation set of questions with expected tickets to measure retrieval quality (for example, hit rate at 5).
