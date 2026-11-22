# Retrieval evaluation (k = 5)

90 questions with labels derived from the data (see `eval/build_eval_set.py`).

## Literal questions (n = 60)

| Retriever | precision@5 | hit@5 | MRR |
|---|---:|---:|---:|
| tfidf | 0.563 | 0.967 | 0.762 |
| filtered(rules)+tfidf | 1.000 | 1.000 | 1.000 |
| random (expected) | 0.026 | | |

## Paraphrased questions (n = 30)

| Retriever | precision@5 | hit@5 | MRR |
|---|---:|---:|---:|
| tfidf | 0.353 | 0.700 | 0.502 |
| filtered(rules)+tfidf | 0.353 | 0.700 | 0.502 |
| random (expected) | 0.016 | | |
