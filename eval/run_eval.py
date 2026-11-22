"""Compare retrievers on the evaluation set and write a markdown report.

Metrics (k = 5):
- precision@5: share of the top 5 tickets that satisfy every constraint in the question
- hit@5:       1 if at least one of the top 5 is relevant
- MRR:         1 / rank of the first relevant ticket (0 if none in the top 5)

Run:
    python -m eval.run_eval                 # TF-IDF and filtered TF-IDF (fast, no downloads)
    python -m eval.run_eval --dense         # also Sentence Transformers (downloads all-MiniLM-L6-v2)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rag.data import load_tickets
from rag.retrievers import FilteredRetriever, TfidfRetriever

EVAL_SET = Path(__file__).with_name("eval_set.jsonl")
RESULTS = Path(__file__).resolve().parents[1] / "results"
K = 5


def load_eval_set() -> list[dict]:
    return [json.loads(line) for line in EVAL_SET.read_text().splitlines() if line.strip()]


def score_retriever(retriever, questions: list[dict], k: int = K) -> dict:
    per_kind: dict[str, dict[str, list[float]]] = {}
    for q in questions:
        relevant = set(q["relevant_ids"])
        ids = [h.ticket_id for h in retriever.search(q["question"], k=k)]
        flags = [i in relevant for i in ids]
        precision = sum(flags) / k
        hit = float(any(flags))
        mrr = next((1 / (r + 1) for r, f in enumerate(flags) if f), 0.0)
        bucket = per_kind.setdefault(q["kind"], {"precision": [], "hit": [], "mrr": []})
        for name, value in (("precision", precision), ("hit", hit), ("mrr", mrr)):
            bucket[name].append(value)
    return {kind: {m: float(np.mean(v)) for m, v in metrics.items()} | {"n": len(metrics["hit"])}
            for kind, metrics in per_kind.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", action="store_true", help="include the Sentence Transformers retriever")
    parser.add_argument("--llm-parser", action="store_true", help="also filter with Claude-extracted filters (needs ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    df = load_tickets()
    questions = load_eval_set()
    tfidf = TfidfRetriever(df)
    retrievers = [tfidf, FilteredRetriever(df, tfidf)]
    if args.dense:
        from rag.retrievers import DenseRetriever  # noqa: PLC0415

        dense = DenseRetriever(df)
        retrievers += [dense, FilteredRetriever(df, dense)]
    if args.llm_parser:
        import functools  # noqa: PLC0415

        import anthropic  # noqa: PLC0415

        from rag.answer import DEFAULT_MODEL  # noqa: PLC0415
        from rag.query_parser import parse_filters_llm  # noqa: PLC0415

        cache: dict[str, dict] = {}

        def llm_parser(question, values, _client=anthropic.Anthropic()):
            if question not in cache:  # one API call per question, reused across retrievers
                cache[question] = parse_filters_llm(question, values, _client, DEFAULT_MODEL)
            return cache[question]

        retrievers.append(FilteredRetriever(df, tfidf, parser=llm_parser, parser_name="claude"))

    results = {r.name: score_retriever(r, questions) for r in retrievers}

    # Random baseline: expected precision if 5 tickets were picked at random.
    base = {}
    for q in questions:
        base.setdefault(q["kind"], []).append(q["n_relevant"] / len(df))
    baseline = {kind: float(np.mean(v)) for kind, v in base.items()}

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval_eval.json").write_text(json.dumps({"k": K, "results": results, "random_precision": baseline}, indent=2))

    lines = [f"# Retrieval evaluation (k = {K})", "",
             f"{len(questions)} questions with labels derived from the data (see `eval/build_eval_set.py`).", ""]
    for kind in ("literal", "paraphrased"):
        n = next(iter(results.values()))[kind]["n"]
        lines += [f"## {kind.capitalize()} questions (n = {n})", "",
                  "| Retriever | precision@5 | hit@5 | MRR |", "|---|---:|---:|---:|"]
        for name, res in results.items():
            r = res[kind]
            lines.append(f"| {name} | {r['precision']:.3f} | {r['hit']:.3f} | {r['mrr']:.3f} |")
        lines += [f"| random (expected) | {baseline[kind]:.3f} | | |", ""]
    (RESULTS / "retrieval_eval.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
