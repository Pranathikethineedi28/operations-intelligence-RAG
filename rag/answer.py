"""The generation step: retrieve tickets, ask Claude to answer using only those tickets, verify citations.

Grounding guardrail: the model must cite ticket IDs like [T119738]. After the answer comes back,
every cited ID is checked against the tickets that were actually retrieved. Citations to tickets
that were not provided are flagged, and an answer with no valid citations is marked ungrounded.

Usage:
    export ANTHROPIC_API_KEY=...
    python -m rag.answer "Why are urgent Payment Gateway tickets breaching SLA?"
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field

import pandas as pd

from rag.data import load_tickets
from rag.retrievers import FilteredRetriever, Hit, TfidfRetriever

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5")
CITATION = re.compile(r"\[T(\d+)\]")

SYSTEM_PROMPT = """You are an operations analyst answering questions about customer-support tickets.
Use ONLY the tickets provided. Cite every claim with the ticket IDs it comes from, written as [T<id>],
for example [T119738]. If the tickets do not contain enough information to answer, say so plainly
instead of guessing. Report counts and averages only when you can compute them from the tickets shown."""


@dataclass
class Answer:
    question: str
    text: str
    retrieved_ids: list[int]
    cited_ids: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        return bool(self.cited_ids) and not self.invalid_citations


def build_context(df: pd.DataFrame, hits: list[Hit]) -> str:
    blocks = []
    for h in hits:
        text = df.at[h.row, "rag_text"].strip()
        blocks.append(f"<ticket id=\"T{h.ticket_id}\">\n{text}\n</ticket>")
    return "\n\n".join(blocks)


def check_citations(text: str, retrieved_ids: list[int]) -> tuple[list[int], list[int]]:
    cited = sorted({int(m) for m in CITATION.findall(text)})
    allowed = set(retrieved_ids)
    return cited, [c for c in cited if c not in allowed]


def answer_question(question: str, df: pd.DataFrame, retriever, client, k: int = 8, model: str = DEFAULT_MODEL) -> Answer:
    hits = retriever.search(question, k=k)
    ids = [h.ticket_id for h in hits]
    if not hits:
        return Answer(question, "No tickets matched this question, so I can't answer it from the data.", ids)

    message = client.messages.create(
        model=model,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"{build_context(df, hits)}\n\nQuestion: {question}"}],
    )
    text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
    cited, invalid = check_citations(text, ids)
    return Answer(question, text, ids, cited, invalid)


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a question about support tickets with citations.")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=8, help="number of tickets to retrieve")
    args = parser.parse_args()

    import anthropic  # noqa: PLC0415 - only needed when actually calling the API

    df = load_tickets()
    retriever = FilteredRetriever(df, TfidfRetriever(df))
    result = answer_question(args.question, df, retriever, anthropic.Anthropic(), k=args.k)
    print(result.text)
    print(f"\nRetrieved: {result.retrieved_ids}")
    print(f"Cited: {result.cited_ids}  |  invalid citations: {result.invalid_citations}  |  grounded: {result.grounded}")


if __name__ == "__main__":
    main()
