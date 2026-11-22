"""Build a retrieval evaluation set with known correct answers.

Each question constrains 2-4 structured fields (product, category, priority, ...). A retrieved
ticket counts as relevant only if it matches ALL of the question's constraints, so relevance
labels come straight from the data rather than from human judgment.

Two kinds of questions:
- "literal": field values appear verbatim ("Urgent", "Payment Gateway").
- "paraphrased": the same constraints in everyday words ("critical", "payments gateway"),
  to test how retrievers cope when users don't use the exact vocabulary.

Run:  python -m eval.build_eval_set   ->  eval/eval_set.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from rag.data import load_tickets

OUT = Path(__file__).with_name("eval_set.jsonl")
SEED = 7
N_LITERAL = 60
N_PARAPHRASED = 30
MIN_RELEVANT = 5  # skip constraint combinations with fewer matching tickets than we retrieve

PARAPHRASE = {
    "priority": {"Urgent": "critical", "High": "high-severity", "Medium": "mid-priority", "Low": "minor"},
    "channel": {"Email": "e-mailed", "Phone": "called-in", "Chat": "live chat", "Web Form": "website form",
                "Social Media": "social"},
    "region": {"North America": "US and Canada", "South America": "Latin American", "Europe": "European",
               "Asia": "Asian", "Africa": "African", "Australia": "Australian"},
}


def phrase(field: str, value: str, paraphrase: bool) -> str:
    v = PARAPHRASE.get(field, {}).get(value, value) if paraphrase else value
    return {
        "product": f"{v}",
        "category": f"about {v.lower() if paraphrase else v}",
        "priority": f"{v} priority" if not paraphrase else v,
        "channel": f"submitted via {v}" if not paraphrase else v,
        "region": f"from {v}" if not paraphrase else v,
        "sla_breached": "where the SLA was breached" if value == "Yes" else "handled within the SLA",
        "escalated": "that were escalated" if value == "Yes" else "that were not escalated",
    }[field]


def make_question(constraints: dict[str, str], paraphrase: bool) -> str:
    adjectives = [phrase(f, constraints[f], paraphrase) for f in ("priority", "region", "channel") if f in constraints and paraphrase]
    parts = ["Show me"]
    parts += adjectives
    if "product" in constraints:
        parts.append(phrase("product", constraints["product"], paraphrase))
    parts.append("tickets")
    for f in ("category", "priority", "channel", "region", "sla_breached", "escalated"):
        if f in constraints and not (paraphrase and f in ("priority", "region", "channel")):
            parts.append(phrase(f, constraints[f], paraphrase))
    return " ".join(parts)


def main() -> None:
    df = load_tickets()
    rng = random.Random(SEED)
    fields = ["product", "category", "priority", "channel", "region", "sla_breached", "escalated"]
    rows, seen = [], set()

    for kind, target in (("literal", N_LITERAL), ("paraphrased", N_PARAPHRASED)):
        made = 0
        while made < target:
            anchor = df.iloc[rng.randrange(len(df))]
            pool = fields if kind == "literal" else ["priority", "channel", "region", "category", "product"]
            chosen = rng.sample(pool, rng.randint(2, 4 if kind == "literal" else 3))
            if kind == "paraphrased" and not any(f in PARAPHRASE for f in chosen):
                continue
            constraints = {f: str(anchor[f]) for f in chosen}
            key = (kind, tuple(sorted(constraints.items())))
            if key in seen:
                continue
            mask = (df[list(constraints)].astype(str) == pd_row(constraints)).all(axis=1)
            relevant = df.loc[mask, "ticket_id"].astype(int).tolist()
            if len(relevant) < MIN_RELEVANT:
                continue
            seen.add(key)
            rows.append({
                "id": f"{kind[0]}{made:03d}",
                "kind": kind,
                "question": make_question(constraints, paraphrase=(kind == "paraphrased")),
                "constraints": constraints,
                "n_relevant": len(relevant),
                "relevant_ids": relevant,
            })
            made += 1

    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} questions to {OUT}")


def pd_row(constraints: dict[str, str]):
    import pandas as pd  # noqa: PLC0415

    return pd.Series(constraints)


if __name__ == "__main__":
    main()
