"""Turn a plain-English question into structured filters.

Example: "urgent Payment Gateway tickets about login issues where the SLA was breached"
-> {"priority": "Urgent", "product": "Payment Gateway", "category": "Login Issue", "sla_breached": "Yes"}

This is deliberately simple and transparent: exact (case-insensitive) phrase matching against the
values that actually exist in the data, plus a few phrasings for the yes/no fields. It is easy to
test and its failure mode is predictable: if a value is phrased differently, no filter is applied.
"""

from __future__ import annotations

import re

# Phrasings that map onto the yes/no columns.
FLAG_PATTERNS = {
    "sla_breached": {
        "Yes": [r"\bsla (was |were )?breach(ed|es)?\b", r"\bbreach(ed)? (the )?sla\b", r"\bmissed (the )?sla\b"],
        "No": [r"\bwithin (the )?sla\b", r"\bsla (was |were )?(met|not breached)\b"],
    },
    "escalated": {
        "Yes": [r"\bescalated\b", r"\bescalations?\b"],
        "No": [r"\bnot escalated\b", r"\bwithout escalation\b"],
    },
}


def parse_filters(question: str, values: dict[str, list[str]]) -> dict[str, str]:
    """Return {field: value} for every known value mentioned in the question."""
    q = question.lower()
    filters: dict[str, str] = {}

    for field, options in values.items():
        if field in FLAG_PATTERNS:
            continue
        # Prefer the longest match so "Payment Gateway" beats a shorter overlapping value.
        for option in sorted(options, key=len, reverse=True):
            if re.search(rf"\b{re.escape(option.lower())}\b", q):
                filters[field] = option
                break

    for field, by_value in FLAG_PATTERNS.items():
        # Check "No" first so "not escalated" is not read as "escalated".
        for value in ("No", "Yes"):
            if any(re.search(p, q) for p in by_value[value]):
                filters[field] = value
                break
    return filters


LLM_PARSER_PROMPT = """Extract search filters from a question about customer-support tickets.
Allowed fields and values (use these exact strings, nothing else):
{allowed}

Map everyday wording onto the allowed values (for example "critical" -> priority "Urgent",
"called in" -> channel "Phone"). Only include a field when the question clearly constrains it.
Reply with a single JSON object mapping field to value, and nothing else. Reply {{}} if there are no filters."""


def parse_filters_llm(question: str, values: dict[str, list[str]], client, model: str) -> dict[str, str]:
    """Ask Claude to map a question onto the allowed field values.

    Guardrail: the reply is parsed as JSON and every field/value is checked against the real
    vocabulary, so a hallucinated value can never become a filter.
    """
    import json  # noqa: PLC0415

    allowed = "\n".join(f"- {field}: {', '.join(opts)}" for field, opts in values.items())
    message = client.messages.create(
        model=model,
        max_tokens=200,
        system=LLM_PARSER_PROMPT.format(allowed=allowed),
        messages=[{"role": "user", "content": question}],
    )
    text = "".join(b.text for b in message.content if getattr(b, "type", "") == "text").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return {f: v for f, v in raw.items() if f in values and isinstance(v, str) and v in values[f]}
