"""Load the ticket sample and expose the structured fields used for filtering and evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "support_tickets_rag_sample_10k.csv"

# Structured fields a question can constrain. Values are matched exactly in the data.
FILTER_FIELDS = ["product", "category", "priority", "channel", "region", "escalated", "sla_breached"]


def load_tickets(path: str | Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"ticket_id", "rag_text", *FILTER_FIELDS} - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    df["rag_text"] = df["rag_text"].fillna("")
    return df.reset_index(drop=True)


def field_values(df: pd.DataFrame) -> dict[str, list[str]]:
    """All distinct values of each filterable field, e.g. {'priority': ['High', 'Low', ...]}."""
    return {f: sorted(df[f].astype(str).unique()) for f in FILTER_FIELDS}
