from types import SimpleNamespace

import pytest

from eval.run_eval import score_retriever
from rag.answer import answer_question, check_citations
from rag.data import field_values, load_tickets
from rag.query_parser import parse_filters, parse_filters_llm
from rag.retrievers import FilteredRetriever, TfidfRetriever


@pytest.fixture(scope="module")
def df():
    return load_tickets()


@pytest.fixture(scope="module")
def values(df):
    return field_values(df)


class FakeClaude:
    """Stands in for anthropic.Anthropic(): returns a fixed reply and records the prompt."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.reply)])


def test_rule_parser_reads_literal_values(values):
    q = "urgent Payment Gateway tickets about Login Issue where the SLA was breached"
    assert parse_filters(q, values) == {
        "priority": "Urgent", "product": "Payment Gateway", "category": "Login Issue", "sla_breached": "Yes",
    }


def test_rule_parser_handles_negation(values):
    assert parse_filters("tickets that were not escalated", values) == {"escalated": "No"}


def test_rule_parser_misses_paraphrases(values):
    # The known limitation the evaluation measures: "critical" is not mapped to "Urgent".
    assert "priority" not in parse_filters("critical tickets", values)


def test_filtered_retriever_only_returns_matching_tickets(df):
    retriever = FilteredRetriever(df, TfidfRetriever(df))
    hits = retriever.search("High priority Mobile App tickets from Europe", k=5)
    assert len(hits) == 5
    for h in hits:
        row = df.iloc[h.row]
        assert (row["priority"], row["product"], row["region"]) == ("High", "Mobile App", "Europe")


def test_filtered_retriever_returns_nothing_when_no_ticket_matches(df):
    retriever = FilteredRetriever(df, TfidfRetriever(df), parser=lambda q, v: {"region": "Antarctica"})
    assert retriever.search("anything", k=5) == []


def test_llm_parser_drops_hallucinated_values(values):
    client = FakeClaude('{"priority": "Urgent", "region": "Mars", "mood": "angry"}')
    assert parse_filters_llm("critical tickets from Mars", values, client, "test-model") == {"priority": "Urgent"}


def test_llm_parser_survives_non_json_reply(values):
    assert parse_filters_llm("hello", values, FakeClaude("I cannot help with that."), "m") == {}


def test_citation_check_flags_tickets_that_were_not_retrieved():
    cited, invalid = check_citations("Delays [T1] and refunds [T2][T99]", retrieved_ids=[1, 2, 3])
    assert cited == [1, 2, 99] and invalid == [99]


def test_answer_marks_grounded_and_ungrounded(df):
    retriever = FilteredRetriever(df, TfidfRetriever(df))
    question = "Urgent API Service tickets where the SLA was breached"
    first_id = retriever.search(question, k=3)[0].ticket_id

    good = answer_question(question, df, retriever, FakeClaude(f"Mostly billing issues [T{first_id}]."), k=3)
    assert good.grounded and good.invalid_citations == []

    bad = answer_question(question, df, retriever, FakeClaude("Caused by outages [T1]."), k=3)
    assert not bad.grounded and bad.invalid_citations == [1]

    uncited = answer_question(question, df, retriever, FakeClaude("Caused by outages."), k=3)
    assert not uncited.grounded


def test_answer_sends_retrieved_tickets_to_the_model(df):
    retriever = FilteredRetriever(df, TfidfRetriever(df))
    client = FakeClaude("ok [T1]")
    result = answer_question("Low priority Chat tickets", df, retriever, client, k=2)
    prompt = client.calls[0]["messages"][0]["content"]
    assert all(f'id="T{i}"' in prompt for i in result.retrieved_ids)
    assert "ONLY the tickets provided" in client.calls[0]["system"]


def test_metrics_on_a_toy_retriever():
    class Fixed:
        def search(self, q, k):
            return [SimpleNamespace(ticket_id=i) for i in [9, 1, 9, 9, 9]]

    questions = [{"kind": "literal", "question": "x", "relevant_ids": [1]}]
    r = score_retriever(Fixed(), questions, k=5)["literal"]
    assert r["precision"] == pytest.approx(0.2) and r["hit"] == 1.0 and r["mrr"] == pytest.approx(0.5)
