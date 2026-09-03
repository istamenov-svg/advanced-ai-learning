"""Offline tests. Force stub mode so no API key is ever needed here."""
import os
os.environ["AIHC_STUB"] = "1"

import shutil  # noqa: E402

import pytest  # noqa: E402

from aihc import config  # noqa: E402
from aihc import data_gen  # noqa: E402
from aihc.metrics import quarter, query_metrics, compare_periods, anomalies, dataset_fingerprint  # noqa: E402
from aihc.rag.chunk import load_chunks  # noqa: E402
from aihc.rag.index import get_index  # noqa: E402
from aihc.rag.router import parse_metric_question  # noqa: E402
from aihc.rag.answer import answer  # noqa: E402
from aihc.evals.run_evals import main as run_evals  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def data():
    if not (config.DATA_DIR / "funnel_weekly.csv").exists():
        data_gen.write_all()
    shutil.rmtree(config.CACHE_DIR, ignore_errors=True)


def test_data_is_deterministic():
    a = data_gen.generate()[1]
    b = data_gen.generate()[1]
    assert a.equals(b)


def test_metrics_identities():
    t = query_metrics(quarter(2026, 2))["totals"]
    assert t["cpl"] == round(t["spend"] / t["leads"], 4)
    assert t["cps"] == round(t["spend"] / t["shows"], 4)
    assert 0 < t["show_rate"] < 1


def test_grouped_sums_match_total():
    q = quarter(2026, 2)
    total = query_metrics(q)["totals"]["spend"]
    by_region = query_metrics(q, group_by="region")["rows"]
    assert abs(sum(r["spend"] for r in by_region.values()) - total) < 0.05


def test_planted_anomaly_is_found():
    flags = anomalies(quarter(2026, 2), quarter(2026, 1))
    assert flags and flags[0]["cell"] == "South Florida | Paid Social"
    assert flags[0]["pct_change"] > 15


def test_compare_periods_shape():
    c = compare_periods(quarter(2026, 2), quarter(2026, 1))
    assert set(c["totals"]) >= {"cpl", "cps", "spend"}
    assert "pct_change" in c["totals"]["cpl"]


def test_chunk_ids_are_stable():
    assert [c.id for c in load_chunks()] == [c.id for c in load_chunks()]
    assert all("#" in c.id and len(c.id.split("#")[1]) == 8 for c in load_chunks())


def test_bm25_ranks_relevant_doc_first():
    hits = get_index().search("speed to lead first call attempt", 3)
    assert hits[0][0].source == "lead_followup_playbook.md"


def test_router():
    mq = parse_metric_question("What was cost per show in Q2 2026 for the Tampa region?")
    assert mq and mq.metric == "cps" and mq.region == "Tampa" and mq.period.label == "Q2 2026"
    assert parse_metric_question("What does the policy say about testimonials?") is None


def test_answer_is_cached_and_repeatable():
    q = "What was cost per show in Q2 2026 for the Tampa region?"
    a1 = answer(q)
    a2 = answer(q)
    assert a1.text == a2.text
    assert a2.cache_hit is True
    assert a1.facts["values"]["total"] == "$181.83"


def test_docs_answer_cites_valid_chunk():
    a = answer("What paid social CPL level triggers a budget hold?")
    assert a.route == "docs" and a.citations and a.citations_valid


def test_abstains_out_of_scope():
    a = answer("What is the company's policy on employee parking reimbursement?")
    assert a.text.startswith("Not in the provided documents")


def test_fingerprint_changes_when_data_changes(tmp_path):
    fp1 = dataset_fingerprint()
    p = config.DOCS_DIR / "kpi_definitions.md"
    original = p.read_bytes()          # bytes, not text: Windows would rewrite \n as \r\n and change the hash
    try:
        p.write_bytes(original + b"\nTemporary edit.\n")
        assert dataset_fingerprint() != fp1
    finally:
        p.write_bytes(original)
    assert dataset_fingerprint() == fp1


def test_eval_gates_pass():
    assert run_evals([]) == 0
