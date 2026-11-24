"""JSON output of the retrieval evaluation (app/evaluate_rag.py --json); scoring is unchanged."""

import json

from app import evaluate_rag as er


def fake_search(question, k):
    """Perfect retriever for in-scope rows, nothing for out-of-scope ones (no ChromaDB)."""

    row = next(r for r in er.load_evaluation_dataset() if r["question"] == question)
    if not row["expected_policy_ids"]:
        return []
    policy = row["expected_policy_ids"][0]
    sections = row["expected_sections"] or ["PURPOSE"]
    return [({"policy_id": policy, "section": s}, 0.5) for s in sections][:k]


def test_report_contains_exactly_the_printed_figures():
    records, summary, sweep = er.run_evaluation(fake_search)
    report = er.build_report(records, summary, sweep, {"created_utc": "x"})
    restored = json.loads(json.dumps(report))
    assert restored["summary"] == json.loads(json.dumps(summary))
    assert restored["run"]["settings"] == {"k": er.TOP_K, "max_distance": er.MAX_DISTANCE,
                                           "sweep_thresholds": list(er.SWEEP_THRESHOLDS)}
    assert set(restored["sweep"]) == {str(t) for t in er.SWEEP_THRESHOLDS}
    assert len(restored["records"]) == len(er.load_evaluation_dataset())
    assert restored["summary"]["in_scope"]["hit_rate"] == 1.0
    assert restored["summary"]["no_answer"]["false_positive_rate"] == 0.0


def test_cli_json_flag_writes_report_without_changing_output(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(er, "_store_search", fake_search)
    er.main([])
    plain = capsys.readouterr().out
    out = tmp_path / "retrieval.json"
    er.main(["--json", str(out)])
    with_json = capsys.readouterr().out
    assert with_json.startswith(plain) and "JSON report:" in with_json      # same report, plus the path
    saved = json.loads(out.read_text())
    assert saved["run"]["kind"] == "retrieval" and len(saved["run"]["dataset"]["sha256"]) == 64
    assert saved["run"]["policy_index_fingerprint"] and saved["run"]["embedding_id"]


def test_production_settings_and_thresholds_are_unchanged():
    assert (er.TOP_K, er.MAX_DISTANCE) == (3, 1.2)
    assert er.SWEEP_THRESHOLDS == (1.0, 1.1, 1.2, 1.3, 1.4)
