"""Tests for scripts/check_ratios.py (the CI compression-ratio gate)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "scripts" / "check_ratios.py"
_spec = importlib.util.spec_from_file_location("check_ratios", _path)
cr = importlib.util.module_from_spec(_spec)
sys.modules["check_ratios"] = cr  # dataclasses look the module up
_spec.loader.exec_module(cr)


def doc(**entries):
    return {"format": 1, "entries": {k: {"values": v[0], "bits": v[1]} for k, v in entries.items()}}


def test_bits_per_value():
    assert cr.bits_per_value({"values": 4, "bits": 10}) == 2.5
    assert cr.bits_per_value({"values": 0, "bits": 0}) == 0.0


def test_identical_is_clean():
    r = cr.compare(doc(a=(1000, 5000)), doc(a=(1000, 5000)))
    assert r.failures == [] and r.changes == []


def test_growth_within_tolerance_passes_but_is_reported():
    r = cr.compare(doc(a=(1000, 10_000)), doc(a=(1000, 10_050)))  # +0.5% exactly
    assert r.failures == []
    assert [c.name for c in r.changes] == ["a"]
    assert r.changes[0].relative == pytest.approx(0.005)


def test_growth_beyond_tolerance_fails():
    r = cr.compare(doc(a=(1000, 10_000)), doc(a=(1000, 10_051)))
    assert r.failures == ["a: 10.000 -> 10.051 bits/value (+0.51%)"]


def test_improvements_pass_and_are_reported():
    r = cr.compare(doc(a=(1000, 10_000)), doc(a=(1000, 9_000)))
    assert r.failures == []
    assert r.changes[0].relative == pytest.approx(-0.1)


def test_missing_entry_fails_new_entry_is_reported():
    r = cr.compare(doc(a=(10, 10), b=(10, 10)), doc(a=(10, 10), c=(10, 20)))
    assert r.failures == ["b: missing from the new results"]
    assert [(c.name, c.old_bpv, c.new_bpv) for c in r.changes] == [("b", 1.0, None), ("c", None, 2.0)]


def test_value_count_change_fails():
    r = cr.compare(doc(a=(1000, 10_000)), doc(a=(999, 9_990)))
    assert r.failures == ["a: value count changed 1000 -> 999 (dataset changed?)"]


def test_compares_bits_per_value_not_raw_bits():
    # Same bits/value from a different count is still flagged by the count check only.
    r = cr.compare(doc(a=(2, 4)), doc(a=(2, 4)))
    assert r.changes == []


def test_markdown_table():
    r = cr.compare(doc(a=(1000, 10_000), b=(1000, 2_000)), doc(a=(1000, 9_000), b=(1000, 2_100)))
    md = r.markdown()
    assert md.splitlines()[:4] == [
        "### Compression ratio check: 1 regression(s) beyond 0.5%",
        "",
        "| entry | baseline bits/value | now | change |",
        "| --- | ---: | ---: | ---: |",
    ]
    assert "| a | 10.000 | 9.000 | -10.00% |" in md
    assert "| b | 2.000 | 2.100 | +5.00% :x: |" in md
    clean = cr.compare(doc(a=(1, 1)), doc(a=(1, 1))).markdown()
    assert clean == "### Compression ratio check: no changes"


def test_main_exit_codes_and_summary(tmp_path, monkeypatch, capsys):
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    old.write_text(json.dumps(doc(a=(1000, 10_000))))
    new.write_text(json.dumps(doc(a=(1000, 10_000))))
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    assert cr.main([str(old), str(new)]) == 0
    assert summary.read_text() == "### Compression ratio check: no changes\n"
    new.write_text(json.dumps(doc(a=(1000, 20_000))))
    assert cr.main([str(old), str(new)]) == 1
    assert "regression" in capsys.readouterr().out
    assert cr.main([str(old)]) == 2


def test_rejects_unknown_format(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"format": 2, "entries": {}}))
    assert cr.main([str(p), str(p)]) == 2


def test_entries_after_a_missing_or_new_one_are_still_checked():
    r = cr.compare(doc(a=(10, 10), z=(1000, 1000)), doc(b=(10, 10), z=(1000, 2000)))
    assert r.failures == ["a: missing from the new results", "z: 1.000 -> 2.000 bits/value (+100.00%)"]
    r = cr.compare(doc(a=(10, 10), z=(10, 10)), doc(a=(5, 5), z=(10, 20)))
    assert r.failures[1] == "z: 1.000 -> 2.000 bits/value (+100.00%)"


def test_markdown_lists_failures_new_and_missing():
    md = cr.compare(doc(a=(10, 10)), doc(a=(5, 5))).markdown()
    assert md.splitlines()[0] == "### Compression ratio check: 0 regression(s) beyond 0.5%"
    assert md.endswith("\n\nFailures:\n- a: value count changed 10 -> 5 (dataset changed?)")
    md = cr.compare(doc(a=(10, 10)), doc(b=(10, 20))).markdown()
    assert "| a | 1.000 | – | missing :x: |" in md
    assert "| b | – | 2.000 | new |" in md


def test_main_reports_usage_and_errors_on_stderr(tmp_path, capsys):
    assert cr.main([]) == 2
    assert "check_ratios.py bench/expected-ratios.json" in capsys.readouterr().err
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"format": 2, "entries": {}}))
    assert cr.main([str(p), str(p)]) == 2
    assert capsys.readouterr().err.strip() == f"{p}: unknown format 2"
    assert cr.main([str(tmp_path / "nope.json"), str(p)]) == 2
    assert "nope.json" in capsys.readouterr().err
