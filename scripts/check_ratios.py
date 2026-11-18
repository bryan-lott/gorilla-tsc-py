#!/usr/bin/env python3
"""Compare compression ratios against a committed baseline.

    cargo run --release -p gorilla-tsc --example bench -- --ratios > ratios.json
    python3 scripts/check_ratios.py bench/expected-ratios.json ratios.json

Sizes are exact payload bits, so any change is real, not noise. Fails if an
entry grew by more than the tolerance or disappeared; prints a Markdown table
of every change (also to $GITHUB_STEP_SUMMARY when set). After an intended
change, regenerate the baseline with the first command and commit it.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from fractions import Fraction

TOLERANCE = 0.005  # 0.5% larger is a failure


@dataclass(frozen=True)
class Change:
    name: str
    old_bpv: float | None  # None: new entry
    new_bpv: float | None  # None: entry disappeared

    @property
    def relative(self) -> float | None:
        if self.old_bpv is None or self.new_bpv is None or self.old_bpv == 0:
            return None
        return self.new_bpv / self.old_bpv - 1


@dataclass(frozen=True)
class Report:
    changes: list[Change]
    failures: list[str]

    def markdown(self) -> str:
        if not self.changes and not self.failures:
            return "### Compression ratio check: no changes"
        regressions = sum(1 for f in self.failures if "bits/value" in f)
        lines = [
            f"### Compression ratio check: {regressions} regression(s) beyond {TOLERANCE:.1%}",
            "",
            "| entry | baseline bits/value | now | change |",
            "| --- | ---: | ---: | ---: |",
        ]
        for c in self.changes:
            old = "–" if c.old_bpv is None else f"{c.old_bpv:.3f}"
            new = "–" if c.new_bpv is None else f"{c.new_bpv:.3f}"
            rel = c.relative
            if rel is None:
                delta = "new" if c.old_bpv is None else "missing :x:"
            else:
                failed = any(f.startswith(f"{c.name}: ") and "bits/value" in f for f in self.failures)
                delta = f"{rel:+.2%}" + (" :x:" if failed else "")
            lines.append(f"| {c.name} | {old} | {new} | {delta} |")
        if self.failures:
            lines += ["", "Failures:", *(f"- {f}" for f in self.failures)]
        return "\n".join(lines)


def bits_per_value(entry: dict) -> float:
    return entry["bits"] / entry["values"] if entry["values"] else 0.0


def compare(expected: dict, actual: dict, tolerance: float = TOLERANCE) -> Report:
    old, new = expected["entries"], actual["entries"]
    changes, failures = [], []
    for name in sorted(old.keys() | new.keys()):
        o, n = old.get(name), new.get(name)
        if n is None:
            changes.append(Change(name, bits_per_value(o), None))
            failures.append(f"{name}: missing from the new results")
            continue
        if o is None:
            changes.append(Change(name, None, bits_per_value(n)))
            continue
        if o["values"] != n["values"]:
            failures.append(f"{name}: value count changed {o['values']} -> {n['values']} (dataset changed?)")
            continue
        if o["bits"] == n["bits"]:
            continue
        c = Change(name, bits_per_value(o), bits_per_value(n))
        changes.append(c)
        # Exact: bits and counts are integers, so no float rounding at the boundary.
        limit = Fraction(o["bits"], o["values"]) * (1 + Fraction(tolerance).limit_denominator(10**6))
        if Fraction(n["bits"], n["values"]) > limit:
            failures.append(f"{name}: {c.old_bpv:.3f} -> {c.new_bpv:.3f} bits/value ({c.relative:+.2%})")
    return Report(changes, failures)


def _load(path: str) -> dict:
    data = json.loads(open(path, encoding="utf-8").read())
    if data.get("format") != 1:
        raise ValueError(f"{path}: unknown format {data.get('format')!r}")
    return data


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        expected, actual = _load(argv[0]), _load(argv[1])
    except (OSError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    report = compare(expected, actual)
    md = report.markdown()
    print(md)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as f:
            f.write(md + "\n")
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
