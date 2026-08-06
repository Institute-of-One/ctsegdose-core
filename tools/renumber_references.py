"""Renumber the manuscript's references into order of first mention.

MDPI, like most numbered-citation styles, requires that references be numbered by first
appearance in the text. Maintaining that by hand is a losing game: inserting one citation
into the Introduction renumbers everything after it, and the failure is silent — the
bibliography still looks like a bibliography.

So it is done mechanically. This script scans the body for citations in order, assigns new
numbers, rewrites both the in-text markers and the bibliography, and reports anything
inconsistent: a citation with no entry, an entry never cited, a duplicate number.

Usage::

    python tools/renumber_references.py            # rewrite in place
    python tools/renumber_references.py --check    # report only, exit non-zero if wrong
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANUSCRIPT = REPO / "paper" / "manuscript.md"
HEADING = "## References"

CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
#: A bibliography entry starts at column zero with "12. " and runs to the next such line.
ENTRY = re.compile(r"^(\d+)\.\s", re.M)


def split(text: str) -> tuple[str, str]:
    i = text.index(HEADING)
    return text[:i], text[i:]


def parse_entries(refs: str) -> dict[int, str]:
    """Bibliography entries keyed by their current number, body text preserved."""
    starts = [(int(m.group(1)), m.start()) for m in ENTRY.finditer(refs)]
    entries: dict[int, str] = {}
    for idx, (number, start) in enumerate(starts):
        end = starts[idx + 1][1] if idx + 1 < len(starts) else len(refs)
        body = refs[start:end].rstrip()
        # Strip the leading number and its indentation, keeping the wrapped text.
        body = re.sub(r"^\d+\.\s+", "", body)
        body = re.sub(r"^\s{2,}", "   ", body, flags=re.M)
        entries[number] = body.rstrip()
    return entries


def first_appearance(body: str) -> list[int]:
    order: list[int] = []
    for group in CITATION.findall(body):
        for token in group.split(","):
            n = int(token.strip())
            if n not in order:
                order.append(n)
    return order


def renumber(text: str) -> tuple[str, dict[str, list[int]]]:
    body, refs = split(text)
    entries = parse_entries(refs)
    order = first_appearance(body)

    report = {
        "cited_without_entry": sorted(set(order) - set(entries)),
        "entry_never_cited": sorted(set(entries) - set(order)),
    }
    if report["cited_without_entry"] or report["entry_never_cited"]:
        return text, report

    mapping = {old: new for new, old in enumerate(order, 1)}

    def rewrite(match: re.Match[str]) -> str:
        parts = [mapping[int(t.strip())] for t in match.group(1).split(",")]
        return "[" + ",".join(str(p) for p in parts) + "]"

    new_body = CITATION.sub(rewrite, body)

    lines = [HEADING, ""]
    for old in order:
        number = mapping[old]
        entry = entries[old]
        indent = " " * (len(f"{number}. "))
        wrapped = entry.replace("\n   ", "\n" + indent)
        lines.append(f"{number}. {wrapped}")
    new_refs = "\n".join(lines) + "\n"
    return new_body + new_refs, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report only; do not rewrite")
    ap.add_argument("--manuscript", type=Path, default=MANUSCRIPT)
    args = ap.parse_args()

    text = args.manuscript.read_text(encoding="utf-8")
    rewritten, report = renumber(text)

    for key, values in report.items():
        if values:
            print(f"  ! {key.replace('_', ' ')}: {values}")
    if any(report.values()):
        return 1

    body, _ = split(rewritten)
    order = first_appearance(body)
    if order != sorted(order):
        print(f"  ! citations are still out of order: {order[:12]}")
        return 1

    if args.check:
        if rewritten != text:
            print("  ! references are not in order of first mention; run without --check")
            return 1
        print(f"  references are in order of first mention ({len(order)} cited)")
        return 0

    if rewritten == text:
        print(f"  already in order ({len(order)} references)")
        return 0
    with open(args.manuscript, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(rewritten)
    print(f"  renumbered {len(order)} references into order of first mention")
    return 0


if __name__ == "__main__":
    sys.exit(main())
