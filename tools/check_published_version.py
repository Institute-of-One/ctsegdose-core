"""Check the published article against what was accepted, inside the 24-hour window.

Tomography's assistant editor asked the author to check the final published version and
to report anything wrong **within 24 hours of publication**. That is not long enough to
work out what to compare, so the comparison is written now, while the accepted text and
every result file are still to hand.

Usage, once the article is online::

    python tools/check_published_version.py --url https://www.mdpi.com/...        # fetch
    python tools/check_published_version.py --text published.txt                 # or paste

Give it the article's plain text, however obtained. It reports every discrepancy it can
find and exits non-zero if any is fatal; a clean run means the published text agrees with
the accepted manuscript on everything this repository can check.

What it does not check: typesetting, figure rendering, and anything requiring the eye.
Those are still the author's job, and Section "look at these by hand" below lists them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Exactly as Crossref carries it for the author's already-published paper. Two orderings
#: from one author read as two affiliations to a matching system.
AFFILIATION = "Institute of One, LISIT Co., Ltd., Tokyo 150-0044, Japan"

#: The release the Data Availability Statement must name. v0.1.1 did not contain two
#: result files the paper quotes, which is why v0.1.2 exists.
RELEASE = "v0.1.2"
COMMIT = "0ca9d57ef9fbc1a321cd5b671d21eb5857def518"
VERSION_DOI = "10.5281/zenodo.22143005"
SUPERSEDED = ("v0.1.1", "10.5281/zenodo.21817307", "46edee64")

ABSTRACT_SUBHEADINGS = ("Background/Objectives", "Methods", "Results", "Conclusions")

#: Defined at first use in the abstract, the main text and the first table, at the English
#: Editor's request.
DEFINITIONS = (
    "computed tomography (CT)",
    "(CTDIvol)",
    "International Commission on Radiological Protection (ICRP)",
)

#: Locations production added on the author's behalf after the second proof.
EQUIPMENT = ("GE", "Siemens", "Canon", "Philips")

BY_HAND = """
Look at these by hand -- no script can:

  * Figures 1, 3 and 5 must be the replaced versions. Figure 3(b): the dashed scan-CTDIvol
    line must not strike through the two weight labels nearest it. Figure 5: the five
    median labels must sit clear of their median bars. Figure 1: the count 4768 must have
    no comma, while 47,181 and 16,615 keep theirs.
  * Every figure caption attached to the right figure, numbered by first mention.
  * Display equations rendered as equations, not as source.
  * Table 1 intact, all five organ rows.
  * No highlighting or tracked-change marks left anywhere.
"""


def load(args) -> str:
    if args.text:
        return Path(args.text).read_text(encoding="utf-8", errors="replace")
    request = urllib.request.Request(
        args.url, headers={"User-Agent": "IORN-006 published-version check (yamamoto@lisit.jp)"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        html = response.read().decode("utf-8", "replace")
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S)
    return re.sub(r"<[^>]+>", " ", text)


def numbers_the_manuscript_quotes() -> list[str]:
    """Every figure the accepted manuscript states, so the published text can be checked
    against them rather than against a list typed by hand."""
    path = REPO / "paper" / "manuscript.md"
    body = path.read_text(encoding="utf-8")
    body = body.split("## 1. Introduction", 1)[1].split("## References", 1)[0]
    flat = " ".join(body.split())
    found = set()
    for pattern in (
        # The weight range, anchored so it cannot also match a section reference:
        # "Sections 2.1 to 2.12" in the AI-use statement was being reported as a missing
        # result, which is the kind of false alarm that gets a checklist ignored.
        r"\b0\.\d+ to \d\.\d+\b",
        r"\b\d{2,3}%\b",                       # percentages
        r"\b\d+\.\d+ mGy\b",                   # doses
        r"\b\d{3} (?:requested )?organ",       # record counts
    ):
        found.update(re.findall(pattern, flat))
    return sorted(found)


def check(text: str) -> tuple[list[str], list[str]]:
    flat = " ".join(text.split())
    # Whitespace removed entirely. A DOI, a commit hash or a version tag can be split
    # across elements by the typesetter -- in the Word file "v0.1.2" is stored as "v0.1."
    # and "2" in two runs, and an HTML page can do the same -- so an identifier absent
    # from `flat` may still be present on the page. Checking the squashed form as well
    # stops the check reporting a problem that is only in the extraction.
    squashed = re.sub(r"\s+", "", text)

    def present(value: str) -> bool:
        return value in flat or re.sub(r"\s+", "", value) in squashed

    fatal: list[str] = []
    warn: list[str] = []

    if not present(AFFILIATION):
        fatal.append(
            f"affiliation is not the canonical string.\n      expected: {AFFILIATION}"
        )
    for name in ("Shuji Yamamoto", "0000-0001-9211-1071", "yamamoto@lisit.jp"):
        if not present(name):
            fatal.append(f"missing from the published text: {name}")

    for value, label in (
        (RELEASE, "release tag"),
        (COMMIT, "commit hash"),
        (VERSION_DOI, "Zenodo version DOI"),
    ):
        if not present(value):
            fatal.append(f"Data Availability: {label} {value} is absent")
    for stale in SUPERSEDED:
        if present(stale):
            fatal.append(
                f"Data Availability still carries the superseded {stale}; that release "
                "does not contain two result files the paper quotes"
            )

    for heading in ABSTRACT_SUBHEADINGS:
        if not present(heading):
            fatal.append(f"abstract subheading missing: {heading}")
    for definition in DEFINITIONS:
        if not present(definition):
            fatal.append(f"abbreviation not defined at first use: {definition}")

    for vendor in EQUIPMENT:
        if vendor not in flat:
            warn.append(f"manufacturer {vendor} not found; production added these")

    for quoted in numbers_the_manuscript_quotes():
        if not present(quoted):
            warn.append(f"a value the accepted manuscript states is absent: {quoted!r}")

    if "Institute of One" in flat and "LISIT,Inc" in flat:
        warn.append("both 'LISIT Co., Ltd.' and 'LISIT,Inc.' appear")

    return fatal, warn


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="the published article's page")
    source.add_argument("--text", help="a file holding the article's plain text")
    args = ap.parse_args(argv)

    fatal, warn = check(load(args))

    if fatal:
        print("MUST BE CORRECTED (contact the editor within 24 hours):\n")
        for item in fatal:
            print(f"  - {item}")
        print()
    if warn:
        print("CHECK BY EYE -- may be a false alarm from the text extraction:\n")
        for item in warn:
            print(f"  - {item}")
        print()
    if not fatal and not warn:
        print("everything this repository can check agrees with the accepted manuscript\n")

    print(BY_HAND)
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
