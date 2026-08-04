"""Every number the manuscript states must be the number the results hold.

A manuscript is the one artefact nothing else checks: a figure regenerates, a table
regenerates, but prose is typed once and then edited by hand for a year. This suite
reads the quoted values back out of the text and compares them with
``results/analysis_1.5mm.json``, so a sentence cannot outlive the data behind it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MANUSCRIPT = REPO / "paper" / "manuscript_skeleton.md"
TABLES = REPO / "results" / "analysis_1.5mm.json"


def _load():
    if not (MANUSCRIPT.exists() and TABLES.exists()):
        pytest.skip("manuscript or analysis tables not generated in this checkout")
    return MANUSCRIPT.read_text(encoding="utf-8"), json.loads(TABLES.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def text():
    """The manuscript with its line wrapping flattened.

    Prose wraps, and a quoted figure can land either side of a line break. Matching
    against the raw file would make these tests fail on reflowing rather than on a
    number changing, which would train the author to ignore them.
    """
    return re.sub(r"\s+", " ", _load()[0])


@pytest.fixture(scope="module")
def tables():
    return _load()[1]


def test_the_cohort_size_in_the_text_matches_the_results(text, tables):
    cohort = tables["cohort"]
    assert f"{cohort['n_series']} abdominal CT series" in text or f"{cohort['n_series']} series" in text
    assert f"{cohort['n_organ_records']} organ records" in text


def test_the_cohort_breadth_quoted_in_the_text_is_the_computed_one(text, tables):
    """Added after the draft claimed 44 collections and 30 models against an actual 21
    and 23 — a number written from memory rather than from the results."""
    cohort = tables["cohort"]
    assert f"{len(cohort['collections'])} collections" in text
    assert f"{len(cohort['scanner_models'])} scanner\nmodels".replace("\n", " ") in text


def test_the_organ_mass_ratios_quoted_in_the_text_are_the_computed_ones(text, tables):
    overall = tables["organ_mass"]["overall"]
    for organ, label in (
        ("liver", "liver"), ("spleen", "spleen"), ("pancreas", "pancreas"),
    ):
        ratio = overall[organ]["median_over_reference"]
        assert f"{ratio:.2f}" in text, (
            f"the {label} median-over-reference is {ratio:.2f}; the manuscript does not "
            "quote that number"
        )


def test_the_availability_counts_quoted_in_the_text_are_the_computed_ones(text, tables):
    overall = tables["availability"]["overall"]
    assert f"{overall['recorded']} of 40 series" in text or f"Recorded {overall['recorded']}/40" in text
    assert f"{overall['unrecoverable']} — all GE" in text or f"{overall['unrecoverable']}/40, all GE" in text


def test_the_headline_vendor_claim_is_still_true_of_the_data(text, tables):
    ge = tables["availability"]["by_vendor"]["GE"]
    assert ge["recorded"] == 0
    assert "GE 0/10" in text, "the headline is quoted as GE 0/10; keep text and data together"


def test_the_truncation_range_quoted_in_the_text_matches(text, tables):
    rates = {
        v: b["truncated_rate"] * 100
        for v, b in tables["study_limits"]["truncation"]["by_vendor"].items()
    }
    assert f"{min(rates.values()):.1f}" in text
    assert f"{max(rates.values()):.1f}" in text


def test_the_flat_weighting_count_matches(text, tables):
    n = tables["study_limits"]["flat_weighting"]["n_series"]
    words = {1: "One", 2: "Two", 3: "Three", 4: "Four"}
    assert f"{words.get(n, n)} series show no usable variation" in text


def test_the_two_offsets_are_attributed_and_the_attribution_is_cited(text):
    """Neither offset may sit in the Results unexplained: each has an attribution, and
    the attribution names its source."""
    assert "Wasserthal" in text and "10.1148/ryai.230024" in text
    assert "0.887" in text and "0.983" in text, (
        "the pancreas and spleen attributions rest on TotalSegmentator's own per-class "
        "Dice; quote them so a reader can check the reasoning"
    )
    assert "Dice is symmetric" in text.replace("*", ""), (
        "Dice supports lower boundary agreement but not the sign of a bias; the text "
        "must say so rather than over-claim the pancreas attribution"
    )


def test_no_absorbed_dose_in_milligray_is_claimed_anywhere(text):
    """IORN-006 stops at the index. A claimed mGy organ dose would mean a coefficient
    table was used, which nothing in this repository is licensed to ship."""
    for pattern in (r"organ dose of [\d.]+\s*mGy", r"absorbed (organ )?dose (was|of) [\d.]+"):
        assert not re.search(pattern, text, re.IGNORECASE), (
            f"the manuscript appears to quote an absorbed organ dose: {pattern}"
        )


def test_the_affiliation_convention_holds(text):
    assert "Institute of One, LISIT Co., Ltd., Tokyo, Japan" in text
    assert "0000-0001-9211-1071" in text
    assert "National Cancer Center" not in text and "NCC" not in text


def test_the_title_does_not_begin_with_open():
    """A standing convention for this series: the title leads with the finding."""
    raw = MANUSCRIPT.read_text(encoding="utf-8") if MANUSCRIPT.exists() else pytest.skip("no manuscript")
    title = next(line for line in raw.splitlines() if line.startswith("# "))
    assert not title[2:].strip().lower().startswith("open")


def test_every_figure_the_text_references_exists(text):
    referenced = set(re.findall(r"\*\*Figure (\d)\*\*", text))
    figures = REPO / "paper" / "figures"
    if not figures.exists():
        pytest.skip("figures not generated in this checkout")
    present = {p.name[3] for p in figures.glob("fig*.png")}
    assert referenced <= present, f"referenced but missing: {referenced - present}"
