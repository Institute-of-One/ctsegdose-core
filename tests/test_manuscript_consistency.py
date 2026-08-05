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
MANUSCRIPT = REPO / "paper" / "manuscript.md"
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
    flat = re.sub(r"\s+", " ", _load()[0])
    # Prose spells small numbers out; the results hold digits. Normalising here lets the
    # manuscript read naturally while the tests still compare against the computed value.
    words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
        "forty": 40,
    }
    for word, digit in words.items():
        flat = re.sub(rf"\b{word}\b", str(digit), flat, flags=re.IGNORECASE)
    return flat


@pytest.fixture(scope="module")
def raw():
    """The manuscript exactly as written.

    Needed wherever the assertion is about literal text rather than a number: the
    number-word normalisation above would turn "Institute of One" into "Institute of 1".
    """
    return _load()[0]


@pytest.fixture(scope="module")
def tables():
    return _load()[1]


def _abstract(raw: str) -> str:
    return raw.split("## Abstract", 1)[1].split("**Keywords", 1)[0]


def test_the_abstract_fits_the_journal_limit(raw):
    """Tomography (MDPI) asks for about 200 words. An abstract grows during revision, so
    the limit is pinned rather than checked once."""
    assert len(_abstract(raw).split()) <= 200


def test_the_abstract_is_one_paragraph_without_structured_headings(raw):
    """MDPI wants the background-to-conclusion arc woven into a single paragraph, not
    the Purpose:/Methods:/Results: labels a structured abstract uses."""
    body = _abstract(raw).strip()
    assert "\n\n" not in body, "the abstract must be a single paragraph"
    for label in ("Purpose:", "Background:", "Methods:", "Results:", "Conclusions:"):
        assert label not in body, f"remove the structured-abstract label {label!r}"


def test_the_mdpi_back_matter_sections_are_present_and_ordered(raw):
    required = [
        "## Author Contributions",
        "## Funding",
        "## Institutional Review Board Statement",
        "## Informed Consent Statement",
        "## Data Availability Statement",
        "## Conflicts of Interest",
        "## Use of Generative Artificial Intelligence",
        "## References",
    ]
    positions = []
    for heading in required:
        assert heading in raw, f"missing MDPI section: {heading}"
        positions.append(raw.index(heading))
    assert positions == sorted(positions), "MDPI back-matter sections are out of order"


def test_the_imrad_headings_are_the_mdpi_ones(raw):
    for heading in (
        "## 1. Introduction", "## 2. Materials and Methods", "## 3. Results",
        "## 4. Discussion", "## 5. Conclusions",
    ):
        assert heading in raw, f"missing MDPI heading: {heading}"


def test_the_conflict_of_interest_statement_discloses_the_commercial_interest(raw):
    coi = raw.split("## Conflicts of Interest", 1)[1].split("##", 1)[0]
    for term in ("LISIT", "TexelCraft", "commercial interest", "MIT licence"):
        assert term in coi, f"the conflicts statement must disclose {term!r}"


def test_the_ai_use_is_disclosed(raw):
    section = raw.split("## Use of Generative Artificial Intelligence", 1)[1]
    assert "Claude" in section and "Anthropic" in section
    assert "responsibility" in section


def test_every_reference_is_cited_and_every_citation_resolves(raw):
    """A reference list that has drifted from the text is the classic late-stage defect."""
    numbers = {int(n) for n in re.findall(r"^(\d+)\.\s", raw.split("## References")[1], re.M)}
    cited = set()
    for group in re.findall(r"\[(\d+(?:,\s*\d+)*)\]", raw.split("## References")[0]):
        cited.update(int(n) for n in group.split(","))
    assert cited <= numbers, f"cited but not listed: {sorted(cited - numbers)}"
    assert numbers <= cited, f"listed but never cited: {sorted(numbers - cited)}"


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
    assert f"{overall['recorded']} of 40 series" in text
    assert f"{overall['reconstructed']} were reconstructable" in text
    assert f"All {overall['unrecoverable']} unrecoverable series are GE" in text


def test_the_organ_mass_table_in_the_text_matches_the_computed_medians(text, tables):
    """The Results table is prose, so it is the easiest thing in the paper to edit by
    hand and the hardest to notice going stale."""
    overall = tables["organ_mass"]["overall"]
    for organ, label in (
        ("liver", "liver"), ("spleen", "spleen"),
        ("kidney_left", "kidney (left)"), ("kidney_right", "kidney (right)"),
        ("pancreas", "pancreas"),
    ):
        block = overall[organ]
        row = (
            f"| {label} | {block['mass_g']['n']} | {block['mass_g']['median']:.0f} g "
            f"| {block['median_over_reference']:.2f} |"
        )
        assert row in text, f"the {label} row does not match the computed values: {row}"


def test_the_organ_weight_range_quoted_in_the_text_is_the_computed_one(text, tables):
    weights = tables["weighted_ctdivol"]["by_organ"]
    lo = min(b["relative_weight"]["min"] for b in weights.values())
    hi = max(b["relative_weight"]["max"] for b in weights.values())
    assert f"{lo:.2f} to {hi:.2f}" in text, (
        f"the organ weights span {lo:.2f}-{hi:.2f}; the manuscript quotes something else"
    )


def test_the_spleen_review_claims_match_the_reviewed_cases(text):
    """The spleen attribution rests on four specific overlays; the overlays must exist
    and the text must quote the ratios they were reviewed at."""
    review = REPO / "paper" / "figures" / "review"
    if not review.exists():
        pytest.skip("review overlays not generated in this checkout")
    rendered = sorted(p.name for p in review.glob("spleen_*.png"))
    assert len(rendered) >= 4
    for name in rendered[:4]:
        ratio = name.split("_")[2].removesuffix("x")
        assert ratio in text, f"the text does not quote the reviewed ratio {ratio}x"


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
    assert f"{n} of the 40 series showed a peak-to-peak spread" in text


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


def test_the_affiliation_convention_holds(raw):
    assert "Institute of One, LISIT Co., Ltd., Tokyo, Japan" in raw
    assert "0000-0001-9211-1071" in raw
    assert "National Cancer Center" not in raw and "NCC" not in raw


def test_the_title_does_not_begin_with_open():
    """A standing convention for this series: the title leads with the finding."""
    raw = MANUSCRIPT.read_text(encoding="utf-8") if MANUSCRIPT.exists() else pytest.skip("no manuscript")
    title = next(line for line in raw.splitlines() if line.startswith("# "))
    assert not title[2:].strip().lower().startswith("open")


def test_every_figure_is_referenced_captioned_and_present(raw):
    """Strengthened after the first version passed vacuously: it matched a bold marker
    the text did not use, so an empty set trivially satisfied it."""
    figures = REPO / "paper" / "figures"
    if not figures.exists():
        pytest.skip("figures not generated in this checkout")

    discussed = {int(n) for n in re.findall(r"\bFigure (\d)\b", raw)}
    assert discussed == {1, 2, 3, 4}, f"the text discusses figures {sorted(discussed)}"

    embedded = re.findall(r"!\[\*\*Figure (\d)\.\*\*[^\]]+\]\(figures/([^)]+?)\)", raw)
    assert {int(n) for n, _ in embedded} == {1, 2, 3, 4}, (
        "every figure needs an embedded image with an MDPI-style caption, or it will "
        "not reach the reviewer in the built document"
    )
    for _, filename in embedded:
        assert (figures / filename.split("{")[0].strip()).exists(), f"missing {filename}"
