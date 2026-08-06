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


@pytest.fixture(scope="module")
def payload_or_skip():
    path = REPO / "results" / "organ_dose_1.5mm.json"
    if not path.exists():
        pytest.skip("per-series records not generated in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


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


def test_the_reference_list_is_the_size_an_original_article_needs(raw):
    numbers = {int(n) for n in re.findall(r"^(\d+)\.\s", raw.split("## References")[1], re.M)}
    assert 18 <= len(numbers) <= 24, f"{len(numbers)} references; the target is 18-22"


def test_references_are_numbered_in_order_of_first_mention(raw):
    """MDPI requires it, and maintaining it by hand fails silently: inserting one citation
    early renumbers everything after it while the bibliography still looks correct."""
    body = raw.split("## References")[0]
    order: list[int] = []
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", body):
        for token in group.split(","):
            n = int(token.strip())
            if n not in order:
                order.append(n)
    assert order == sorted(order), (
        f"references are not in order of first mention: {order[:12]}...  "
        "run tools/renumber_references.py"
    )
    assert order == list(range(1, len(order) + 1)), (
        f"citation numbering has gaps or duplicates: {order[:12]}"
    )


def test_a_software_reference_naming_a_version_cites_that_version_doi(raw):
    """A concept DOI resolves to whichever release is latest, so pairing it with an
    explicit version number cites two different things at once. The companion software is
    cited at Version 0.1.1, so it must carry that version's DOI, not the concept DOI."""
    refs = raw.split("## References")[1]
    entry = next(
        (line for line in refs.split("\n18. ")[1].split("\n19. ")[0].splitlines()),
        "",
    ) + refs.split("\n18. ")[1].split("\n19. ")[0]
    assert "Version 0.1.1" in entry
    assert "zenodo.21636719" in entry, "must be the v0.1.1 version DOI"
    assert "zenodo.21636082" not in entry, (
        "that is the concept DOI, which resolves to the latest release rather than to "
        "the version the entry names"
    )


def test_the_three_closest_studies_are_cited_with_verified_dois(raw):
    """Added at the author's direction; each DOI was checked against Crossref."""
    for doi, who in (
        ("10.1002/acm2.70321", "Nuntue et al. 2025, DICOM-header TCM profiles"),
        ("10.3390/tomography10120151", "Eom et al. 2024, TotalSegmentator dose calculation"),
        ("10.1002/mp.15402", "Li et al. 2022, SSDE(z)"),
    ):
        assert doi in raw, f"missing citation: {who}"
    for author in ("Nuntue", "Eom", "Li, X."):
        assert author in raw, f"{author} must appear in the bibliography"


def test_the_criterion_is_not_described_as_prespecified(raw):
    """It was formalised after the anomalous series was found, so calling it prespecified
    would misrepresent the sequence."""
    for word in ("prespecified", "pre-specified", "stated in advance", "defined a priori"):
        assert word not in raw.lower(), f"the criterion is described as {word!r}"
    assert "rule-based acquisition-constancy criterion" in raw


def test_priority_claims_are_not_overstated(raw):
    body = raw.split("## References")[0]
    for overclaim in (
        "The first multi-vendor", "first multi-vendor", "never reported",
        "not previously reported", "no previous study", "unprecedented",
        "The existing work does not answer",
    ):
        assert overclaim not in body, f"unsupported priority claim: {overclaim!r}"
    assert "only partially answers" in body


def test_the_prior_art_for_the_quantity_is_cited(raw):
    """The organ-specific weighted CTDIvol is a reported quantity, and the papers that
    reported it must be visible in the Introduction, not only in the bibliography."""
    intro = raw.split("## 2. Materials and Methods")[0]
    for author in ("Khatonabadi", "Tian", "Angel", "McCollough"):
        assert author in intro, f"the Introduction must situate the work against {author}"
    assert "10.1118/1.4907955" in raw, "Tian et al. 2015 must be cited"
    assert "10.1118/1.4798561" in raw, "Khatonabadi et al. 2013 must be cited"
    assert "10.1007/s10278-013-9622-7" in raw, "the TCIA archive paper must be cited"


def test_the_work_is_positioned_as_operationalisation_not_invention(raw):
    """The claim is an open multi-vendor operationalisation of a reported quantity."""
    assert "not new" in raw and "no new index is proposed" in raw
    assert "operationalisation and empirical" in raw
    assert "Relation to previous work" in raw
    for overclaim in (
        "This study therefore constructs an **anatomy-weighted CTDIvol index**",
        "we propose a new index",
        "a novel index",
    ):
        assert overclaim not in raw, f"invention framing remains: {overclaim!r}"


def test_the_literature_gap_is_stated_in_both_introduction_and_discussion(raw):
    intro = raw.split("## 2. Materials and Methods")[0]
    discussion = raw.split("## 4. Discussion")[1].split("## 5. Conclusions")[0]
    for section, name in ((intro, "Introduction"), (discussion, "Discussion")):
        assert "projection data" in section, (
            f"the {name} must say why archived data does not support the earlier designs"
        )
        assert "single-institution" in section, (
            f"the {name} must name the cohort limitation of the prior work"
        )


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
    assert f"{overall['recorded']} of 40 series retained" in text
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


def test_the_vendor_availability_claim_is_true_of_the_data(tables):
    assert tables["availability"]["by_vendor"]["GE"]["recorded"] == 0
    table = tables["availability"]["ge_vs_rest_recorded"]["table"]
    assert table["ge_recorded"] == 0
    assert table["ge_recorded"] + table["ge_not_recorded"] == 10


def test_no_significance_test_is_reported_for_the_availability_comparison(raw, tables):
    """Removed deliberately: series in a curated archive are not independent with respect
    to collection, site, scanner model or export pathway, so a p-value over that structure
    would describe a sampling model the data do not satisfy. The text may *explain* that
    absence — what it must not do is report a test result."""
    assert "p_value" not in tables["availability"]["ge_vs_rest_recorded"]
    assert "inference" in tables["availability"]["ge_vs_rest_recorded"]
    assert "No significance test is applied" in raw
    for banned in ("Fisher", "p = ", "p < 0.0", "χ2", "chi-square"):
        assert banned not in raw, f"a significance test is still reported: {banned!r}"


def test_the_availability_claim_is_scoped_to_this_archive_sample(raw):
    """The finding concerns archived DICOM headers in one curated archive, not what a
    manufacturer's scanners inherently do."""
    assert "retained a recorded CTDIvol in the archived" in raw
    assert "none of the ten sampled GE series" in raw
    for overreach in ("GE recorded a CTDIvol in none", "GE records no CTDIvol", "GE 0/10"):
        assert overreach not in raw, f"unscoped vendor claim: {overreach!r}"


def test_the_confounders_of_the_availability_finding_are_named(text):
    for confounder in (
        "scanner implementation", "acquisition site", "export pathway",
        "de-identification", "archive curation", "collection composition",
    ):
        assert confounder in text, f"the Discussion must name {confounder!r} as a confounder"


def test_the_record_flow_in_the_text_reconciles_455_with_480(text, tables):
    flow = tables["record_flow"]
    assert f"{flow['expected_organ_series_combinations']} requested organ–series combinations" in text
    assert f"Records were produced for {flow['organ_records_produced']}" in text
    assert f"The remaining {flow['absent_combinations']['total']}" in text
    assert f"{flow['whole_organ_records']} records are untruncated" in text
    assert f"{flow['records_with_an_organ_weighted_ctdivol']} records" in text
    assert f"the remaining {flow['records_with_a_modulation_weight_only']} records" in text
    assert f"{flow['records_in_the_reference_mass_comparison']} untruncated records" in text
    by_organ = flow["absent_combinations"]["by_organ"]
    assert f"urinary bladder in {by_organ['urinary_bladder']} series" in text
    assert f"gallbladder in {by_organ['gallbladder']}" in text


def test_the_two_record_conditions_are_stated_as_independent_not_nested(text, tables):
    """386 is not a subset of 408: truncation is a property of the organ, index
    availability a property of the series. Reporting only the totals invites the reader to
    nest them."""
    flow = tables["record_flow"]
    assert "independent axes rather" in text
    assert f"hold together for {flow['records_untruncated_and_with_index']} records" in text
    truncated_with_index = (
        flow["records_with_an_organ_weighted_ctdivol"] - flow["records_untruncated_and_with_index"]
    )
    untruncated_without = (
        flow["whole_organ_records"] - flow["records_untruncated_and_with_index"]
    )
    assert f"{truncated_with_index} truncated records still carry an index" in text
    assert f"{untruncated_without} untruncated records do not" in text


def test_the_weighting_assumption_is_stated_and_its_verification_reported(text, tables):
    """The index treats tube current as a proxy for scanner output; that holds only if the
    other output-governing parameters are fixed within a series."""
    assert "The weighting assumes that, within each series" in text
    assert "acquisition_constancy.json" in text
    constancy = REPO / "results" / "acquisition_constancy.json"
    if not constancy.exists():
        pytest.skip("acquisition constancy check has not been run in this checkout")
    summary = json.loads(constancy.read_text(encoding="utf-8"))["summary"]
    kvp = summary["tube_voltage_kvp"]
    assert kvp["constant"] == 40 and kvp["varies"] == 0
    assert f"verified constant in all {kvp['constant']} series" in text
    # An attribute absent from the headers must not be described as verified.
    for name, label in (
        ("exposure_time_ms", "Exposure time"),
        ("spiral_pitch_factor", "pitch"),
        ("rotation_time_s", "rotation time"),
    ):
        block = summary[name]
        if block["absent"]:
            assert f"absent in {block['absent']}" in text or \
                   f"absent from the\nheaders of {block['absent']}".replace("\n", " ") in text, (
                       f"{label} is absent in {block['absent']} series; say so rather than "
                       "implying it was verified"
                   )


def test_the_ineligible_series_is_reported_and_excluded(text, tables):
    """The criterion must be stated, its effect reported, and the excluded series kept in
    the analyses it remains valid for."""
    e = tables.get("modulation_eligibility")
    if not e:
        pytest.skip("eligibility has not been computed in this checkout")
    assert e["n_ineligible"] == 1
    assert f"{e['n_eligible']} met the acquisition-constancy criterion" in text
    assert "retained for the segmentation, estimated-mass and archive-availability" in text
    assert "not proportional to scanner output" in text
    # The withdrawn justification must not creep back in.
    assert "no reported result depends on it" not in text


def test_the_modulation_analysis_excludes_the_ineligible_series(payload_or_skip, tables):
    """The exclusion must be real in the numbers, not only described in the prose."""
    e = tables["modulation_eligibility"]
    ineligible = {r["series_instance_uid"] for r in e["ineligible_series"]}
    flow = tables["record_flow"]
    assert flow["modulation_eligible_series"] == e["n_eligible"]
    counted = 0
    for s in payload_or_skip["series"]:
        if s["series_instance_uid"] in ineligible:
            continue
        counted += sum(
            1 for o in s.get("organs", []) if o.get("organ_weighted_ctdivol_mgy") is not None
        )
    assert flow["records_in_the_modulation_analysis"] == counted
    assert counted < flow["records_with_an_organ_weighted_ctdivol"], (
        "the exclusion must actually remove records from the modulation analysis"
    )


def test_the_rounding_tolerance_separates_the_two_kinds_of_variation(tables):
    """Three Philips series vary by under 1% and must survive; one GE series varies by a
    factor of two and must not."""
    from ctsegdose_core.eligibility import ROUNDING_TOLERANCE

    e = tables["modulation_eligibility"]
    assert e["rounding_tolerance"] == ROUNDING_TOLERANCE
    counts = e["classification_counts"]["exposure_time_ms"]
    assert counts.get("negligible_variation", 0) >= 1
    assert counts.get("materially_variable", 0) == 1
    assert counts.get("absent", 0) >= 1, "absence must be its own class, not a failure"


def test_no_unqualified_claim_that_every_input_is_openly_licensed(raw):
    """The HU-density anchor values are used by citation, not redistributed, so a blanket
    claim would be wrong."""
    for overclaim in (
        "openly licensed inputs throughout",
        "all inputs are openly licensed",
        "fully open measurement chain",
        "entirely open inputs",
    ):
        assert overclaim not in raw, f"over-broad openness claim: {overclaim!r}"


def test_no_placeholder_or_todo_survives_into_the_prose(raw):
    """Release fields are `{{NAME}}` placeholders that the submission build resolves or
    refuses on; nothing else of the kind may be in the source."""
    assert "TODO-AUTHOR" not in raw
    assert "TODO" not in raw
    for name in re.findall(r"\{\{([A-Z_]+)\}\}", raw):
        assert name in {"RELEASE_TAG", "COMMIT_HASH", "ZENODO_VERSION_DOI"}, (
            f"unexpected placeholder {name!r}"
        )


def test_the_index_is_never_presented_as_an_absorbed_organ_dose(text):
    """The central scoping requirement of this paper."""
    assert "not an estimate of absorbed organ dose" in text
    for omitted in (
        "scattered radiation", "angular", "organ depth",
        "Monte-Carlo radiation transport", "absorbed organ dose in milligray",
    ):
        assert omitted in text, f"Section 2.9 must state that {omitted!r} is not accounted for"
    for overreach in (
        "the dose received by", "exposure received by an organ",
        "we characterised ct dose at the organ level", "organ-level dosimetry",
    ):
        assert overreach not in text.lower(), f"dose overreach: {overreach!r}"


def test_organ_mass_is_presented_as_a_model_based_estimate(text):
    assert "model-based estimates rather than physical ground truth" in text
    for factor in ("Contrast enhancement", "reconstruction kernel", "scanner-specific HU"):
        assert factor in text, f"the mass limitation must name {factor!r}"


def test_the_reference_comparison_is_not_called_a_calibration(text):
    """ICRP 89 is an external anchor, not a ground truth and not a calibration target."""
    assert "external reference comparison" in text.lower()
    assert "calibration of the segmented organ masses" not in text
    assert "cannot be determined without subject-level reference contours" in text


def test_the_two_offsets_are_explored_rather_than_asserted(text):
    assert "most consistent with cohort anatomy among the explanations examined" in text
    for overclaim in (
        "The +72% median is cohort anatomy",
        "The pancreas offset is attributed to the segmentation",
        "The spleen offset is attributed to the cohort",
    ):
        assert overclaim not in text, f"over-strong attribution remains: {overclaim!r}"


def test_the_truncation_range_quoted_in_the_text_matches(text, tables):
    rates = {
        v: b["truncated_rate"] * 100
        for v, b in tables["study_limits"]["truncation"]["by_vendor"].items()
    }
    assert f"{min(rates.values()):.1f}" in text
    assert f"{max(rates.values()):.1f}" in text


def test_the_flat_weighting_count_matches(text, tables):
    """Reported against the eligible denominator, since the weight spread is a modulation
    statement and the ineligible series contributes none."""
    n = tables["study_limits"]["flat_weighting"]["n_series"]
    eligible = tables["modulation_eligibility"]["n_eligible"]
    assert (
        f"{n} of the {eligible} series eligible for quantitative modulation analysis "
        "showed a peak-to-peak spread"
    ) in text


def test_the_two_offsets_are_examined_with_their_evidence_quoted(text):
    """Neither offset may sit in the Results unexplained; equally, neither explanation may
    be stated more strongly than the evidence supports."""
    assert "Wasserthal" in text and "10.1148/ryai.230024" in text
    assert "0.887" in text and "0.983" in text, (
        "the pancreas and spleen discussions rest on TotalSegmentator's own per-class "
        "Dice; quote them so a reader can check the reasoning"
    )
    assert "Dice is\nsymmetric".replace("\n", " ") in text.replace("*", ""), (
        "Dice supports lower boundary agreement but not the direction of a disagreement; "
        "the text must say so rather than over-claim the pancreas explanation"
    )
    assert "alternative contributors that the present design\ncannot separate".replace("\n", " ") in text


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

    discussed = [int(n) for n in re.findall(r"\bFigure (\d)\b", raw)]
    assert set(discussed) == {1, 2, 3, 4, 5}, f"the text discusses figures {sorted(set(discussed))}"

    # MDPI, like every journal, numbers figures by first mention.
    first_mention: list[int] = []
    for n in discussed:
        if n not in first_mention:
            first_mention.append(n)
    assert first_mention == [1, 2, 3, 4, 5], (
        f"figures are first mentioned in the order {first_mention}; they must be numbered "
        "in order of first mention"
    )

    # Images carry no alt text and the caption is a following paragraph. Using pandoc's
    # implicit figures instead would make the renderer add its own "Figure N:" on top of
    # the caption's own label, which is where the doubled numbering came from.
    embedded = re.findall(r"!\[\]\(figures/([^){]+)\)", raw)
    assert len(embedded) == 5, f"expected five embedded figures, found {len(embedded)}"
    for i, filename in enumerate(embedded, 1):
        assert (figures / filename.strip()).exists(), f"missing {filename}"
        assert filename.startswith(f"fig{i}"), (
            f"figure {i} embeds {filename}; the file stem must match its number"
        )

    captions = {int(n) for n in re.findall(r"^\*\*Figure (\d)\.\*\* ", raw, re.M)}
    assert captions == {1, 2, 3, 4, 5}, f"figures captioned: {sorted(captions)}"


def test_no_figure_caption_carries_a_doubled_number(raw):
    """The rendered output read 'Figure 1: Figure 1.' while the caption sat inside the
    image's alt text and the renderer numbered it as well."""
    assert not re.search(r"!\[\*\*Figure", raw), (
        "a caption inside the image alt text makes the renderer number it twice"
    )
