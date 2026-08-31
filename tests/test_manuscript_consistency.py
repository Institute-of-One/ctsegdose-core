"""Every number the manuscript states must be the number the results hold.

A manuscript is the one artefact nothing else checks: a figure regenerates, a table
regenerates, but prose is typed once and then edited by hand for a year. This suite
reads the quoted values back out of the text and compares them with
``results/analysis_1.5mm.json``, so a sentence cannot outlive the data behind it.
"""

from __future__ import annotations

import collections
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


#: The subheadings Tomography's production editor asked for at proof, in the order
#: they asked for them. They are the journal's, not ours: the first version of this
#: file asserted the opposite, that the abstract must carry none of them.
ABSTRACT_SUBHEADINGS = (
    "Background/Objectives:",
    "Methods:",
    "Results:",
    "Conclusions:",
)


def test_the_abstract_fits_the_journal_limit(raw):
    """Tomography (MDPI) asks for about 200 words. An abstract grows during revision, so
    the limit is pinned rather than checked once.

    The four mandated subheadings are not counted: they are labels the journal requires,
    not abstract prose, and counting them would make the limit depend on the journal's
    own formatting.
    """
    body = _abstract(raw)
    for label in ABSTRACT_SUBHEADINGS:
        body = body.replace(f"**{label}**", "").replace(label, "")
    # 215, which is the length of the abstract actually sent at second proof, not a
    # target. The journal's English Editor required abbreviations to be defined at first
    # use, which cost thirteen words on an abstract that was already 202 in the proof.
    # The editors were asked whether they want it inside 200 and offered a thirteen-word
    # cut that touches no definition and no number; if they say yes, this comes down
    # again. Until then the limit records what was sent rather than what was wished for.
    assert len(body.split()) <= 215


def test_the_abstract_carries_the_subheadings_the_journal_requires(raw):
    """This test asserted the reverse until the proof arrived.

    It read: "MDPI wants the background-to-conclusion arc woven into a single paragraph,
    not the Purpose:/Methods:/Results: labels a structured abstract uses." The production
    editor asked for exactly those labels. The journal is the authority on its own
    template, so the assertion is inverted rather than argued with.
    """
    body = _abstract(raw).strip()
    assert "\n\n" not in body, "the abstract must still be a single paragraph"
    positions = []
    for label in ABSTRACT_SUBHEADINGS:
        assert label in body, f"the abstract is missing the subheading {label!r}"
        positions.append(body.index(label))
    assert positions == sorted(positions), (
        f"the subheadings are out of order: they must run {ABSTRACT_SUBHEADINGS}"
    )
    assert "Purpose:" not in body, "the journal's label is Background/Objectives"


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
    """The floor was 18 until the Tomography Academic Editor asked, at pre-check, that the
    several citations to Prof. Samei and Prof. McNitt-Gray be reduced to one representative
    paper each. That removed seven entries the list could not replace without citing work
    the argument does not rest on, so the floor follows the editor rather than the reverse.
    """
    numbers = {int(n) for n in re.findall(r"^(\d+)\.\s", raw.split("## References")[1], re.M)}
    assert 14 <= len(numbers) <= 24, f"{len(numbers)} references; the target is 14-22"


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
    cited at Version 0.1.1, so it must carry that version's DOI, not the concept DOI.

    The entry is located by its content rather than by its number: the number moves
    whenever a reference is added or removed, and pinning it made this test fail with an
    IndexError that said nothing about the DOI it exists to guard.
    """
    refs = raw.split("## References")[1]
    entries = re.split(r"^\d+\.\s", refs, flags=re.M)
    entry = next((e for e in entries if "ctdose-core" in e), "")
    assert entry, "the companion software reference is missing from the bibliography"
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
    reported it must be visible in the Introduction, not only in the bibliography.

    Angel et al. was dropped from this list when the Academic Editor asked for one
    representative paper per author: it supported the *magnitude* of the modulation effect,
    which Khatonabadi and Tian also carry, whereas those two are the attribution for the
    quantity itself and are therefore the ones that may never leave the Introduction.
    """
    intro = raw.split("## 2. Materials and Methods")[0]
    for author in ("Khatonabadi", "Tian", "McCollough"):
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
    # The postal code sits between the company and the city, as MDPI prints it, so the
    # string is matched in two halves rather than as one literal.
    assert "Institute of One, LISIT Co., Ltd." in raw
    assert "Tokyo, Japan" in raw
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

    # Derived from the manuscript rather than written as a literal. A hardcoded
    # {1..5} fails the moment a figure is added, which says nothing about whether the
    # numbering is right -- and adding a figure is exactly when this test should be
    # doing its job rather than demanding to be edited.
    captioned = sorted({int(n) for n in re.findall(r"\*\*Figure (\d+)\.\*\*", raw)})
    assert captioned, "the manuscript captions no figures"
    assert captioned == list(range(1, len(captioned) + 1)), (
        f"figure captions are numbered {captioned}; they must run 1..n"
    )

    discussed = [int(n) for n in re.findall(r"\bFigure (\d+)\b", raw)]
    assert set(discussed) == set(captioned), (
        f"the text discusses figures {sorted(set(discussed))} and captions {captioned}"
    )

    # MDPI, like every journal, numbers figures by first mention.
    first_mention: list[int] = []
    for n in discussed:
        if n not in first_mention:
            first_mention.append(n)
    assert first_mention == captioned, (
        f"figures are first mentioned in the order {first_mention}; they must be numbered "
        "in order of first mention"
    )

    # Images carry no alt text and the caption is a following paragraph. Using pandoc's
    # implicit figures instead would make the renderer add its own "Figure N:" on top of
    # the caption's own label, which is where the doubled numbering came from.
    embedded = re.findall(r"!\[\]\(figures/([^){]+)\)", raw)
    assert len(embedded) == len(captioned), (
        f"{len(embedded)} figures are embedded and {len(captioned)} are captioned"
    )
    for i, filename in enumerate(embedded, 1):
        assert (figures / filename.strip()).exists(), f"missing {filename}"
        assert filename.startswith(f"fig{i}"), (
            f"figure {i} embeds {filename}; the file stem must match its number"
        )

    # The stem check above is why the files were renamed when the pipeline flowchart
    # became Figure 1: without it, fig1_segmentation could sit under "Figure 2" and
    # nothing would notice until someone opened the wrong image.
    labelled = {int(n) for n in re.findall(r"^\*\*Figure (\d+)\.\*\* ", raw, re.M)}
    assert labelled == set(captioned), f"figures captioned: {sorted(labelled)}"


def test_no_figure_caption_carries_a_doubled_number(raw):
    """The rendered output read 'Figure 1: Figure 1.' while the caption sat inside the
    image's alt text and the renderer numbered it as well."""
    assert not re.search(r"!\[\*\*Figure", raw), (
        "a caption inside the image alt text makes the renderer number it twice"
    )


def _segmentation_checks():
    path = REPO / "results" / "segmentation_checks_1.5mm.json"
    if not path.is_file():
        pytest.skip("segmentation checks not present in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_segmentation_quality_control_counts_are_the_computed_ones(text):
    """Added for the round-1 revision. Both reviewers asked whether the segmentation
    was checked; the answer is a number, and a number in prose goes stale."""
    checks = _segmentation_checks()
    flags = [
        check
        for report in checks["reports"]
        for check in report["checks"]
        if not check.get("passed", True)
    ]
    assert f"{checks['n_series_with_failures']} of the {checks['n_series_checked']}" in text
    assert f"{len(flags)} flags in all" in text

    by_name = collections.Counter(check.get("name", "?") for check in flags)
    mass = sum(n for name, n in by_name.items() if "mass plausible" in name)
    bracket = sum(n for name, n in by_name.items() if "bracket the scan mean" in name)
    assert f"{mass} were organ masses outside" in text
    assert f"{bracket} were the weights\nfailing".replace("\n", " ") in text.replace(
        "\n", " "
    )


def test_no_laterality_or_inversion_failure_is_claimed_that_the_checks_contradict(text):
    """The manuscript says none was a laterality failure. If one ever appears, the
    sentence becomes false and this is what says so."""
    checks = _segmentation_checks()
    lateral = [
        check
        for report in checks["reports"]
        for check in report["checks"]
        if not check.get("passed", True)
        and ("left" in check.get("name", "") and "right" in check.get("name", ""))
    ]
    if "None was a laterality\nfailure".replace("\n", " ") in text.replace("\n", " "):
        assert not lateral, f"the text claims no laterality failure; {len(lateral)} exist"


def _reconstruction_validation():
    path = REPO / "results" / "reconstruction_validation.json"
    if not path.is_file():
        pytest.skip("run tools/validate_reconstruction.py")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_reconstruction_agreement_quoted_in_the_text_is_the_measured_one(text):
    """Section 3.4 was written for the round-1 revision and every number in it is a
    measurement, so every number in it can go stale."""
    v = _reconstruction_validation()
    assert f"{v['n_recorded_series']} series with a recorded CTDIvol" in text
    assert f"{v['n_compared']} could be reconstructed" in text
    median = v["agreement"]["median_absolute_relative_error"]
    assert f"{median:.1%}" in text, f"the median absolute difference is {median:.1%}"

    for model, stats in v["by_model"].items():
        rendered = f"{stats['median_relative_error']:+.1%}".replace("+", "+").replace(
            "-", "−"
        )
        plain = f"{stats['median_relative_error']:+.1%}"
        assert rendered in text or plain in text, (
            f"{model} differs by {plain}; the manuscript does not quote that number"
        )


def test_the_coverage_of_the_reconstruction_check_is_stated(text):
    """The limitation is the point of the section: the models the study reconstructs
    for are mostly not the models the check could reach."""
    coverage = _reconstruction_validation()["coverage_of_use"]
    assert (
        f"{coverage['n_series_using_a_reconstructed_value']} series whose index rests"
        in text
    )
    assert f"the other {coverage['n_on_an_unmeasured_model']} are GE" in text

    # The one series on a model the check could reach must be quoted at the error that
    # record holds. It was quoted at +7.7% -- one of the two per-series errors for that
    # model -- two paragraphs after the same model was reported at its median of -2.0%,
    # so the section disagreed with itself on the number a reviewer asking about
    # reconstruction accuracy would go looking for.
    measured = [s for s in coverage["series"] if s["measured"]]
    for s in measured:
        rendered = f"{s['median_relative_error']:+.1%}"
        assert rendered in text or rendered.replace("-", "−") in text, (
            f"the {s['model_name']} series using a reconstructed value is recorded at "
            f"{rendered}; the manuscript quotes a different number for it"
        )


def test_the_index_is_defined_as_display_mathematics(raw):
    """Reviewer 1 objected that the formula was presented "as a single line". It was:
    an ASCII transliteration with sigma written as a letter. It is now LaTeX display
    mathematics, which pandoc renders as Word equation objects rather than printing
    the source, and this pins the notation so it cannot drift back."""
    section = raw.split("### 2.8.", 1)[1].split("###", 1)[0]
    assert section.count("$$") >= 6, "the index must be defined in display mathematics"
    for token in (r"\frac", r"\sum_z", r"\bar{I}", r"\mathrm{CTDIvol}"):
        assert token in section, f"the definition no longer uses {token}"
    # The transliterated form, and anything like it, must not come back.
    assert "Σ_z" not in raw, "a sigma written as a letter is not an equation"


def test_no_equation_tag_survives_that_the_renderer_drops(raw):
    r"""\tag{} is silently dropped on the Word and Typst paths, so an equation number
    written that way appears in the source and not in the document a reader gets."""
    assert r"\tag{" not in raw, (
        r"\tag{} does not render here; either number the equation another way or do "
        "not number it"
    )


LETTER = REPO / "docs" / "review_r1" / "response_to_reviewers.md"


def _letter():
    if not LETTER.is_file():
        pytest.skip("no response letter in this checkout")
    return " ".join(LETTER.read_text(encoding="utf-8").split())


def test_the_response_letter_points_at_sections_that_exist(raw):
    """The letter is a copy of the manuscript's claims and drifts like every other
    copy. A reply citing a section number the paper does not have is the worst kind of
    error to make in front of an editor."""
    letter = _letter()
    for section in re.findall(r"Section (\d+\.\d+)", letter):
        assert f"### {section}." in raw or f"## {section}." in raw, (
            f"the reply cites Section {section}, which the manuscript does not have"
        )


def test_the_response_letter_quotes_the_manuscripts_own_numbers(raw):
    """The round-1 letter, checked against what the manuscript said when it was sent.

    The abstract word count was checked here too, until the production editor asked at
    proof for the four MDPI subheadings. Adding them changed the count, and the round-1
    letter is a document that was sent in August: it cannot be wrong about a manuscript
    that changed after it. A historical letter is validated against its own state, not
    against the current one, so the coupling is removed rather than the letter edited.

    The reference count stays: the letter's claim about the size of the bibliography is
    about a list that has not changed and must not silently drift.
    """
    letter = _letter()
    references = len(re.findall(r"(?m)^\d+\.\s", raw.split("## References")[1]))
    assert f"list to {references}" in letter, (
        f"the reply says the list reaches a different number than the {references} present"
    )


def test_the_response_letter_does_not_claim_changes_that_were_not_made(raw):
    """Two comments are answered by pointing at text that was already there. The letter
    must keep saying so: presenting an unchanged passage as a revision is the one thing
    an editor can check instantly and will not forgive."""
    letter = _letter()
    assert "Already present in the submitted version" in letter
    assert "Largely present in the submitted version" in letter
    assert "We have not restated what was there" in letter


BUILDS = (
    REPO / "paper" / "manuscript_tomography.docx",
    REPO / "paper" / "manuscript_tomography_revised_highlighted.docx",
)


def _document_xml(path: Path) -> str:
    import zipfile

    with zipfile.ZipFile(path) as archive:
        return archive.read("word/document.xml").decode("utf-8")


def _plain(path: Path) -> str:
    return re.sub(r"<[^>]+>", "", _document_xml(path))


def test_no_built_document_carries_an_unresolved_release_placeholder():
    """The highlighted build read the source directly instead of going through the
    preparation step, so {{RELEASE_TAG}} reached a document meant for an editor while
    the clean build, which refuses to produce one, was fine. Both are checked now,
    because a check that covers one of two build paths covers neither."""
    for path in BUILDS:
        if not path.is_file():
            continue
        left = _plain(path).count("{{")
        assert left == 0, f"{path.name} carries {left} unresolved placeholder(s)"


def test_the_reference_list_survives_as_separate_entries_in_every_build():
    """Wrapping the bibliography in a highlight span collapsed twenty list items into
    one paragraph: entry numbers and DOIs ran together mid-line. It renders, so nothing
    upstream complains; a reviewer sees a broken bibliography."""
    for path in BUILDS:
        if not path.is_file():
            continue
        text = _plain(path)
        run_on = re.findall(r"https://doi\.org/[^\s]+ \d+\. ", text)
        assert not run_on, (
            f"{path.name}: {len(run_on)} reference entries run into the next; the list "
            "has been flattened into a paragraph"
        )


def test_the_highlighted_build_marks_some_paragraphs_and_not_all():
    """A build that highlights nothing has silently stopped marking; one that
    highlights everything tells a reviewer nothing about what changed."""
    path = REPO / "paper" / "manuscript_tomography_revised_highlighted.docx"
    if not path.is_file():
        pytest.skip("highlighted revision not built in this checkout")
    document = _document_xml(path)
    marked = document.count("w:highlight")
    assert marked > 0, "the highlighted build marks nothing"
    assert marked < document.count("<w:r>"), "the highlighted build marks everything"


REVIEW_DIR = REPO / "docs" / "review_r1"


def _reports_plain() -> str:
    path = REVIEW_DIR / "reviewer_reports.md"
    if not path.is_file():
        pytest.skip("reviewer reports not in this checkout")
    lines = path.read_text(encoding="utf-8").splitlines()
    stripped = [
        line[2:] if line.startswith("> ") else ("" if line.strip() == ">" else line)
        for line in lines
    ]
    return " ".join(" ".join(stripped).split())


def test_every_comment_in_the_per_reviewer_replies_is_quoted_verbatim():
    """SuSy's template asks for the reviewer's own words above each response. A
    paraphrase there reads as an author answering a question that was not asked."""
    report = _reports_plain()
    seen = 0
    for n in (1, 2):
        path = REVIEW_DIR / f"response_reviewer_{n}.md"
        if not path.is_file():
            pytest.skip("per-reviewer replies not generated in this checkout")
        text = path.read_text(encoding="utf-8")
        for number, comment in re.findall(r"\*\*Comments (\d+):\*\* (.+)", text):
            seen += 1
            probe = " ".join(comment.split()).lstrip("1234567890) ")[:80]
            assert probe in report, (
                f"reviewer {n} comment {number} is not quoted verbatim from the report"
            )
    assert seen == 18, f"18 comments were received; the replies carry {seen}"


def test_each_reply_pairs_every_comment_with_exactly_one_response():
    for n in (1, 2):
        path = REVIEW_DIR / f"response_reviewer_{n}.md"
        if not path.is_file():
            pytest.skip("per-reviewer replies not generated in this checkout")
        text = path.read_text(encoding="utf-8")
        comments = [int(x) for x in re.findall(r"\*\*Comments (\d+):\*\*", text)]
        responses = [int(x) for x in re.findall(r"\*\*Response (\d+):\*\*", text)]
        assert comments == responses == list(range(1, len(comments) + 1)), (
            f"reviewer {n}: comments {comments} against responses {responses}"
        )


def test_the_revision_cover_letter_is_a_revision_letter_not_a_submission_one(raw):
    """The portal pre-fills the box with the original submission's cover letter, which
    opens "I am pleased to submit ... for consideration". Sending that back with a
    revision tells an editor nobody read the screen."""
    path = REVIEW_DIR / "cover_letter_revision.txt"
    if not path.is_file():
        pytest.skip("no revision cover letter in this checkout")
    letter = " ".join(path.read_text(encoding="utf-8").split())
    assert "pleased to submit" not in letter
    assert "revised version of manuscript tomography-4516935" in letter

    references = len(re.findall(r"(?m)^\d+\.\s", raw.split("## References")[1]))
    assert references == 20 and "twenty" in letter, (
        f"the letter says twenty references and the manuscript holds {references}"
    )
    for section in re.findall(r"Section (\d+\.\d+)", letter):
        assert f"### {section}." in raw, (
            f"the cover letter cites Section {section}, which does not exist"
        )


# --- round-2 additions -------------------------------------------------------


def _sensitivity():
    path = REPO / "results" / "sensitivity_reconstructed_1.5mm.json"
    if not path.is_file():
        pytest.skip("run tools/sensitivity_reconstructed.py")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_sensitivity_numbers_in_the_discussion_are_the_computed_ones(text):
    """Reviewer 2 asked what a reconstructed CTDIvol does to the index. The answer is a
    measurement, so it can go stale exactly like every other measurement here."""
    s = _sensitivity()
    e = s["exposure"]
    assert f"{e['n_series_analysed']} series carrying an" in text
    assert f"{e['n_series_on_a_reconstructed_value']} rest on a reconstructed" in text
    assert (
        f"{e['n_organ_records_on_a_reconstructed_value']} of "
        f"{e['n_organ_records_analysed']} organ records" in text
    )

    worst_weight = max(
        abs(r["relative_weight"]["delta"]) for r in s["by_organ"].values()
    )
    worst_index = max(
        abs(r["organ_weighted_ctdivol_mgy"]["delta"]) for r in s["by_organ"].values()
    )
    assert f"at most {worst_weight:.3f}" in text, (
        f"the largest weight shift is {worst_weight:.3f}"
    )
    assert f"at most {worst_index:.2f} mGy" in text, (
        f"the largest index shift is {worst_index:.2f} mGy"
    )


def test_the_ge_consequence_is_not_dropped_from_the_discussion(text):
    """The least convenient number in the sensitivity analysis, and the one a response
    letter is most tempted to leave out."""
    reach = _sensitivity()["vendor_reach"]
    assert reach["recorded_only"]["GE"] == 0
    assert f"from {reach['published']['GE']} series to none" in text
    assert "no GE series in this cohort" in text


def test_the_per_model_counts_in_the_limitations_are_the_cohort_s(text, tables):
    """Reviewer 2 asked about sample size per manufacturer. The sharper number is per
    model, and it is the one a reader cannot get from the cohort table."""
    path = REPO / "results" / "series_1.5mm.csv"
    if not path.is_file():
        pytest.skip("run tools/make_analysis.py")
    import csv

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    per_vendor = {
        vendor: len({r["model_name"] for r in rows if r["vendor"] == vendor})
        for vendor in ("GE", "Siemens", "Canon/Toshiba", "Philips")
    }
    assert (
        f"{per_vendor['GE']} distinct GE models, {per_vendor['Siemens']} Siemens, "
        f"{per_vendor['Canon/Toshiba']} Canon/Toshiba and {per_vendor['Philips']} Philips"
        in text
    ), f"the cohort holds {per_vendor}"


def test_the_failed_mask_is_reported_with_what_removing_it_does(text):
    """Section 2.6 explained only the high tail of the mass flags. The low tail is a
    segmentation failure, and reporting it without its effect would be an admission
    with no number attached."""
    import copy

    from ctsegdose_core.analysis import weighted_ctdivol
    from ctsegdose_core.eligibility import assess

    records = REPO / "results" / "organ_dose_1.5mm.json"
    constancy = REPO / "results" / "acquisition_constancy.json"
    if not (records.is_file() and constancy.is_file()):
        pytest.skip("run tools/run_organ_dose.py")
    payload = json.loads(records.read_text(encoding="utf-8"))
    eligible = {
        uid
        for uid, e in assess(json.loads(constancy.read_text(encoding="utf-8"))).items()
        if e.eligible
    }

    worst = min(
        (
            o
            for s in payload["series"]
            for o in s.get("organs", [])
            if o["organ"] == "kidney_left" and not o.get("truncated") and o.get("mass_g")
        ),
        key=lambda o: float(o["mass_g"]),
    )
    assert f"{float(worst['mass_g']):.1f} g" in text
    assert f"{float(worst['volume_cm3']):.1f} cm" in text

    # Recomputed through the published code path, not over the raw CSV. Quoting a
    # median from a looser population than the table the paper prints is how a
    # sentence ends up almost right: the first draft of this one said 1.037 to 1.036,
    # which is the pair for records that skip the acquisition-constancy screen.
    def median_for(series):
        return weighted_ctdivol(series, eligible)["by_organ"]["kidney_left"][
            "relative_weight"
        ]["median"]

    trimmed = copy.deepcopy(payload["series"])
    for s in trimmed:
        s["organs"] = [
            o for o in s.get("organs", [])
            if not (
                o["organ"] == "kidney_left"
                and o.get("mass_g")
                and float(o["mass_g"]) == float(worst["mass_g"])
            )
        ]
    with_it, without = median_for(payload["series"]), median_for(trimmed)
    assert f"from {with_it:.3f} to {without:.3f}" in text, (
        f"removing it moves the published median from {with_it:.3f} to {without:.3f}"
    )


def test_the_ai_usage_statement_is_in_the_methods_where_the_editor_asked(raw):
    """The assistant editor asked for it in the Methods section. The manuscript already
    carried one in the back matter, which is the version of this request most easily
    marked as already done."""
    # Flattened: these are phrase assertions, and prose rewraps. The module docstring
    # makes the same point -- a test that fails on reflowing trains its reader to ignore
    # it -- and this one failed that way the moment the statement was rewrapped.
    methods = " ".join(
        raw.split("## 2. Materials and Methods", 1)[1].split("## 3. Results", 1)[0].split()
    )
    assert "Generative Artificial Intelligence" in methods, (
        "the AI usage statement is not in the Methods section"
    )
    assert "not used to generate, impute or select any reported value" in methods

    # Naming the model, not just the product. MDPI asked which version was used after
    # the round-1 revision, and a statement that says only "Claude" invites the same
    # question again. Both statements must name it, and must name the same one.
    flat = " ".join(raw.split())
    named = set(re.findall(r"\(Claude ([\w. ]+?), Anthropic\)", flat))
    assert named, "the AI usage statement does not name a model version"
    assert len(named) == 1, f"the two statements name different models: {named}"
    model = f"(Claude {next(iter(named))}, Anthropic)"
    back_matter = flat.split("Use of Generative Artificial Intelligence", 1)[-1]
    for section, where in ((methods, "Methods"), (back_matter, "back matter")):
        assert model in section, f"the {where} statement does not name the model version"


def test_the_release_the_paper_cites_contains_the_results_the_paper_quotes():
    """The Data Availability Statement claims the machine-readable results the
    manuscript quotes are in the repository at the release it names. That claim was
    false after the round-2 revision: the sensitivity analysis and its result file were
    added after v0.1.1, which is the release the statement pointed at, so a reader
    following the DOI would not have found the analysis the Discussion describes.

    Checked against the git tag rather than the working tree, because the working tree
    is exactly what the reader does not get.
    """
    import subprocess

    metadata = REPO / "paper" / "release_metadata.json"
    if not metadata.is_file():
        pytest.skip("no release metadata in this checkout")
    tag = json.loads(metadata.read_text(encoding="utf-8-sig")).get("RELEASE_TAG")
    if not tag:
        pytest.skip("no release tag recorded yet")

    listed = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", tag],
        cwd=REPO, capture_output=True, text=True,
    )
    if listed.returncode:
        pytest.skip(f"tag {tag} is not in this checkout")
    inside = set(listed.stdout.split())

    required = {
        path.relative_to(REPO).as_posix()
        for path in (REPO / "results").glob("*.json")
    }
    missing = sorted(required - inside)
    assert not missing, (
        f"the manuscript cites release {tag}, which does not contain "
        f"{missing}. Cut a new release and update paper/release_metadata.json."
    )


def test_the_archive_metadata_declares_the_version_the_package_does():
    """Zenodo builds the deposit from .zenodo.json, so a stale version there labels the
    archived snapshot with the number of the previous release -- in metadata a minted
    DOI makes permanent. It was stale when v0.1.2 was being prepared, which is how this
    check came to exist.
    """
    path = REPO / ".zenodo.json"
    if not path.is_file():
        pytest.skip("no Zenodo metadata in this checkout")
    declared = re.search(
        r'^version = "([^"]+)"',
        (REPO / "pyproject.toml").read_text(encoding="utf-8"),
        re.M,
    )
    assert declared, "pyproject.toml declares no version"
    archive = json.loads(path.read_text(encoding="utf-8"))
    assert archive["version"] == declared.group(1), (
        f".zenodo.json says {archive['version']}, pyproject says {declared.group(1)}"
    )

    citation = REPO / "CITATION.cff"
    if citation.is_file():
        assert f"version: {declared.group(1)}" in citation.read_text(encoding="utf-8"), (
            "CITATION.cff declares a different version again"
        )


def test_the_concept_doi_is_a_relation_and_not_a_top_level_doi():
    """A top-level "doi" in .zenodo.json tells Zenodo the DOI was assigned elsewhere and
    stops it minting a version DOI for the release. The file's own notes say so; this
    keeps a later edit from undoing it."""
    path = REPO / ".zenodo.json"
    if not path.is_file():
        pytest.skip("no Zenodo metadata in this checkout")
    archive = json.loads(path.read_text(encoding="utf-8"))
    assert "doi" not in archive, (
        "a top-level doi in .zenodo.json stops Zenodo versioning this release"
    )
    relations = {r["relation"] for r in archive.get("related_identifiers", [])}
    assert "isVersionOf" in relations, "the concept DOI is not recorded as a relation"


#: The affiliation exactly as Crossref carries it for this author's published paper,
#: 10.3390/jimaging12080392, checked on 2026-08-30. City before postcode.
CANONICAL_AFFILIATION = "Institute of One, LISIT Co., Ltd., Tokyo 150-0044, Japan"


def test_the_affiliation_is_the_string_the_published_record_carries(raw):
    """Not merely "has a postal code" -- the exact string, in the exact order.

    The second Tomography proof went out reading "150-0044 Tokyo", while the author's
    already-published Journal of Imaging paper is indexed as "Tokyo 150-0044". Affiliation
    matching, and the ROR curation this programme is building towards, compare these
    strings: one author with two orderings is one author who looks like two.

    MDPI production also asks for the postal code at proof on every submission, and asked
    twice before it went into the source, so carrying the whole string here stops both
    problems at once.
    """
    header = raw.split("## Simple Summary", 1)[0]
    assert CANONICAL_AFFILIATION in header, (
        "the affiliation must read exactly as the published record carries it:\n"
        f"  {CANONICAL_AFFILIATION}"
    )
    assert "0000-0001-9211-1071" in header


def test_no_line_break_splits_a_hyphenated_word(raw):
    """A markdown line break renders as a space, so wrapping after a hyphen turns
    "single-institution" into "single- institution" in the built document.

    It happened while rewrapping the abstract, and it survived a second rewrap because
    the damage was already in the text being rewrapped. Nothing else in the pipeline
    would have caught it: the word count is unchanged, every number still matches, and
    the source looks like ordinary wrapping.
    """
    offenders = []
    lines = raw.splitlines()
    for i, line in enumerate(lines[:-1]):
        if not line.endswith("-"):
            continue
        nxt = lines[i + 1].strip()
        if nxt and nxt[0].islower():
            offenders.append(f"line {i + 1}: {line[-40:]!r} + {nxt[:30]!r}")
    assert not offenders, (
        "a line break inside a hyphenated word renders as a space:\n"
        + "\n".join(offenders)
    )


def test_a_cover_letter_for_a_web_form_is_not_hard_wrapped():
    """SuSy's cover-letter box is a textarea: it keeps the newlines it is given.

    A letter wrapped at 90 columns for readability in the repository arrives in the
    editor's box broken mid-sentence on every line. It happened on this proof, and the
    fix is one line per paragraph with blank lines between them.

    Scoped to docs/proof/, which is the letter being sent. The round-1 and round-2
    letters are records of what was already submitted and are left as they were sent.
    """
    directory = REPO / "docs" / "proof"
    letters = sorted(directory.glob("cover_letter*.txt")) if directory.is_dir() else []
    if not letters:
        pytest.skip("no cover letter staged for a proof return")
    for path in letters:
        lines = path.read_text(encoding="utf-8").split("\n")
        for i, line in enumerate(lines[:-1]):
            nxt = lines[i + 1]
            if not line or not nxt:
                continue
            # A continued line whose predecessor does not close a sentence is a wrap.
            if not line.rstrip().endswith((".", ",", ":", ";", '"', "?", "!")):
                continue
            assert not (line.rstrip().endswith(",") and nxt[:1].islower()), (
                f"{path.name} line {i + 1} looks hard-wrapped: {line[-40:]!r}"
            )
        longest = max(len(l) for l in lines)
        assert longest > 120, (
            f"{path.name} has no line longer than {longest} characters, which means its "
            "paragraphs are wrapped; a textarea keeps those newlines"
        )
