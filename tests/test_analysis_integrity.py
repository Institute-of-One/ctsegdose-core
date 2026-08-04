"""Anti-fabrication: every number the manuscript quotes, recomputed from the raw records.

``results/analysis_<tag>.json`` is the table the paper reads from. It is derived, so it
can drift: from a hand edit, from a stale file left after a re-run, from an analysis
change that was never re-applied. These tests recompute each published figure directly
from ``results/organ_dose_<tag>.json`` -- independently of the code that produced the
tables, wherever that is practical -- and fail on disagreement.

They skip when the files are absent, so a fresh clone stays green.
"""

from __future__ import annotations

import json
import statistics as st
from collections import Counter
from pathlib import Path

import pytest

from ctsegdose_core.analysis import FLAT_WEIGHT_THRESHOLD, ICRP89_REFERENCE_MASS_G, build

REPO = Path(__file__).resolve().parents[1]
TAG = "1.5mm"
SOURCE = REPO / "results" / f"organ_dose_{TAG}.json"
TABLES = REPO / "results" / f"analysis_{TAG}.json"


def _load(path: Path):
    if not path.exists():
        pytest.skip(f"{path.relative_to(REPO)} has not been generated in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def payload():
    return _load(SOURCE)


@pytest.fixture(scope="module")
def tables():
    return _load(TABLES)


@pytest.fixture(scope="module")
def organs(payload):
    """Every organ record, with its series' vendor attached."""
    return [
        {**o, "vendor": s["vendor"], "series": s["series_instance_uid"]}
        for s in payload["series"]
        for o in s.get("organs", [])
    ]


# --- the shipped tables must be the tables the code produces -----------------------------


def test_the_shipped_tables_are_reproduced_exactly_from_the_per_series_records(payload, tables):
    """The whole file, regenerated. A hand edit anywhere fails here."""
    rebuilt = build(payload)
    shipped = {k: v for k, v in tables.items() if k != "provenance"}
    assert rebuilt == shipped


def test_the_tables_name_the_records_they_came_from(tables):
    assert tables["provenance"]["source"].endswith(f"organ_dose_{TAG}.json")
    assert tables["provenance"]["ctsegdose_core_version"]


# --- the cohort --------------------------------------------------------------------------


def test_the_cohort_counts_match_the_records(payload, tables, organs):
    cohort = tables["cohort"]
    assert cohort["n_series"] == len(payload["series"])
    assert cohort["n_organ_records"] == len(organs)
    assert sum(cohort["series_by_vendor"].values()) == cohort["n_series"]


def test_the_sample_is_balanced_across_the_four_vendors(tables):
    counts = set(tables["cohort"]["series_by_vendor"].values())
    assert counts == {10}, "the multi-vendor claim rests on this being balanced"


# --- (1) availability, the headline ------------------------------------------------------


def test_the_availability_counts_are_re_derived_from_each_series_ctdivol_source(payload, tables):
    for vendor, block in tables["availability"]["by_vendor"].items():
        rows = [s for s in payload["series"] if s["vendor"] == vendor]
        assert block["n_series"] == len(rows)
        recorded = sum(1 for s in rows if "recorded" in s["ctdivol_source"])
        reconstructed = sum(1 for s in rows if "reconstructed" in s["ctdivol_source"])
        assert block["recorded"] == recorded
        assert block["reconstructed"] == reconstructed
        assert block["unrecoverable"] == len(rows) - recorded - reconstructed


def test_every_series_falls_into_exactly_one_availability_class(tables):
    overall = tables["availability"]["overall"]
    assert (
        overall["recorded"] + overall["reconstructed"] + overall["unrecoverable"]
        == overall["n_series"]
    )


def test_the_headline_vendor_difference_is_what_the_records_say(payload, tables):
    """GE recording no CTDIvol is the paper's headline; it is checked, not asserted."""
    ge = [s for s in payload["series"] if s["vendor"] == "GE"]
    assert all("recorded" not in s["ctdivol_source"] for s in ge)
    assert tables["availability"]["by_vendor"]["GE"]["recorded"] == 0

    table = tables["availability"]["ge_vs_rest_recorded"]["table"]
    assert table["ge_recorded"] + table["ge_not_recorded"] == len(ge)
    p = tables["availability"]["ge_vs_rest_recorded"].get("p_value")
    if p is not None:
        assert p < 0.001, "the quoted significance must follow from the quoted counts"


def test_the_unrecoverable_series_really_carry_no_dose_index(payload):
    for s in payload["series"]:
        if "unavailable" in s["ctdivol_source"]:
            assert s["ctdivol_mgy"] is None
            assert all(o["absorbed_dose_mgy"] is None for o in s.get("organs", []))


# --- (2) organ mass ----------------------------------------------------------------------


def test_the_organ_mass_medians_are_re_derived_from_the_untruncated_records(tables, organs):
    for organ, block in tables["organ_mass"]["overall"].items():
        masses = [o["mass_g"] for o in organs if o["organ"] == organ and not o["truncated"]]
        assert block["mass_g"]["n"] == len(masses)
        assert block["mass_g"]["median"] == pytest.approx(round(st.median(masses), 1))


def test_the_icrp_ratios_follow_from_the_medians_and_the_reference(tables):
    for organ, block in tables["organ_mass"]["overall"].items():
        reference = ICRP89_REFERENCE_MASS_G.get(organ)
        if reference is None:
            assert "median_over_reference" not in block
            continue
        assert block["icrp89_reference_mass_g"] == reference
        assert block["median_over_reference"] == pytest.approx(
            round(block["mass_g"]["median"] / reference, 3), abs=0.002
        )


def test_truncated_organs_are_excluded_from_the_mass_comparison(tables, organs):
    """Comparing a partly-scanned organ with a whole-organ reference measures the field
    of view, not the segmentation."""
    for organ, block in tables["organ_mass"]["overall"].items():
        everything = sum(1 for o in organs if o["organ"] == organ)
        assert block["mass_g"]["n"] <= everything
        cut = sum(1 for o in organs if o["organ"] == organ and o["truncated"])
        assert block["mass_g"]["n"] == everything - cut


def test_the_two_systematic_offsets_the_paper_reports_are_real(tables):
    """The pancreas low and the spleen high are findings, not artefacts of rounding;
    if either moves inside 20% of the reference the manuscript text must change."""
    overall = tables["organ_mass"]["overall"]
    assert overall["pancreas"]["median_over_reference"] < 0.8
    assert overall["spleen"]["median_over_reference"] > 1.2
    assert 0.9 < overall["liver"]["median_over_reference"] < 1.15


# --- (3) the organ-level index -----------------------------------------------------------


def test_every_organ_weighted_ctdivol_is_its_ctdivol_times_its_weight(payload):
    for s in payload["series"]:
        for o in s["organs"]:
            if s["ctdivol_mgy"] is None:
                assert o["organ_weighted_ctdivol_mgy"] is None
                continue
            assert o["organ_weighted_ctdivol_mgy"] == pytest.approx(
                round(s["ctdivol_mgy"] * o["relative_weight"], 3), abs=0.01
            )


def test_a_missing_dose_index_is_json_null_and_never_a_nan(payload):
    """NaN is not JSON, and a reader must not be able to mistake it for a number."""
    raw = SOURCE.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    json.loads(raw, parse_constant=lambda c: pytest.fail(f"non-JSON constant {c!r}"))


def test_the_weight_distributions_are_re_derived(tables, organs):
    for organ, block in tables["weighted_ctdivol"]["by_organ"].items():
        weights = [
            o["relative_weight"] for o in organs
            if o["organ"] == organ
            and not o["truncated"]
            and o["organ_weighted_ctdivol_mgy"] is not None
        ]
        assert block["relative_weight"]["n"] == len(weights)
        assert block["relative_weight"]["median"] == pytest.approx(
            round(st.median(weights), 3), abs=0.002
        )


def test_no_absorbed_dose_is_reported_anywhere_in_the_shipped_results(payload):
    """IORN-006 stops at the index. A dose in mGy appearing here would mean a
    coefficient table was used, which nothing in this repository is licensed to do."""
    for s in payload["series"]:
        for o in s.get("organs", []):
            assert o["absorbed_dose_mgy"] is None
            assert o["coefficient_mgy_per_mgy"] is None


# --- (4) study limits --------------------------------------------------------------------


def test_the_truncation_tallies_are_re_derived(tables, organs):
    for vendor, block in tables["study_limits"]["truncation"]["by_vendor"].items():
        rows = [o for o in organs if o["vendor"] == vendor]
        assert block["n_organ_records"] == len(rows)
        assert block["n_truncated"] == sum(1 for o in rows if o["truncated"])


def test_the_flat_weighted_series_are_exactly_those_below_the_threshold(payload, tables):
    flat = set()
    for s in payload["series"]:
        w = [o["relative_weight"] for o in s.get("organs", [])]
        if len(w) > 1 and max(w) - min(w) < FLAT_WEIGHT_THRESHOLD:
            flat.add(s["series_instance_uid"])
    reported = {x["series"] for x in tables["study_limits"]["flat_weighting"]["series"]}
    assert reported == flat
    assert tables["study_limits"]["flat_weighting"]["n_series"] == len(flat)


def test_truncation_is_vendor_skewed_as_the_paper_states(tables):
    rates = {
        v: b["truncated_rate"]
        for v, b in tables["study_limits"]["truncation"]["by_vendor"].items()
    }
    assert rates["Philips"] > 2 * rates["GE"], (
        "the paper reports truncation as vendor-skewed; if that stops holding, the "
        "claim must be rewritten rather than the threshold moved"
    )


# --- the shipped CSVs --------------------------------------------------------------------


def test_the_organ_csv_has_one_row_per_organ_record(organs):
    path = REPO / "results" / f"organs_{TAG}.csv"
    if not path.exists():
        pytest.skip("organs CSV has not been generated")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) - 1 == len(organs)


def test_the_series_csv_has_one_row_per_series(payload):
    path = REPO / "results" / f"series_{TAG}.csv"
    if not path.exists():
        pytest.skip("series CSV has not been generated")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) - 1 == len(payload["series"])


def test_every_organ_appears_in_the_cohort_tally(tables, organs):
    assert Counter(o["organ"] for o in organs) == Counter(tables["cohort"]["organs_segmented"])
