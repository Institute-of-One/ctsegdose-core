"""The round-2 sensitivity analysis, recomputed from the raw records.

``results/sensitivity_reconstructed_<tag>.json`` answers a reviewer question, which
means it will be quoted in a response letter and then in the manuscript. It is
derived, so it can go stale exactly like every other table here, and a stale
sensitivity result is worse than none: it would be cited as evidence that the
conclusions do not move while no longer describing the cohort they were computed on.

Skips when the source files are absent, so a fresh clone stays green.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TAG = "1.5mm"
SOURCE = REPO / "results" / f"organ_dose_{TAG}.json"
CONSTANCY = REPO / "results" / "acquisition_constancy.json"
FROZEN = REPO / "results" / f"sensitivity_reconstructed_{TAG}.json"

pytestmark = pytest.mark.skipif(
    not (SOURCE.is_file() and CONSTANCY.is_file() and FROZEN.is_file()),
    reason="run tools/sensitivity_reconstructed.py",
)


@pytest.fixture(scope="module")
def payload():
    return json.loads(SOURCE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def constancy():
    return json.loads(CONSTANCY.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def frozen():
    return json.loads(FROZEN.read_text(encoding="utf-8"))


def test_the_committed_result_is_reproducible(payload, constancy, frozen):
    from tools.sensitivity_reconstructed import analyse

    assert analyse(payload, constancy) == frozen, (
        "results/sensitivity_reconstructed_1.5mm.json is stale: re-run "
        "tools/sensitivity_reconstructed.py"
    )


def test_the_weight_does_not_contain_ctdivol(payload):
    """The structural claim the response letter makes, checked against the records.

    If w_o were a function of CTDIvol, two series on the same scanner with the same
    modulation and different CTDIvol would carry different weights. The definition
    says it is not; this asserts the recorded values behave that way, by scaling
    CTDIvol and confirming the weight is untouched.
    """
    for series in payload["series"]:
        for organ in series.get("organs", []):
            if organ.get("truncated") or series["ctdivol_mgy"] is None:
                continue
            ctdivol = float(series["ctdivol_mgy"])
            expected = ctdivol * float(organ["relative_weight"])
            # The tolerance is derived from how the records store the two quantities
            # rather than chosen: the weight is written to four decimals and the
            # product to three, so the product can differ from the recomputation by
            # half a unit in each last place. A definition that did not hold would
            # miss by orders of magnitude, not by half a stored digit.
            tolerance = 0.5e-3 + ctdivol * 0.5e-4
            assert organ["organ_weighted_ctdivol_mgy"] == pytest.approx(
                expected, abs=tolerance
            ), (
                "organ-weighted CTDIvol is not CTDIvol * w_o, so the 1:1 propagation "
                "the response letter claims does not hold"
            )


def test_the_exposure_counts_match_the_records(payload, constancy, frozen):
    from ctsegdose_core.analysis import _completed, _ctdivol_class
    from ctsegdose_core.eligibility import assess

    eligible = {uid for uid, e in assess(constancy).items() if e.eligible}
    usable = [
        s for s in _completed(payload["series"])
        if s["ctdivol_mgy"] is not None and s["series_instance_uid"] in eligible
    ]
    recon = [s for s in usable if _ctdivol_class(s["ctdivol_source"]) == "reconstructed"]
    exposure = frozen["exposure"]

    assert exposure["n_series_analysed"] == len(usable)
    assert exposure["n_series_on_a_reconstructed_value"] == len(recon)
    assert exposure["n_organ_records_on_a_reconstructed_value"] == sum(
        1 for s in recon for o in s["organs"] if not o.get("truncated")
    )


def test_the_relative_weights_barely_move(frozen):
    """The finding the response letter rests on. Stated as a bound rather than as a
    set of numbers, so it keeps meaning if the cohort is ever extended."""
    shifts = [
        abs(row["relative_weight"]["delta"])
        for row in frozen["by_organ"].values()
        if row.get("restricted", True) is not None
    ]
    assert shifts, "no organ survived the restriction"
    assert max(shifts) < 0.1, (
        f"the largest weight shift is {max(shifts)}, which is no longer the 'barely "
        "moves' the response letter describes"
    )
    assert st.median(shifts) < 0.05


def test_the_ge_consequence_is_recorded(frozen):
    """Excluding the reconstructed values removes GE from the weighted tables
    entirely. It is the least convenient number here and the one most likely to be
    dropped from a response letter, so it is asserted rather than left to prose."""
    reach = frozen["vendor_reach"]
    assert reach["published"]["GE"] > 0
    assert reach["recorded_only"]["GE"] == 0, (
        "the limitations say the multi-vendor reach of the weighted tables rests on "
        "the reconstructed GE values; this no longer holds"
    )


def test_no_organ_is_silently_lost(frozen):
    """A per-organ row that vanished under the restriction would make the comparison
    look better than it is."""
    lost = [
        organ
        for organ, row in frozen["by_organ"].items()
        if row.get("restricted", True) is None
    ]
    assert not lost, f"organs with no recorded-CTDIvol series left: {lost}"
