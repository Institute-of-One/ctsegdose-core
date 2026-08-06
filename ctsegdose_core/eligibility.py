"""Which series may enter the quantitative modulation analysis, and why.

The anatomy-weighted index treats the recorded tube current as a proxy for the
longitudinal variation in scanner output. That substitution is valid only while the other
quantities governing output are fixed within the series. Where they are not, the weight
still describes the recorded current faithfully but no longer describes the output, and
the series cannot contribute to a quantitative statement about modulation.

The criterion is therefore stated once, in advance, and applied mechanically:

    A series is eligible for quantitative anatomy-weighted CTDIvol analysis only when the
    acquisition parameters required for scanner output to remain proportional to the
    recorded tube current are constant within that series, to the extent verifiable from
    the archived DICOM headers.

Four outcomes are distinguished per attribute, and only the last disqualifies:

``verified_constant``
    One value across every slice.
``absent``
    Never written to the archived headers. Constancy cannot be verified either way, and
    absence alone does not disqualify -- excluding on it would remove series for a
    property of the de-identified export rather than of the acquisition, and would
    disproportionately remove whole manufacturers.
``negligible_variation``
    Varies, but by less than :data:`ROUNDING_TOLERANCE` in relative terms. Exposure time
    is written as an integer number of milliseconds, so a one-unit step on a value of a
    few hundred is a rounding artefact of the representation, not a change in technique.
``materially_variable``
    Varies by at least that tolerance. The acquisition genuinely changed part-way through
    the series, so tube current alone cannot describe how the output varied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Relative spread below which a variation is attributed to numeric rounding in the
#: header rather than to the acquisition.
#:
#: Two percent. Exposure time is recorded as an integer number of milliseconds, so the
#: representation alone can move a 400--700 ms value by up to about 0.25%; two percent
#: leaves an order of magnitude of headroom above that while remaining far below any
#: real change of technique, the smallest of which -- halving or doubling the rotation
#: time -- is a hundred percent. The threshold is therefore not fitted to the data: any
#: value between roughly 1% and 50% classifies this cohort identically.
ROUNDING_TOLERANCE = 0.02

#: The attributes that, with tube current, determine scanner output. Absence of any of
#: these is tolerated; material variation in any of them is not.
OUTPUT_GOVERNING = (
    "tube_voltage_kvp",
    "exposure_time_ms",
    "rotation_time_s",
    "spiral_pitch_factor",
    "total_collimation_width_mm",
)

VERIFIED = "verified_constant"
ABSENT = "absent"
NEGLIGIBLE = "negligible_variation"
MATERIAL = "materially_variable"


def classify(entry: dict[str, Any], *, tolerance: float = ROUNDING_TOLERANCE) -> str:
    """Classify one attribute's behaviour within one series."""
    status = entry.get("status")
    if status in ("absent", "partially_absent"):
        return ABSENT
    if status == "constant":
        return VERIFIED
    spread = entry.get("relative_spread")
    if spread is None:
        # Varies but is not numeric (a text attribute); treat any change as material.
        return MATERIAL
    return NEGLIGIBLE if abs(spread) < tolerance else MATERIAL


@dataclass(frozen=True)
class Eligibility:
    """Whether one series may enter the quantitative modulation analysis."""

    series_uid: str
    vendor: str
    eligible: bool
    by_attribute: dict[str, str]
    disqualifying: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_instance_uid": self.series_uid,
            "vendor": self.vendor,
            "eligible": self.eligible,
            "by_attribute": dict(self.by_attribute),
            "disqualifying_attributes": list(self.disqualifying),
        }


def assess(constancy: dict[str, Any], *, tolerance: float = ROUNDING_TOLERANCE) -> dict[str, Eligibility]:
    """Apply the criterion to every series in an acquisition-constancy record."""
    out: dict[str, Eligibility] = {}
    for uid, report in constancy["by_series"].items():
        by_attribute = {
            name: classify(report[name], tolerance=tolerance)
            for name in OUTPUT_GOVERNING
            if name in report
        }
        disqualifying = tuple(n for n, c in by_attribute.items() if c == MATERIAL)
        out[uid] = Eligibility(
            series_uid=uid,
            vendor=report.get("vendor", ""),
            eligible=not disqualifying,
            by_attribute=by_attribute,
            disqualifying=disqualifying,
        )
    return out


def summarise(assessed: dict[str, Eligibility]) -> dict[str, Any]:
    """The criterion's effect on the cohort, for the record and for the manuscript."""
    ineligible = [e for e in assessed.values() if not e.eligible]
    counts: dict[str, dict[str, int]] = {}
    for name in OUTPUT_GOVERNING:
        tally: dict[str, int] = {}
        for e in assessed.values():
            key = e.by_attribute.get(name)
            if key:
                tally[key] = tally.get(key, 0) + 1
        counts[name] = tally
    return {
        "criterion": (
            "A series is eligible for quantitative anatomy-weighted CTDIvol analysis "
            "only when the acquisition parameters required for scanner output to remain "
            "proportional to the recorded tube current are constant within that series, "
            "to the extent verifiable from the archived DICOM headers."
        ),
        "rounding_tolerance": ROUNDING_TOLERANCE,
        "output_governing_attributes": list(OUTPUT_GOVERNING),
        "n_series_assessed": len(assessed),
        "n_eligible": sum(1 for e in assessed.values() if e.eligible),
        "n_ineligible": len(ineligible),
        "ineligible_series": [e.to_dict() for e in ineligible],
        "classification_counts": counts,
        "note": (
            "Ineligible series are retained for archive-availability, segmentation, "
            "estimated-mass and provenance analyses; they are excluded only from "
            "quantitative modulation-weight and anatomy-weighted-index summaries."
        ),
    }
