"""Anatomical sanity checks on a completed segmentation.

A mirrored or inverted segmentation is the most dangerous failure this pipeline can
have, because it does not look like a failure. Volumes stay plausible, masses stay
plausible, Hounsfield values stay plausible — and every organ is weighted by the tube
current of the wrong part of the patient. Nothing downstream would notice.

So the results are checked against facts of gross anatomy that hold for every adult:

* the left kidney is to the patient's left of the right kidney, and the spleen is left
  of the liver — a left/right mirror breaks both;
* the liver lies superior to the urinary bladder, and the adrenal glands superior to the
  kidneys — a head/foot inversion breaks both;
* solid-organ volumes fall within a wide band around reference values — a gross
  segmentation failure, or a wrong voxel volume, breaks this while the flip checks pass.

These are screens, not validation. They catch the failures that are silent; they say
nothing about boundary accuracy, which is what comparing model resolutions is for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: ICRP Publication 89 reference masses for the adult male, in grams. Used only to set a
#: wide plausibility band: a factor of 2.5 either side, which no correct segmentation of
#: an adult abdomen leaves, and which a mirrored or mis-scaled one usually does.
#:
#: Only solid organs appear here. TotalSegmentator segments the stomach, bowel,
#: gallbladder and bladder including their *contents*, whereas ICRP tabulates wall mass,
#: so comparing those two would manufacture failures rather than find them.
ICRP89_REFERENCE_MASS_G: dict[str, float] = {
    "liver": 1800.0,
    "spleen": 150.0,
    "kidney_left": 155.0,
    "kidney_right": 155.0,
    "pancreas": 140.0,
}
ICRP89_REFERENCE = (
    "ICRP Publication 89 (2002), Basic Anatomical and Physiological Data for Use in "
    "Radiological Protection: Reference Values. Adult male reference organ masses."
)
PLAUSIBILITY_FACTOR = 2.5


@dataclass
class Check:
    """One assertion about a series, and whether it held.

    ``advisory`` marks an observation that has a legitimate explanation and so must not
    fail a run: every organ weight lying above the scan mean is what a chest-abdomen
    acquisition *should* produce, because the abdomen carries the higher tube current.
    Reporting it as a defect would train the reader to ignore the checks.
    """

    name: str
    passed: bool
    detail: str
    advisory: bool = False

    @property
    def is_failure(self) -> bool:
        return not self.passed and not self.advisory

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "advisory": self.advisory,
            "detail": self.detail,
        }


def _by_organ(series: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {o["organ"]: o for o in series.get("organs", [])}


def _centroid(organs: dict[str, dict[str, Any]], name: str) -> tuple[float, float, float] | None:
    row = organs.get(name)
    c = row.get("centroid_zyx") if row else None
    return tuple(float(v) for v in c) if c else None


def check_left_right(series: dict[str, Any]) -> list[Check]:
    """The patient's left is at a higher column index, because DICOM +x is left.

    Image Orientation (Patient) is expressed in patient coordinates, so this holds for a
    head-first and a feet-first acquisition alike — which is what makes it a usable
    invariant rather than a positioning assumption.
    """
    organs = _by_organ(series)
    out: list[Check] = []
    for left, right, what in (
        ("kidney_left", "kidney_right", "left kidney is left of right kidney"),
        ("spleen", "liver", "spleen is left of liver"),
    ):
        a, b = _centroid(organs, left), _centroid(organs, right)
        if a is None or b is None:
            out.append(Check(what, True, f"not checked: {left} or {right} absent"))
            continue
        ok = a[2] > b[2]
        out.append(
            Check(what, ok, f"x centroid {left}={a[2]:.0f} vs {right}={b[2]:.0f}"
                  + ("" if ok else "  <-- left/right appear mirrored"))
        )
    return out


def check_superior_inferior(series: dict[str, Any]) -> list[Check]:
    """Slice index increases towards the head, because the grid is ordered along +z."""
    organs = _by_organ(series)
    out: list[Check] = []
    for upper, lower, what in (
        ("liver", "urinary_bladder", "liver is superior to bladder"),
        ("adrenal_gland_left", "kidney_left", "left adrenal is superior to left kidney"),
        ("adrenal_gland_right", "kidney_right", "right adrenal is superior to right kidney"),
    ):
        a, b = _centroid(organs, upper), _centroid(organs, lower)
        if a is None or b is None:
            out.append(Check(what, True, f"not checked: {upper} or {lower} absent"))
            continue
        ok = a[0] > b[0]
        out.append(
            Check(what, ok, f"z centroid {upper}={a[0]:.0f} vs {lower}={b[0]:.0f}"
                  + ("" if ok else "  <-- head/foot appear inverted"))
        )
    return out


def check_masses(series: dict[str, Any], *, factor: float = PLAUSIBILITY_FACTOR) -> list[Check]:
    """Solid-organ masses within a wide band of the ICRP 89 reference adult."""
    organs = _by_organ(series)
    out: list[Check] = []
    for organ, reference in ICRP89_REFERENCE_MASS_G.items():
        row = organs.get(organ)
        if row is None:
            out.append(Check(f"{organ} mass plausible", True, "not segmented in this series"))
            continue
        if row.get("truncated"):
            # A partly-scanned organ *should* weigh less than a whole one. Testing it
            # against a whole-organ reference would report the scan's field of view as
            # a segmentation defect.
            out.append(
                Check(f"{organ} mass plausible", True,
                      f"not checked: truncated by the scan boundary ({row['mass_g']:.0f} g "
                      "of an unknown whole)")
            )
            continue
        mass = float(row["mass_g"])
        lo, hi = reference / factor, reference * factor
        ok = lo <= mass <= hi
        out.append(
            Check(
                f"{organ} mass plausible",
                ok,
                f"{mass:.0f} g against ICRP 89 reference {reference:.0f} g "
                f"(band {lo:.0f}-{hi:.0f} g)" + ("" if ok else "  <-- outside the band"),
            )
        )
    return out


def check_modulation(series: dict[str, Any]) -> list[Check]:
    """The organ weights must actually vary, and must bracket the scan mean.

    A weighting in which every organ comes out at 1.000 means I(z) was constant over
    every organ — which for a modulated acquisition means the weighting is not being
    applied, not that the patient is uniform.
    """
    weights = [float(o["relative_weight"]) for o in series.get("organs", [])]
    if len(weights) < 2:
        return [Check("organ weights vary", True, "fewer than two organs")]
    spread = max(weights) - min(weights)
    ok = spread > 1e-3
    checks = [
        Check("organ weights vary", ok,
              f"weights span {min(weights):.3f}-{max(weights):.3f}"
              + ("" if ok else "  <-- either the weighting is not being applied, or the "
                 "tube current is constant across every organ in this series; the second "
                 "is a real acquisition and makes the series uninformative for a "
                 "modulation study rather than faulty"))
    ]
    brackets = min(weights) <= 1.0 <= max(weights)
    checks.append(
        Check(
            "weights bracket the scan mean", brackets,
            f"1.000 {'lies within' if brackets else 'is outside'} "
            f"{min(weights):.3f}-{max(weights):.3f}"
            + ("" if brackets else "  (expected when the scan extends beyond the "
               "abdomen: a chest-abdomen acquisition gives every abdominal organ more "
               "tube current than the scan mean)"),
            advisory=True,
        )
    )
    return checks


def check_series(series: dict[str, Any]) -> dict[str, Any]:
    """Every check for one series, with a single verdict."""
    checks = (
        check_left_right(series)
        + check_superior_inferior(series)
        + check_masses(series)
        + check_modulation(series)
    )
    failed = [c for c in checks if c.is_failure]
    skipped = [c for c in checks if "not checked" in c.detail or "not segmented" in c.detail]
    organs = series.get("organs", [])
    truncated = [o["organ"] for o in organs if o.get("truncated")]
    return {
        "series_instance_uid": series.get("series_instance_uid", ""),
        "vendor": series.get("vendor", ""),
        "passed": not failed,
        "n_failed": len(failed),
        "n_organs": len(organs),
        "truncated_organs": truncated,
        "n_whole_organs": len(organs) - len(truncated),
        # A check that could not run is not a check that passed. Counting these keeps a
        # series with half its organs missing from reading as fully verified.
        "n_not_applicable": len(skipped),
        "checks": [c.to_dict() for c in checks],
        "reference": ICRP89_REFERENCE,
    }
