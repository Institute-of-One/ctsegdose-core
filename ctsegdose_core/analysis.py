"""The tables the manuscript quotes, derived from the per-series records and nothing else.

Every figure in the paper comes from here, and every number here is recomputed from
``results/organ_dose_<tag>.json`` by :mod:`tests.test_analysis_integrity`. Nothing is
typed into a table by hand, so a quoted figure cannot drift from the data behind it.

The statistics are deliberately plain. Ten series per vendor supports a median and an
interquartile range; it does not support a confident distributional claim, and dressing
it as one would be worse than saying so. Where a comparison is made -- dose-index
availability across vendors -- it is a count, and the test applied to it is exact.
"""

from __future__ import annotations

import statistics as st
from collections import Counter
from dataclasses import dataclass
from typing import Any

VENDORS: tuple[str, ...] = ("GE", "Siemens", "Canon/Toshiba", "Philips")

#: ICRP Publication 89 reference masses for the adult male, in grams. Used as the
#: comparison point for the segmented solid organs -- not as a ground truth for these
#: patients, who are an oncology cohort and are not the reference adult, but as the
#: published anchor a reader can check the method against.
ICRP89_REFERENCE_MASS_G: dict[str, float] = {
    "liver": 1800.0,
    "spleen": 150.0,
    "kidney_left": 155.0,
    "kidney_right": 155.0,
    "pancreas": 140.0,
}
ICRP89_CITATION = (
    "ICRP Publication 89 (2002), Basic Anatomical and Physiological Data for Use in "
    "Radiological Protection: Reference Values. Adult male reference organ masses."
)

#: Below this peak-to-peak spread, the organ weights of a series are indistinguishable:
#: the tube current does not vary across the organs, so the modulation weighting has
#: nothing to weight and the series cannot inform an organ-level modulation result.
FLAT_WEIGHT_THRESHOLD = 0.02


def _ctdivol_class(source: str) -> str:
    if "recorded" in source:
        return "recorded"
    if "reconstructed" in source:
        return "reconstructed"
    return "unrecoverable"


@dataclass
class Distribution:
    """A robust summary of one set of values."""

    n: int
    median: float
    p25: float
    p75: float
    minimum: float
    maximum: float

    @classmethod
    def of(cls, values: list[float]) -> Distribution | None:
        v = sorted(float(x) for x in values)
        if not v:
            return None
        return cls(
            n=len(v),
            median=st.median(v),
            p25=v[len(v) // 4],
            p75=v[(3 * len(v)) // 4],
            minimum=v[0],
            maximum=v[-1],
        )

    def to_dict(self, digits: int = 1) -> dict[str, Any]:
        return {
            "n": self.n,
            "median": round(self.median, digits),
            "p25": round(self.p25, digits),
            "p75": round(self.p75, digits),
            "min": round(self.minimum, digits),
            "max": round(self.maximum, digits),
        }


def _completed(series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [s for s in series if s.get("organs")]


# --- (1) dose-index availability, the headline ------------------------------------------


def availability(series: list[dict[str, Any]]) -> dict[str, Any]:
    """Where each series' CTDIvol came from, by vendor.

    Three outcomes, and the third is the point: a series carrying no recorded CTDIvol
    whose scanner is outside the open coefficient database has no dose index at all, so
    no organ-level index can be formed from it however good the segmentation is.
    """
    by_vendor: dict[str, dict[str, Any]] = {}
    for vendor in VENDORS:
        rows = [s for s in series if s["vendor"] == vendor]
        counts = Counter(_ctdivol_class(s["ctdivol_source"]) for s in rows)
        n = len(rows)
        by_vendor[vendor] = {
            "n_series": n,
            "recorded": counts["recorded"],
            "reconstructed": counts["reconstructed"],
            "unrecoverable": counts["unrecoverable"],
            "recorded_rate": round(counts["recorded"] / n, 3) if n else None,
            "index_available_rate": (
                round((counts["recorded"] + counts["reconstructed"]) / n, 3) if n else None
            ),
        }
    overall = Counter(_ctdivol_class(s["ctdivol_source"]) for s in series)
    return {
        "definition": {
            "recorded": "CTDIvol read from the image header (0018,9345)",
            "reconstructed": "rebuilt from acquisition physics via the open MIRDct table",
            "unrecoverable": "neither available; no organ-level dose index can be formed",
        },
        "by_vendor": by_vendor,
        "overall": {
            "n_series": len(series),
            "recorded": overall["recorded"],
            "reconstructed": overall["reconstructed"],
            "unrecoverable": overall["unrecoverable"],
        },
        "ge_vs_rest_recorded": _fisher_ge_vs_rest(series),
    }


def _fisher_ge_vs_rest(series: list[dict[str, Any]]) -> dict[str, Any]:
    """Exact test of the recorded-CTDIvol difference between GE and the other vendors.

    A 2x2 count with ten per group is exactly the case Fisher's test exists for; a
    chi-square approximation on cells this small would be the wrong instrument.
    """
    ge = [s for s in series if s["vendor"] == "GE"]
    rest = [s for s in series if s["vendor"] != "GE"]
    a = sum(1 for s in ge if _ctdivol_class(s["ctdivol_source"]) == "recorded")
    b = len(ge) - a
    c = sum(1 for s in rest if _ctdivol_class(s["ctdivol_source"]) == "recorded")
    d = len(rest) - c
    result: dict[str, Any] = {
        "table": {"ge_recorded": a, "ge_not_recorded": b, "other_recorded": c, "other_not_recorded": d},
        "test": "Fisher exact, two-sided",
    }
    try:
        from scipy.stats import fisher_exact

        _, p = fisher_exact([[a, b], [c, d]])
        result["p_value"] = float(p)
    except Exception:  # pragma: no cover - scipy is an extra
        result["p_value"] = None
        result["note"] = "scipy not available; the counts stand on their own"
    return result


# --- (2) segmented organ mass against a published reference -----------------------------


def organ_mass(series: list[dict[str, Any]]) -> dict[str, Any]:
    """Segmented organ mass, overall and by vendor, against ICRP 89.

    Truncated organs are excluded throughout: an organ the scan cuts through has the
    mass of its scanned part, and comparing that with a whole-organ reference would
    measure the field of view rather than the segmentation.
    """
    records = [
        (s["vendor"], o["organ"], float(o["mass_g"]), float(o["volume_cm3"]))
        for s in _completed(series)
        for o in s["organs"]
        if not o.get("truncated")
    ]

    overall: dict[str, Any] = {}
    for organ in sorted({r[1] for r in records}):
        masses = [r[2] for r in records if r[1] == organ]
        volumes = [r[3] for r in records if r[1] == organ]
        block: dict[str, Any] = {
            "mass_g": Distribution.of(masses).to_dict(),
            "volume_cm3": Distribution.of(volumes).to_dict(),
        }
        reference = ICRP89_REFERENCE_MASS_G.get(organ)
        if reference:
            block["icrp89_reference_mass_g"] = reference
            block["median_over_reference"] = round(st.median(masses) / reference, 3)
        overall[organ] = block

    by_vendor: dict[str, dict[str, Any]] = {}
    for vendor in VENDORS:
        by_vendor[vendor] = {}
        for organ in sorted(ICRP89_REFERENCE_MASS_G):
            masses = [r[2] for r in records if r[0] == vendor and r[1] == organ]
            dist = Distribution.of(masses)
            if dist:
                entry = dist.to_dict()
                entry["median_over_reference"] = round(
                    dist.median / ICRP89_REFERENCE_MASS_G[organ], 3
                )
                by_vendor[vendor][organ] = entry

    return {
        "reference": ICRP89_CITATION,
        "excludes": "organs truncated by the scan boundary",
        "overall": overall,
        "solid_organs_by_vendor": by_vendor,
    }


# --- (3) the organ-level dose index -----------------------------------------------------


def weighted_ctdivol(series: list[dict[str, Any]]) -> dict[str, Any]:
    """Organ-specific weighted CTDIvol, and the modulation weight that produced it.

    The weight is the transferable quantity: it is dimensionless, it is what the
    modulation contributes, and unlike the weighted CTDIvol it does not depend on the
    scanner's own output. Both are reported, per organ and per vendor.
    """
    usable = [s for s in _completed(series) if s["ctdivol_mgy"] is not None]
    weights: dict[str, list[float]] = {}
    indices: dict[str, list[float]] = {}
    for s in usable:
        for o in s["organs"]:
            if o.get("truncated"):
                continue
            weights.setdefault(o["organ"], []).append(float(o["relative_weight"]))
            indices.setdefault(o["organ"], []).append(float(o["organ_weighted_ctdivol_mgy"]))

    per_organ = {
        organ: {
            "relative_weight": Distribution.of(w).to_dict(3),
            "organ_weighted_ctdivol_mgy": Distribution.of(indices[organ]).to_dict(2),
        }
        for organ, w in sorted(weights.items())
    }

    by_vendor: dict[str, Any] = {}
    for vendor in VENDORS:
        rows = [s for s in usable if s["vendor"] == vendor]
        w = [
            float(o["relative_weight"])
            for s in rows
            for o in s["organs"]
            if not o.get("truncated")
        ]
        dist = Distribution.of(w)
        by_vendor[vendor] = {
            "n_series_with_an_index": len(rows),
            "relative_weight": dist.to_dict(3) if dist else None,
        }

    return {
        "definition": (
            "w_o = [sum_z n_o(z) I(z) / sum_z n_o(z)] / mean_z I(z); "
            "organ-weighted CTDIvol = CTDIvol * w_o"
        ),
        "excludes": "truncated organs, and series with no CTDIvol",
        "n_series": len(usable),
        "by_organ": per_organ,
        "by_vendor": by_vendor,
    }


# --- (4) what limits an organ-level modulation study ------------------------------------


def study_limits(series: list[dict[str, Any]]) -> dict[str, Any]:
    """Truncation and flat weighting: the two ways a series fails to inform this question.

    Neither is a defect in the data or the method. A scan that ends mid-abdomen has
    truncated organs; a scan whose tube current happens not to vary across the abdomen
    has nothing for a modulation weighting to express. Both have to be counted, because
    both silently reduce what an organ-level study is actually measuring.
    """
    completed = _completed(series)

    truncation: dict[str, Any] = {}
    for vendor in VENDORS:
        rows = [s for s in completed if s["vendor"] == vendor]
        organs = [o for s in rows for o in s["organs"]]
        cut = [o for o in organs if o.get("truncated")]
        truncation[vendor] = {
            "n_organ_records": len(organs),
            "n_truncated": len(cut),
            "truncated_rate": round(len(cut) / len(organs), 3) if organs else None,
            "organs_most_often_cut": [o for o, _ in Counter(x["organ"] for x in cut).most_common(3)],
        }

    spreads = []
    flat = []
    for s in completed:
        w = [float(o["relative_weight"]) for o in s["organs"]]
        if len(w) < 2:
            continue
        spread = max(w) - min(w)
        spreads.append({"vendor": s["vendor"], "series": s["series_instance_uid"], "spread": spread})
        if spread < FLAT_WEIGHT_THRESHOLD:
            flat.append({"vendor": s["vendor"], "series": s["series_instance_uid"],
                         "spread": round(spread, 4)})

    by_vendor_spread = {}
    for vendor in VENDORS:
        v = [x["spread"] for x in spreads if x["vendor"] == vendor]
        dist = Distribution.of(v)
        by_vendor_spread[vendor] = dist.to_dict(3) if dist else None

    return {
        "truncation": {
            "definition": "the organ's mask reaches the first or last slice of the series",
            "by_vendor": truncation,
        },
        "flat_weighting": {
            "definition": (
                f"peak-to-peak spread of the organ weights below {FLAT_WEIGHT_THRESHOLD}; "
                "the tube current does not vary across this series' organs"
            ),
            "n_series": len(flat),
            "series": flat,
            "weight_spread_by_vendor": by_vendor_spread,
        },
    }


# --- assembly ----------------------------------------------------------------------------


def build(payload: dict[str, Any]) -> dict[str, Any]:
    """Every table, from one ``organ_dose_<tag>.json`` payload."""
    series = payload["series"]
    completed = _completed(series)
    return {
        "cohort": {
            "n_series": len(series),
            "n_series_completed": len(completed),
            "n_organ_records": sum(len(s["organs"]) for s in completed),
            "series_by_vendor": {v: sum(1 for s in series if s["vendor"] == v) for v in VENDORS},
            "organs_segmented": dict(
                Counter(o["organ"] for s in completed for o in s["organs"]).most_common()
            ),
            "collections": sorted({s["collection"] for s in series if s.get("collection")}),
            "scanner_models": sorted({s["model_name"] for s in series if s.get("model_name")}),
        },
        "availability": availability(series),
        "organ_mass": organ_mass(series),
        "weighted_ctdivol": weighted_ctdivol(series),
        "study_limits": study_limits(series),
    }
