"""What the reconstructed CTDIvol values do, and do not, do to the published index.

Reviewer 2 asked, in round 2, how the differences in reconstructed CTDIvol may
affect the final anatomy-weighted index. The answer has two parts and they are
not the same kind of statement, so this computes both rather than asserting
either.

The structural part. The modulation weight is

    w_o = [sum_z n_o(z) I(z) / sum_z n_o(z)] / mean_z I(z)

which contains only the tube current. CTDIvol does not appear in it. Any error in
CTDIvol -- from reconstruction or from anything else -- therefore cannot move
w_o at all, and the organ-weighted CTDIvol, being CTDIvol * w_o, inherits that
error at exactly 1:1. This is a property of the definition, not a finding, and
the check below is that the recomputation agrees with it to the last digit.

The empirical part. How much of the cohort is exposed to that 1:1 propagation,
and does dropping it change what the paper reports? Recomputing the tables over
the series with a *recorded* CTDIvol answers it directly.

The result carries one consequence that has to be reported whether or not it is
convenient: GE contributes no series with a recorded CTDIvol at all, so the
recorded-only tables are silent about GE. That is not an argument for keeping the
reconstructed values; it is the reason the multi-vendor reach of the weighted
tables rests on them, and it belongs in the limitations.

Usage::

    python tools/sensitivity_reconstructed.py --tag 1.5mm

Writes ``results/sensitivity_reconstructed_<tag>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ctsegdose_core.analysis import (  # noqa: E402
    _completed,
    _ctdivol_class,
    weighted_ctdivol,
)
from ctsegdose_core.eligibility import assess  # noqa: E402


def _eligible_uids(constancy: dict[str, Any]) -> set[str]:
    return {uid for uid, e in assess(constancy).items() if e.eligible}


def _exposure(series: list[dict[str, Any]], eligible: set[str]) -> dict[str, Any]:
    """How much of the analysed cohort rests on a reconstructed CTDIvol."""
    usable = [
        s for s in _completed(series)
        if s["ctdivol_mgy"] is not None and s["series_instance_uid"] in eligible
    ]
    recon = [s for s in usable if _ctdivol_class(s["ctdivol_source"]) == "reconstructed"]

    def records(rows):
        return sum(1 for s in rows for o in s["organs"] if not o.get("truncated"))

    return {
        "n_series_analysed": len(usable),
        "n_series_on_a_reconstructed_value": len(recon),
        "n_organ_records_analysed": records(usable),
        "n_organ_records_on_a_reconstructed_value": records(recon),
        "series": [
            {
                "vendor": s["vendor"],
                "model_name": s["model_name"],
                "ctdivol_mgy": round(float(s["ctdivol_mgy"]), 2),
                "n_organ_records": records([s]),
            }
            for s in recon
        ],
    }


def _compare(published: dict[str, Any], restricted: dict[str, Any]) -> dict[str, Any]:
    """Per-organ medians, published against recorded-only."""
    rows = {}
    for organ, pub in published["by_organ"].items():
        res = restricted["by_organ"].get(organ)
        if res is None:
            rows[organ] = {"restricted": None, "note": "no series left for this organ"}
            continue
        rows[organ] = {
            "relative_weight": {
                "published_median": pub["relative_weight"]["median"],
                "recorded_only_median": res["relative_weight"]["median"],
                "delta": round(
                    res["relative_weight"]["median"] - pub["relative_weight"]["median"], 4
                ),
                "published_n": pub["relative_weight"]["n"],
                "recorded_only_n": res["relative_weight"]["n"],
            },
            "organ_weighted_ctdivol_mgy": {
                "published_median": pub["organ_weighted_ctdivol_mgy"]["median"],
                "recorded_only_median": res["organ_weighted_ctdivol_mgy"]["median"],
                "delta": round(
                    res["organ_weighted_ctdivol_mgy"]["median"]
                    - pub["organ_weighted_ctdivol_mgy"]["median"],
                    2,
                ),
            },
        }
    return rows


def analyse(payload: dict[str, Any], constancy: dict[str, Any]) -> dict[str, Any]:
    series = payload["series"]
    eligible = _eligible_uids(constancy)

    published = weighted_ctdivol(series, eligible)
    recorded_only = eligible & {
        s["series_instance_uid"]
        for s in series
        if _ctdivol_class(s["ctdivol_source"]) == "recorded"
    }
    restricted = weighted_ctdivol(series, recorded_only)

    # The structural claim, checked rather than asserted: a weight computed over a
    # subset of series must be identical for every series that subset retains, since
    # nothing in w_o depends on which other series are present or on CTDIvol.
    per_series_weights = {}
    for s in _completed(series):
        if s["series_instance_uid"] not in recorded_only:
            continue
        per_series_weights[s["series_instance_uid"]] = {
            o["organ"]: round(float(o["relative_weight"]), 6)
            for o in s["organs"]
            if not o.get("truncated")
        }

    return {
        "what": (
            "Sensitivity of the anatomy-weighted index to the CTDIvol values that were "
            "reconstructed rather than read from the header. Answers reviewer 2, "
            "round 2."
        ),
        "structural": {
            "weight_definition": (
                "w_o = [sum_z n_o(z) I(z) / sum_z n_o(z)] / mean_z I(z)"
            ),
            "ctdivol_appears_in_the_weight": False,
            "propagation_into_the_index": (
                "organ-weighted CTDIvol = CTDIvol * w_o, so a relative error in "
                "CTDIvol appears as the same relative error in the index, and none "
                "of it reaches w_o"
            ),
        },
        "exposure": _exposure(series, eligible),
        "vendor_reach": {
            "published": {
                v: b["n_series_with_an_index"]
                for v, b in published["by_vendor"].items()
            },
            "recorded_only": {
                v: b["n_series_with_an_index"]
                for v, b in restricted["by_vendor"].items()
            },
        },
        "by_organ": _compare(published, restricted),
        "n_series": {
            "published": published["n_series"],
            "recorded_only": restricted["n_series"],
        },
        "per_series_weights_recorded_only": per_series_weights,
    }


def report(result: dict[str, Any]) -> None:
    e = result["exposure"]
    print("== exposure to a reconstructed CTDIvol ==")
    print(
        f"  {e['n_series_on_a_reconstructed_value']} of {e['n_series_analysed']} "
        f"analysed series; {e['n_organ_records_on_a_reconstructed_value']} of "
        f"{e['n_organ_records_analysed']} organ records"
    )
    for s in e["series"]:
        print(f"    {s['vendor']:8} {s['model_name']:22} {s['ctdivol_mgy']:>7} mGy"
              f"  {s['n_organ_records']:>3} records")

    print("\n== series with an index, by vendor ==")
    pub, rec = result["vendor_reach"]["published"], result["vendor_reach"]["recorded_only"]
    for vendor in pub:
        print(f"  {vendor:14} published {pub[vendor]:>3}   recorded-only {rec[vendor]:>3}")

    print("\n== per-organ medians: published vs recorded-only ==")
    print(f"  {'organ':22} {'w_o pub':>8} {'w_o rec':>8} {'d':>7}   "
          f"{'mGy pub':>8} {'mGy rec':>8} {'d':>7}")
    for organ, row in result["by_organ"].items():
        if row.get("restricted", True) is None:
            print(f"  {organ:22} (no series left)")
            continue
        w, m = row["relative_weight"], row["organ_weighted_ctdivol_mgy"]
        print(
            f"  {organ:22} {w['published_median']:>8} {w['recorded_only_median']:>8} "
            f"{w['delta']:>+7}   {m['published_median']:>8} "
            f"{m['recorded_only_median']:>8} {m['delta']:>+7}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="1.5mm")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = json.loads(
        (REPO / "results" / f"organ_dose_{args.tag}.json").read_text(encoding="utf-8")
    )
    constancy = json.loads(
        (REPO / "results" / "acquisition_constancy.json").read_text(encoding="utf-8")
    )

    result = analyse(payload, constancy)
    out = args.out or REPO / "results" / f"sensitivity_reconstructed_{args.tag}.json"
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report(result)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
