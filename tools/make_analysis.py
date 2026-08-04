"""Freeze the per-series records into the tables the manuscript quotes.

Usage::

    python tools/make_analysis.py --tag 1.5mm

Reads ``results/organ_dose_<tag>.json``, writes ``results/analysis_<tag>.json`` and the
flat CSVs a reader or a figure script needs. Nothing here is entered by hand:
``tests/test_analysis_integrity.py`` recomputes every published figure from the same
per-series records and fails if they disagree.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ctsegdose_core import __version__  # noqa: E402
from ctsegdose_core.analysis import build  # noqa: E402

ORGAN_COLUMNS = [
    "vendor", "collection", "model_name", "series_instance_uid", "organ",
    "volume_cm3", "mass_g", "mean_density_g_cm3", "mean_hu", "truncated",
    "relative_weight", "mean_tube_current_ma", "organ_weighted_ctdivol_mgy",
    "water_equivalent_diameter_cm", "ctdivol_mgy", "ctdivol_source",
]

SERIES_COLUMNS = [
    "vendor", "collection", "model_name", "series_instance_uid", "n_slices", "kvp",
    "ctdivol_mgy", "ctdivol_source", "scan_mean_tube_current_ma", "n_organs",
    "n_truncated", "weight_spread",
]


def organ_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for s in payload["series"]:
        for o in s.get("organs", []):
            rows.append({
                "vendor": s["vendor"], "collection": s["collection"],
                "model_name": s["model_name"], "series_instance_uid": s["series_instance_uid"],
                "ctdivol_mgy": s["ctdivol_mgy"], "ctdivol_source": s["ctdivol_source"],
                **{k: o.get(k) for k in (
                    "organ", "volume_cm3", "mass_g", "mean_density_g_cm3", "mean_hu",
                    "truncated", "relative_weight", "mean_tube_current_ma",
                    "organ_weighted_ctdivol_mgy", "water_equivalent_diameter_cm")},
            })
    return rows


def series_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for s in payload["series"]:
        organs = s.get("organs", [])
        weights = [o["relative_weight"] for o in organs]
        rows.append({
            **{k: s.get(k) for k in (
                "vendor", "collection", "model_name", "series_instance_uid", "n_slices",
                "kvp", "ctdivol_mgy", "ctdivol_source", "scan_mean_tube_current_ma")},
            "n_organs": len(organs),
            "n_truncated": sum(1 for o in organs if o.get("truncated")),
            "weight_spread": round(max(weights) - min(weights), 4) if len(weights) > 1 else None,
        })
    return rows


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}")


def report(tables: dict[str, Any]) -> None:
    a = tables["availability"]
    print("\n== dose-index availability by vendor ==")
    print(f"  {'vendor':14s} {'n':>3s} {'recorded':>9s} {'reconstr':>9s} {'unrecov':>8s}")
    for vendor, b in a["by_vendor"].items():
        print(f"  {vendor:14s} {b['n_series']:3d} {b['recorded']:9d} "
              f"{b['reconstructed']:9d} {b['unrecoverable']:8d}")
    p = a["ge_vs_rest_recorded"].get("p_value")
    print(f"  GE vs rest, recorded CTDIvol: Fisher exact p = "
          f"{'n/a' if p is None else format(p, '.2g')}")

    print("\n== segmented organ mass vs ICRP 89 (whole organs only) ==")
    print(f"  {'organ':14s} {'n':>3s} {'median g':>9s} {'IQR':>17s} {'ICRP':>6s} {'ratio':>6s}")
    for organ, b in tables["organ_mass"]["overall"].items():
        if "icrp89_reference_mass_g" not in b:
            continue
        m = b["mass_g"]
        print(f"  {organ:14s} {m['n']:3d} {m['median']:9.0f} "
              f"{m['p25']:7.0f}-{m['p75']:<9.0f} {b['icrp89_reference_mass_g']:6.0f} "
              f"{b['median_over_reference']:6.2f}")

    print("\n== organ modulation weight, all vendors ==")
    for organ, b in tables["weighted_ctdivol"]["by_organ"].items():
        w = b["relative_weight"]
        print(f"  {organ:22s} n={w['n']:3d}  median w={w['median']:.3f}  "
              f"range {w['min']:.3f}-{w['max']:.3f}")

    limits = tables["study_limits"]
    print("\n== what limits the study ==")
    for vendor, b in limits["truncation"]["by_vendor"].items():
        print(f"  {vendor:14s} truncated {b['n_truncated']:3d}/{b['n_organ_records']:3d} "
              f"({(b['truncated_rate'] or 0) * 100:4.1f}%)  most cut: {b['organs_most_often_cut']}")
    print(f"  flat-weighted series: {limits['flat_weighting']['n_series']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="1.5mm")
    ap.add_argument("--results", type=Path, default=REPO / "results")
    args = ap.parse_args()

    source = args.results / f"organ_dose_{args.tag}.json"
    if not source.exists():
        raise SystemExit(f"{source} not found; run tools/run_organ_dose.py first")
    payload = json.loads(source.read_text(encoding="utf-8"))

    tables = build(payload)
    tables["provenance"] = {
        "generated_by": "tools/make_analysis.py",
        "ctsegdose_core_version": __version__,
        "source": str(source.relative_to(REPO)).replace("\\", "/"),
        "source_provenance": payload.get("provenance", {}),
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": (
            "Every value here is recomputed from the per-series records by "
            "tests/test_analysis_integrity.py; none is entered by hand."
        ),
    }

    out = args.results / f"analysis_{args.tag}.json"
    out.write_text(json.dumps(tables, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  wrote {out}")
    write_csv(args.results / f"organs_{args.tag}.csv", ORGAN_COLUMNS, organ_rows(payload))
    write_csv(args.results / f"series_{args.tag}.csv", SERIES_COLUMNS, series_rows(payload))
    report(tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
