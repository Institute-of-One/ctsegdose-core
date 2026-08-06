"""Check the assumption the anatomy-weighted index rests on, series by series.

The weight in Equation (1) treats the recorded tube current as a proxy for the
longitudinal variation in scanner output. That holds only if the other quantities
governing output are fixed within a series: change the tube voltage, the rotation time,
the pitch or the collimation part-way through, and the current alone no longer describes
how the output varied along the patient.

This tool reads every slice header of every retained series and reports, per attribute,
whether it is constant, varies, or is absent from the archived headers. Absence is
reported as absence -- an attribute that was never written cannot be verified constant,
and saying otherwise would be the same class of error the manuscript exists to avoid.

Writes ``results/acquisition_constancy.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pydicom  # noqa: E402

from ctsegdose_core import __version__  # noqa: E402

#: The attributes that, together with tube current, determine scanner output. Keyed by
#: the name used in the manuscript.
ATTRIBUTES: dict[str, tuple[int, int]] = {
    "tube_voltage_kvp": (0x0018, 0x0060),
    "exposure_time_ms": (0x0018, 0x1150),
    "rotation_time_s": (0x0018, 0x9305),
    "spiral_pitch_factor": (0x0018, 0x9311),
    "total_collimation_width_mm": (0x0018, 0x9307),
    "single_collimation_width_mm": (0x0018, 0x9306),
}
#: Reported alongside, because a series mixing acquisition types is not one acquisition.
CONTEXT: dict[str, tuple[int, int]] = {
    "image_type": (0x0008, 0x0008),
    "convolution_kernel": (0x0018, 0x1210),
    "protocol_name": (0x0018, 0x1030),
}

#: Floating-point attributes are compared after rounding: reconstruction writes pitch and
#: collimation as decimal strings, and a difference in the last digit is a formatting
#: artefact rather than a change in the acquisition.
ROUNDING = 4


def _value(ds: pydicom.Dataset, tag: tuple[int, int]) -> Any:
    if tag not in ds:
        return None
    raw = ds[tag].value
    if raw is None or str(raw).strip() == "":
        return None
    if isinstance(raw, (list, tuple)) or type(raw).__name__ == "MultiValue":
        return "\\".join(str(v) for v in raw)
    try:
        return round(float(raw), ROUNDING)
    except (TypeError, ValueError):
        return str(raw).strip()


def inspect_series(directory: Path, series_uid: str) -> dict[str, Any]:
    from ctsegdose_core.pipeline import load_on_uniform_grid

    series, grid = load_on_uniform_grid(directory, series_uid)
    observed: dict[str, Counter] = {k: Counter() for k in {**ATTRIBUTES, **CONTEXT}}
    for path in series.files:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        for name, tag in {**ATTRIBUTES, **CONTEXT}.items():
            observed[name][_value(ds, tag)] += 1

    report: dict[str, Any] = {"n_slices": len(series.files)}
    for name in {**ATTRIBUTES, **CONTEXT}:
        counts = observed[name]
        values = [v for v in counts if v is not None]
        n_absent = counts.get(None, 0)
        if not values:
            status = "absent"
        elif n_absent:
            status = "partially_absent"
        elif len(values) == 1:
            status = "constant"
        else:
            status = "varies"
        entry: dict[str, Any] = {
            "status": status,
            "distinct_values": sorted(str(v) for v in values)[:6],
            "n_distinct": len(values),
            "n_slices_absent": n_absent,
        }
        # The magnitude of a variation decides whether it matters, so it is recorded
        # rather than left to be inferred from a truncated list of distinct values.
        numeric = [v for v in values if isinstance(v, float)]
        if len(numeric) > 1:
            lo, hi = min(numeric), max(numeric)
            entry["min"] = lo
            entry["max"] = hi
            entry["relative_spread"] = (hi - lo) / lo if lo else None
        report[name] = entry
    return report


def summarise(per_series: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in {**ATTRIBUTES, **CONTEXT}:
        statuses = Counter(r[name]["status"] for r in per_series.values())
        out[name] = {
            "constant": statuses.get("constant", 0),
            "varies": statuses.get("varies", 0),
            "partially_absent": statuses.get("partially_absent", 0),
            "absent": statuses.get("absent", 0),
            "series_that_vary": sorted(
                uid for uid, r in per_series.items() if r[name]["status"] == "varies"
            ),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provenance", type=Path, default=REPO / "data" / "PROVENANCE.json")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "acquisition_constancy.json")
    args = ap.parse_args()

    rows = json.loads(args.provenance.read_text(encoding="utf-8"))["series"]
    per_series: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(rows, 1):
        uid = row["series_instance_uid"]
        print(f"[{i:2d}/{len(rows)}] {row['vendor']:14s} {uid[-12:]}")
        per_series[uid] = {"vendor": row["vendor"], **inspect_series(REPO / row["local_path"], uid)}

    payload = {
        "provenance": {
            "generated_by": "tools/verify_acquisition_constancy.py",
            "ctsegdose_core_version": __version__,
            "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "question": (
                "Within each series, are the acquisition parameters other than tube "
                "current constant, so that longitudinal variation in scanner output is "
                "proportional to the recorded tube current?"
            ),
            "note": (
                "An attribute absent from the archived headers is reported as absent, "
                "not as constant: it cannot be verified either way."
            ),
        },
        "summary": summarise(per_series),
        "by_series": per_series,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{'attribute':30s} {'const':>6s} {'varies':>7s} {'partial':>8s} {'absent':>7s}")
    for name, block in payload["summary"].items():
        print(f"{name:30s} {block['constant']:6d} {block['varies']:7d} "
              f"{block['partially_absent']:8d} {block['absent']:7d}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
