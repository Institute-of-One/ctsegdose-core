"""Phase 2: run the organ-dose chain over the series recorded in ``data/PROVENANCE.json``.

Usage::

    python tools/run_organ_dose.py --limit 1              # prove the chain on one series
    python tools/run_organ_dose.py                        # the whole sample
    python tools/run_organ_dose.py --coefficients path.json   # ... as far as mGy

Each series is written to ``results/organ_dose/<series-uid>.json`` as it completes, so an
interrupted batch resumes without repeating inference, and a single failure never costs
the run. ``results/organ_dose.json`` aggregates them per vendor.

Without a coefficient table the chain stops at the organ-specific weighted CTDIvol,
organ volume and organ mass -- all of which are computed, none of which is a dose in mGy.
That is a deliberate refusal, not an unimplemented feature: see
``ctsegdose_core/coefficients.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ctsegdose_core import __version__  # noqa: E402
from ctsegdose_core.coefficients import CoefficientTable, load_table  # noqa: E402
from ctsegdose_core.density import DEFAULT_CALIBRATION  # noqa: E402
from ctsegdose_core.pipeline import SeriesResult, run_series, write_result  # noqa: E402
from ctsegdose_core.segment import ABDOMINAL_ORGANS  # noqa: E402


def load_provenance(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run tools/select_and_download.py first -- Phase 2 operates "
            "on the series Phase 1 retrieved."
        )
    return json.loads(path.read_text(encoding="utf-8"))["series"]


def aggregate(results: list[SeriesResult], table: CoefficientTable | None) -> dict[str, Any]:
    by_vendor: dict[str, dict[str, Any]] = {}
    for r in results:
        block = by_vendor.setdefault(
            r.vendor or "unknown",
            {"n_series": 0, "n_completed": 0, "n_organs": 0, "organ_mass_g": 0.0, "errors": []},
        )
        block["n_series"] += 1
        if r.organs:
            block["n_completed"] += 1
            block["n_organs"] += len(r.organs)
            block["organ_mass_g"] += sum(o.mass_g for o in r.organs)
        if r.error:
            block["errors"].append(f"{r.series_uid[-12:]}: {r.error}")
    for block in by_vendor.values():
        block["organ_mass_g"] = round(block["organ_mass_g"], 1)

    organs_seen = Counter(o.organ for r in results for o in r.organs)
    with_dose = sum(1 for r in results for o in r.organs if o.absorbed_dose_mgy is not None)
    return {
        "n_series": len(results),
        "n_series_completed": sum(1 for r in results if r.organs),
        "n_organ_records": sum(len(r.organs) for r in results),
        "n_organ_records_with_absorbed_dose": with_dose,
        "reached": "absorbed organ dose (mGy)" if with_dose else "organ-specific weighted CTDIvol",
        "by_vendor": by_vendor,
        "organs_segmented": dict(organs_seen.most_common()),
        "stage_reached": dict(Counter(r.stage_reached for r in results).most_common()),
        "ctdivol_sources": dict(Counter(r.ctdivol_source for r in results).most_common()),
        "coefficient_table": table.provenance() if table else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provenance", type=Path, default=REPO / "data" / "PROVENANCE.json")
    ap.add_argument("--out", type=Path, default=REPO / "results")
    ap.add_argument("--work-dir", type=Path, default=REPO / "segmentations")
    ap.add_argument("--coefficients", type=Path, default=None, help="organ dose coefficient table")
    ap.add_argument("--limit", type=int, default=0, help="stop after N series (0 = all)")
    ap.add_argument("--per-vendor", type=int, default=0, help="first N series of each vendor")
    ap.add_argument("--vendor", default="", help="restrict to one vendor")
    ap.add_argument("--series", default="", help="restrict to one Series Instance UID")
    ap.add_argument("--organs", nargs="*", default=list(ABDOMINAL_ORGANS))
    ap.add_argument("--full-resolution", action="store_true", help="1.5 mm model instead of 3 mm")
    ap.add_argument("--device", default="", help="cpu | gpu | mps")
    ap.add_argument(
        "--python", default="",
        help="interpreter that runs inference; use the CUDA environment's python here "
             "while the analysis runs in this one",
    )
    ap.add_argument(
        "--tag", default="",
        help="label for this run's outputs and mask directory (default: the model "
             "resolution, so a 3 mm run and a 1.5 mm run never overwrite each other)",
    )
    args = ap.parse_args()

    tag = args.tag or ("1.5mm" if args.full_resolution else "3mm")
    table = load_table(args.coefficients) if args.coefficients else None
    rows = load_provenance(args.provenance)
    if args.vendor:
        rows = [r for r in rows if r["vendor"] == args.vendor]
    if args.series:
        rows = [r for r in rows if r["series_instance_uid"] == args.series]
    if args.per_vendor:
        seen: Counter = Counter()
        picked = []
        for r in rows:
            if seen[r["vendor"]] < args.per_vendor:
                seen[r["vendor"]] += 1
                picked.append(r)
        rows = picked
    if args.limit:
        rows = rows[: args.limit]

    started = datetime.now(UTC).isoformat(timespec="seconds")
    per_series_dir = args.out / "organ_dose" / tag
    results: list[SeriesResult] = []
    print(f"tag={tag}  model={'1.5 mm' if args.full_resolution else '3 mm'}  "
          f"device={args.device or 'auto'}  interpreter={args.python or 'this one'}")

    for i, row in enumerate(rows, 1):
        directory = REPO / row["local_path"]
        uid = row["series_instance_uid"]
        print(f"[{i:2d}/{len(rows)}] {row['vendor']:14s} {row['collection']:26.26s} {uid[-12:]}")
        t0 = time.monotonic()
        result = run_series(
            directory,
            work_dir=args.work_dir / uid / tag,
            series_uid=uid,
            metadata=row,
            table=table,
            calibration=DEFAULT_CALIBRATION,
            organs=tuple(args.organs),
            fast=not args.full_resolution,
            device=args.device,
            python_executable=args.python,
        )
        results.append(result)
        write_result(per_series_dir / f"{uid}.json", result, table)
        if result.error:
            print(f"          -> {result.stage_reached or 'failed'}: {result.error}")
        else:
            top = max(result.organs, key=lambda o: o.relative_weight)
            ctdivol = (
                "none" if result.ctdivol_mgy is None else f"{result.ctdivol_mgy:.2f} mGy"
            )
            print(
                f"          -> {len(result.organs)} organs, CTDIvol {ctdivol} "
                f"({result.ctdivol_source}), highest weight {top.organ} "
                f"w={top.relative_weight:.3f}, {time.monotonic() - t0:.0f}s"
            )

    payload = {
        "provenance": {
            "layer": "2 - organ segmentation, HU-derived mass, organ-specific weighted CTDIvol"
                     + (", absorbed organ dose" if table else ""),
            "tag": tag,
            "model_resolution": "1.5 mm" if args.full_resolution else "3 mm",
            "generated_by": "tools/run_organ_dose.py",
            "ctsegdose_core_version": __version__,
            "started_utc": started,
            "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "segmentation": "TotalSegmentator (Apache-2.0), inference only, separate process",
            "density_calibration": DEFAULT_CALIBRATION.to_dict(),
            "dose_engine": "ctdose-core (IORN-004) for CTDIvol and the organ weighting",
        },
        "summary": aggregate(results, table),
        "series": [r.to_dict(table) for r in results],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"organ_dose_{tag}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    s = payload["summary"]
    print("\n-- organ dose --")
    print(f"  {s['n_series_completed']}/{s['n_series']} series completed, "
          f"{s['n_organ_records']} organ records, reached: {s['reached']}")
    for vendor, block in s["by_vendor"].items():
        print(
            f"  {vendor:14s} {block['n_completed']:2d}/{block['n_series']:2d} series  "
            f"{block['n_organs']:3d} organs  {block['organ_mass_g'] / 1000:6.2f} kg segmented mass"
        )
    for vendor, block in s["by_vendor"].items():
        for err in block["errors"]:
            print(f"    ! {vendor}: {err}")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
