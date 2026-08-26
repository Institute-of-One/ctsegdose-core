"""Check the physics reconstruction of CTDIvol against the value the header records.

Usage::

    python tools/validate_reconstruction.py

A reviewer asked how the reconstructed CTDIvol values were validated, "particularly
by comparing them with recorded CTDIvol values where both were available". In this
cohort no series has both, and that is by construction rather than by accident:
``resolve_ctdivol`` reconstructs only where the header carries no usable value, so
the two populations are disjoint and the comparison the reviewer wants does not
exist in the pipeline's own output.

It can be made to exist. Every series that carries a recorded CTDIvol also carries
the acquisition parameters the reconstruction needs, so the reconstruction can be run
on those series and compared against the value the scanner wrote. That is a fairer
test than the one asked for: it measures the reconstruction on exactly the series
where the truth is known, and it is blind in the sense that the recorded value plays
no part in producing the reconstructed one.

What it cannot do is validate the reconstruction on the series where it is actually
used. Those are the series whose headers omit the value, and if the omission
correlates with anything that also moves the reconstruction, the error measured here
is optimistic. That limitation is recorded in the output rather than left implicit.

Writes ``results/reconstruction_validation.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PROVENANCE = REPO / "data" / "PROVENANCE.json"
SERIES_RESULTS = REPO / "results" / "organ_dose_1.5mm.json"
DEFAULT_OUT = REPO / "results" / "reconstruction_validation.json"


def _recorded_series() -> list[dict[str, Any]]:
    """The series whose CTDIvol came from the header, with their local paths."""
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    by_uid = {row["series_instance_uid"]: row for row in provenance["series"]}

    results = json.loads(SERIES_RESULTS.read_text(encoding="utf-8"))
    rows = results if isinstance(results, list) else results.get("series", [])

    wanted = []
    for row in rows:
        source = str(row.get("ctdivol_source", ""))
        value = row.get("ctdivol_mgy")
        if not source.startswith("recorded") or not value:
            continue
        entry = by_uid.get(row.get("series_uid") or row.get("series_instance_uid"))
        if entry is None:
            continue
        wanted.append(
            {
                "series_instance_uid": entry["series_instance_uid"],
                "vendor": entry["vendor"],
                "manufacturer": entry["manufacturer"],
                "model_name": entry["model_name"],
                "collection": entry["collection"],
                "local_path": entry["local_path"],
                "recorded_mgy": float(value),
            }
        )
    return wanted


def _reconstruct(entry: dict[str, Any]) -> tuple[float | None, str]:
    """Run the open reconstruction on one series, ignoring the recorded value."""
    from ctdose_core.ctdi_table import CoefficientNotAvailable, resolve_model
    from ctdose_core.metrics import acquisition_from_series, estimate_ctdivol_open

    from ctsegdose_core.pipeline import load_on_uniform_grid

    directory = Path(entry["local_path"])
    if not directory.is_absolute():
        directory = REPO / directory
    if not directory.exists():
        return None, f"series directory missing: {directory}"

    try:
        series, _grid = load_on_uniform_grid(
            directory, entry["series_instance_uid"]
        )
    except Exception as exc:  # noqa: BLE001 - any load failure is a skip, recorded
        return None, f"load failed: {type(exc).__name__}: {exc}"

    try:
        vendor, model = resolve_model(entry["manufacturer"], entry["model_name"])
        acquisition = acquisition_from_series(series)
        quantity = estimate_ctdivol_open(
            acquisition, vendor=vendor, model=model, phantom_cm=32
        )
        return float(quantity.value), f"{vendor} {model}"
    except (CoefficientNotAvailable, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _coverage_of_use(compared: list[dict[str, Any]]) -> dict[str, Any]:
    """Whether the models the study actually reconstructs for are among the measured.

    The question that decides what the comparison is worth. A reconstruction measured
    on models the study does not use tells the reader about the table, not about this
    cohort's numbers.
    """
    results = json.loads(SERIES_RESULTS.read_text(encoding="utf-8"))
    rows = results if isinstance(results, list) else results.get("series", [])
    measured = {row["model_name"] for row in compared}

    used: list[dict[str, Any]] = []
    for row in rows:
        if not str(row.get("ctdivol_source", "")).startswith("reconstructed"):
            continue
        model = str(row.get("model_name", ""))
        errors = [
            entry["relative_error"] for entry in compared if entry["model_name"] == model
        ]
        used.append(
            {
                "vendor": row.get("vendor", ""),
                "model_name": model,
                "ctdivol_mgy": row.get("ctdivol_mgy"),
                "measured": model in measured,
                "median_relative_error": (
                    statistics.median(errors) if errors else None
                ),
            }
        )

    unmeasured = [row for row in used if not row["measured"]]
    return {
        "n_series_using_a_reconstructed_value": len(used),
        "n_on_a_measured_model": len(used) - len(unmeasured),
        "n_on_an_unmeasured_model": len(unmeasured),
        "series": used,
        "note": (
            "A model can only be measured here if some series in the cohort carries a "
            "recorded CTDIvol for it. No GE series in this cohort retained one, so the "
            "reconstruction cannot be checked on any GE model -- which is where this "
            "study uses it most. The validation is unavailable precisely where the "
            "reconstruction carries the most weight, and that is a property of the "
            "archive rather than of the method."
        ),
    }


def validate(limit: int | None = None) -> dict[str, Any]:
    entries = _recorded_series()
    if limit:
        entries = entries[:limit]

    compared: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for entry in entries:
        value, note = _reconstruct(entry)
        if value is None:
            skipped.append(
                {
                    "series_instance_uid": entry["series_instance_uid"],
                    "vendor": entry["vendor"],
                    "model_name": entry["model_name"],
                    "reason": note,
                }
            )
            print(f"  skip  {entry['vendor']:14} {note[:70]}")
            continue
        recorded = entry["recorded_mgy"]
        compared.append(
            {
                "series_instance_uid": entry["series_instance_uid"],
                "vendor": entry["vendor"],
                "model_name": entry["model_name"],
                "collection": entry["collection"],
                "recorded_mgy": recorded,
                "reconstructed_mgy": value,
                "relative_error": value / recorded - 1.0,
                "table_entry": note,
            }
        )
        print(
            f"  ok    {entry['vendor']:14} recorded {recorded:7.2f}  "
            f"reconstructed {value:7.2f}  {value / recorded - 1.0:+7.1%}"
        )

    errors = [abs(row["relative_error"]) for row in compared]
    signed = [row["relative_error"] for row in compared]

    by_vendor: dict[str, Any] = {}
    for vendor in sorted({row["vendor"] for row in compared}):
        values = [
            abs(row["relative_error"]) for row in compared if row["vendor"] == vendor
        ]
        by_vendor[vendor] = {
            "n": len(values),
            "median_absolute_relative_error": statistics.median(values),
            "max_absolute_relative_error": max(values),
        }

    by_model: dict[str, Any] = {}
    for model in sorted({row["model_name"] for row in compared}):
        values = [
            row["relative_error"] for row in compared if row["model_name"] == model
        ]
        by_model[model] = {
            "n": len(values),
            "relative_errors": values,
            "median_relative_error": statistics.median(values),
        }

    return {
        "what": (
            "The open physics reconstruction of CTDIvol, run on the series that carry "
            "a recorded CTDIvol, compared against that recorded value."
        ),
        "by_model": by_model,
        "by_model_note": (
            "The disagreement is consistent within a scanner model rather than "
            "scattered across series, which points at the tabulated coefficient for "
            "that model rather than at the per-series acquisition inputs. Model "
            "resolution is exact after normalisation and rejects near misses, so a "
            "mis-resolved model is not the explanation."
        ),
        "why": (
            "No series in this cohort carries both: the pipeline reconstructs only "
            "where the header has no usable value, so the recorded and reconstructed "
            "populations are disjoint by construction. Forcing the reconstruction on "
            "the recorded series is the only way to measure it against a known value."
        ),
        "limitation": (
            "This measures the reconstruction where the header did retain CTDIvol. "
            "The series it is actually used on are those where the header did not, and "
            "if that omission correlates with anything that also moves the "
            "reconstruction, the agreement reported here is optimistic. It is an upper "
            "bound on accuracy, not an estimate of it."
        ),
        "n_recorded_series": len(entries),
        "n_compared": len(compared),
        "n_skipped": len(skipped),
        "agreement": {
            "median_absolute_relative_error": (
                statistics.median(errors) if errors else None
            ),
            "max_absolute_relative_error": max(errors) if errors else None,
            "median_signed_relative_error": (
                statistics.median(signed) if signed else None
            ),
            "within_10_percent": (
                sum(1 for e in errors if e <= 0.10) / len(errors) if errors else None
            ),
            "within_20_percent": (
                sum(1 for e in errors if e <= 0.20) / len(errors) if errors else None
            ),
        },
        "by_vendor": by_vendor,
        "coverage_of_use": _coverage_of_use(compared),
        "series": compared,
        "skipped": skipped,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--limit", type=int, default=None, help="only the first N series, for a smoke run"
    )
    args = parser.parse_args(argv)

    payload = validate(limit=args.limit)
    if not payload["n_compared"]:
        print("\nnothing could be reconstructed; not writing a result")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    agreement = payload["agreement"]
    print(
        f"\ncompared {payload['n_compared']} of {payload['n_recorded_series']} "
        f"recorded series"
    )
    print(
        f"  median |error| {agreement['median_absolute_relative_error']:.1%}"
        f"   max {agreement['max_absolute_relative_error']:.1%}"
        f"   within 20% {agreement['within_20_percent']:.0%}"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
