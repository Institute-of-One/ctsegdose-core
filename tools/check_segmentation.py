"""Run the anatomical sanity checks over a completed organ-dose run.

Usage::

    python tools/check_segmentation.py --tag 1.5mm

Reads ``results/organ_dose_<tag>.json``, applies the checks in
``ctsegdose_core/checks.py``, writes ``results/segmentation_checks_<tag>.json`` and
prints every failure. A silent mirror or inversion is the failure mode this exists to
catch; a run that passes is not thereby validated, only cleared of the failures that
leave no other trace.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ctsegdose_core.checks import check_series  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="1.5mm")
    ap.add_argument("--results", type=Path, default=REPO / "results")
    args = ap.parse_args()

    path = args.results / f"organ_dose_{args.tag}.json"
    if not path.exists():
        raise SystemExit(f"{path} not found; run tools/run_organ_dose.py --tag {args.tag} first")
    payload = json.loads(path.read_text(encoding="utf-8"))

    reports = [check_series(s) for s in payload["series"] if s.get("organs")]
    failed = [r for r in reports if not r["passed"]]

    out = {
        "source": str(path.relative_to(REPO)).replace("\\", "/"),
        "tag": args.tag,
        "n_series_checked": len(reports),
        "n_series_with_failures": len(failed),
        "reports": reports,
    }
    out_path = args.results / f"segmentation_checks_{args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"checked {len(reports)} series from {path.name}")
    for report in reports:
        mark = "ok " if report["passed"] else "FAIL"
        print(
            f"  {mark} {report['vendor']:14s} {report['series_instance_uid'][-12:]}"
            f"   ({report['n_not_applicable']} checks not applicable)"
        )
        for c in report["checks"]:
            if not c["passed"]:
                label = "advisory" if c["advisory"] else "FAILED  "
                print(f"        - {label} {c['name']}: {c['detail']}")
    print(f"\n{len(reports) - len(failed)}/{len(reports)} series passed every check")
    print(f"wrote {out_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
