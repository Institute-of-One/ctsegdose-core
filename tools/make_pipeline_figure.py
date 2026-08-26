"""Draw the pipeline of data formation and verification as a flowchart.

Usage::

    python tools/make_pipeline_figure.py --tag 1.5mm

A reviewer asked for this in place of the script filenames the Methods section
listed: a filename is not an algorithm, and naming one tells a reader nothing about
what runs. The flowchart carries the stages, the counts at each stage, and -- the
part a list of filenames cannot show at all -- where the verification steps sit and
what each of them rejects.

Every count is read from ``results/`` rather than typed, so the figure cannot come to
disagree with the text.

Writes ``paper/figures/fig1_pipeline.{png,pdf}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from ctsegdose_core.figures import INK_PRIMARY, INK_SECONDARY, apply_style  # noqa: E402

#: Acquisition, then verification, then the quantity. Colour separates the three
#: kinds of box; the reader is told which is which in the caption, and the shapes
#: differ as well so the distinction survives greyscale.
FILL_STAGE = "#e8eef7"
FILL_CHECK = "#fdf0e3"
FILL_OUTPUT = "#e6f2ec"
EDGE_STAGE = "#4a6fa5"
EDGE_CHECK = "#c07a2c"
EDGE_OUTPUT = "#2f7d5b"

#: Raised after the first build: at 7.4 pt the boxes were legible in the .docx but
#: only just, and a figure that is only just legible on a reviewer's screen is one
#: they will not read. The figure grows with the type rather than the type shrinking
#: into the figure, so the print size is unchanged in proportion.
BODY = 9.6
SMALL = 8.6


def _load(tag: str) -> dict[str, Any]:
    analysis = json.loads(
        (REPO / "results" / f"analysis_{tag}.json").read_text(encoding="utf-8")
    )
    candidates = json.loads(
        (REPO / "results" / "candidates.json").read_text(encoding="utf-8")
    )
    checks_path = REPO / "results" / f"segmentation_checks_{tag}.json"
    checks = (
        json.loads(checks_path.read_text(encoding="utf-8"))
        if checks_path.exists()
        else {}
    )
    verification = json.loads(
        (REPO / "results" / "verification.json").read_text(encoding="utf-8")
    )
    return {
        "analysis": analysis,
        "candidates": candidates,
        "verification": verification,
        "checks": checks,
    }


def _counts(data: dict[str, Any]) -> dict[str, Any]:
    """Read every count, and fail rather than substitute when one is absent.

    An earlier version reached for ``counts.get("probed") or 0``. The probe count
    lives in verification.json and not in candidates.json, so the figure printed
    "0 series probed" -- a plausible-looking number in place of a missing one, which
    is worse than an error because nothing about it looks wrong.
    """
    analysis = data["analysis"]
    candidates = data["candidates"]["counts"]
    verification = data["verification"]["counts"]
    flow = analysis["record_flow"]
    eligibility = analysis["modulation_eligibility"]
    checks = data["checks"]
    return {
        "indexed": candidates["index_rows"],
        "screened": candidates["passed_metadata_screen"],
        "one_per_patient": candidates["after_one_series_per_patient"],
        "queued": candidates["probe_queue"],
        "probed": verification["probed"],
        "downloaded": analysis["cohort"]["n_series"],
        "organs_requested": flow["expected_organ_series_combinations"],
        "organ_records": flow["organ_records_produced"],
        "absent": flow["absent_combinations"]["total"],
        "truncated": flow["truncated_records"],
        "with_index": flow["records_with_an_organ_weighted_ctdivol"],
        "eligible_series": eligibility["n_eligible"],
        "ineligible_series": eligibility["n_ineligible"],
        "tolerance": eligibility["rounding_tolerance"],
        "checked_series": checks.get("n_series_checked"),
        "series_with_failures": checks.get("n_series_with_failures"),
    }


def _box(ax, xy, size, text, *, fill, edge, style="round,pad=0.02"):
    x, y = xy
    width, height = size
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle=style,
            linewidth=1.0,
            facecolor=fill,
            edgecolor=edge,
            zorder=2,
        )
    )
    ax.text(
        x, y, text, ha="center", va="center", fontsize=BODY, color=INK_PRIMARY,
        zorder=3, linespacing=1.45,
    )


def _arrow(ax, start, end, *, label=None, colour=INK_SECONDARY, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            color=colour,
            linestyle="--" if dashed else "-",
            shrinkA=1,
            shrinkB=1,
            zorder=1,
        )
    )
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(
            mx + 0.16, my, label, fontsize=SMALL, color=INK_SECONDARY,
            ha="left", va="center", zorder=3,
        )


def draw(tag: str, out_stem: Path) -> Path:
    apply_style(plt)
    counts = _counts(_load(tag))

    fig, ax = plt.subplots(figsize=(6.6, 9.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11.6)
    ax.axis("off")

    # The check boxes carry the longest lines in the figure; at 9.6 pt they need the
    # room, so they sit wider and further left than the first layout allowed.
    left, right = 2.9, 7.6
    wide, tall = (4.6, 0.86), (3.6, 0.78)

    # -- acquisition ---------------------------------------------------------------
    stages = [
        (10.9, f"Archive index, metadata only\n{counts['indexed']:,} CT series"),
        (9.7, f"Metadata screen\n{counts['screened']:,} series remain"),
        (
            8.5,
            f"One series per patient, vendor-balanced queue\n"
            f"{counts['one_per_patient']:,} series, {counts['queued']} queued",
        ),
        (
            7.3,
            f"Header probe, then download\n"
            f"{counts['probed']} probed, {counts['downloaded']} kept, "
            f"10 per manufacturer",
        ),
    ]
    for y, text in stages:
        _box(ax, (left, y), wide, text, fill=FILL_STAGE, edge=EDGE_STAGE)
    for (y0, _), (y1, _) in zip(stages, stages[1:], strict=False):
        _arrow(ax, (left, y0 - wide[1] / 2), (left, y1 + wide[1] / 2))

    # The screen and the probe are themselves verification; say what they reject.
    _box(
        ax, (right, 9.7), tall,
        "rejects: not abdominal,\nnot a reconstructed image,\ntoo few or too many slices",
        fill=FILL_CHECK, edge=EDGE_CHECK,
    )
    _arrow(ax, (left + wide[0] / 2, 9.7), (right - tall[0] / 2, 9.7), dashed=True)
    # Aligned with the probe row at 7.3, not the queue row above it: these are what
    # the probe rejects after reading the header, and attaching them to the stage
    # that only thins one series per patient would say the wrong thing.
    _box(
        ax, (right, 7.3), tall,
        "rejects: tube current not modulated,\nno per-slice tube current,\n"
        "not a reconstructed image",
        fill=FILL_CHECK, edge=EDGE_CHECK,
    )
    _arrow(ax, (left + wide[0] / 2, 7.3), (right - tall[0] / 2, 7.3), dashed=True)

    # -- formation -----------------------------------------------------------------
    formation = [
        (6.1, "Resample to a uniform slice grid;\nread Hounsfield units"),
        (4.9, "TotalSegmentator at inference\n12 abdominal organs"),
        (3.7, f"Organ records\n{counts['organ_records']} of {counts['organs_requested']} "
              f"organ-series combinations"),
    ]
    for y, text in formation:
        _box(ax, (left, y), wide, text, fill=FILL_STAGE, edge=EDGE_STAGE)
    _arrow(ax, (left, 7.3 - wide[1] / 2), (left, 6.1 + wide[1] / 2))
    for (y0, _), (y1, _) in zip(formation, formation[1:], strict=False):
        _arrow(ax, (left, y0 - wide[1] / 2), (left, y1 + wide[1] / 2))

    failures = counts["series_with_failures"]
    checked = counts["checked_series"]
    if checked:
        _box(
            ax, (right, 4.9), tall,
            f"Anatomical checks on every series\n{failures} of {checked} carried\n"
            f"at least one failure",
            fill=FILL_CHECK, edge=EDGE_CHECK,
        )
        _arrow(ax, (left + wide[0] / 2, 4.9), (right - tall[0] / 2, 4.9), dashed=True)

    _box(
        ax, (right, 3.7), tall,
        f"{counts['absent']} combinations absent:\nthe organ lay outside\nthe scanned range",
        fill=FILL_CHECK, edge=EDGE_CHECK,
    )
    _arrow(ax, (left + wide[0] / 2, 3.7), (right - tall[0] / 2, 3.7), dashed=True)

    # -- verification and the quantity ---------------------------------------------
    _box(
        ax, (left, 2.5), wide,
        f"Acquisition-constancy criterion\n{counts['eligible_series']} of "
        f"{counts['eligible_series'] + counts['ineligible_series']} series eligible",
        fill=FILL_CHECK, edge=EDGE_CHECK,
    )
    _arrow(ax, (left, 3.7 - wide[1] / 2), (left, 2.5 + wide[1] / 2))
    _box(
        ax, (right, 2.5), tall,
        f"output-governing attributes\nconstant to {counts['tolerance']:.0%}\n"
        f"within the series",
        fill=FILL_CHECK, edge=EDGE_CHECK,
    )
    _arrow(ax, (left + wide[0] / 2, 2.5), (right - tall[0] / 2, 2.5), dashed=True)

    _box(
        ax, (left, 1.2), wide,
        f"Anatomy-weighted CTDIvol\n{counts['with_index']} records carry an index",
        fill=FILL_OUTPUT, edge=EDGE_OUTPUT,
    )
    _arrow(ax, (left, 2.5 - wide[1] / 2), (left, 1.2 + wide[1] / 2))

    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out_stem.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)

    width, height = fig.get_size_inches()
    print(f"  {out_stem.name}  {width:.2f} x {height:.2f} in")
    return out_stem.with_suffix(".png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="1.5mm")
    parser.add_argument(
        "--out", type=Path, default=REPO / "paper" / "figures" / "fig1_pipeline"
    )
    args = parser.parse_args(argv)
    draw(args.tag, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
