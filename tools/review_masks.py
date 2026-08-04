"""Render mask overlays for visual review before a number becomes a paper number.

Usage::

    python tools/review_masks.py --organ spleen --top 4

Ranks the segmented organs by mass relative to the ICRP 89 reference, takes the most
extreme, and draws each one over its own CT at three levels through the organ. Writes
``paper/figures/review/`` — a review artefact, not a manuscript figure.

This exists because an outlier has two explanations that look identical in a table: a
patient whose organ really is that size, and a mask that has leaked into something else.
Only the image separates them, so the image is what gets looked at.

Orientation is annotated on every panel (R/L, and the slice index) so that laterality
and the inferior-to-superior ordering can be checked at the same time — a mirrored
segmentation is exactly the failure that produces plausible-looking masses.
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
import numpy as np  # noqa: E402

from ctsegdose_core.analysis import ICRP89_REFERENCE_MASS_G  # noqa: E402
from ctsegdose_core.figures import INK_SECONDARY, apply_style  # noqa: E402
from ctsegdose_core.pipeline import load_on_uniform_grid  # noqa: E402
from ctsegdose_core.segment import load_mask  # noqa: E402
from ctsegdose_core.volume import load_volume_hu  # noqa: E402

#: Soft-tissue window: level 50, width 400. The window the organ would be read in.
WINDOW = (-150.0, 250.0)
#: Contour colour: the one hue that stays legible on grayscale tissue without being
#: mistaken for anything anatomical.
CONTOUR = "#eb6834"


def rank(payload: dict[str, Any], organ: str) -> list[dict[str, Any]]:
    reference = ICRP89_REFERENCE_MASS_G.get(organ)
    rows = []
    for s in payload["series"]:
        for o in s.get("organs", []):
            if o["organ"] != organ or o.get("truncated"):
                continue
            rows.append({
                "series": s, "organ": o,
                "ratio": (o["mass_g"] / reference) if reference else None,
            })
    rows.sort(key=lambda r: r["ratio"] or 0.0, reverse=True)
    return rows


def draw(row: dict[str, Any], organ: str, tag: str, out: Path, index: int) -> Path | None:
    series_row, o = row["series"], row["organ"]
    uid = series_row["series_instance_uid"]
    mask_path = REPO / "segmentations" / uid / tag / "masks" / f"{organ}.nii.gz"
    if not mask_path.exists():
        print(f"  ! no mask for {uid[-12:]}")
        return None

    directory = REPO / _local_path(uid)
    series, grid = load_on_uniform_grid(directory, uid)
    volume, _ = load_volume_hu(series)
    mask = load_mask(mask_path, tuple(int(n) for n in volume.shape))

    occupied = np.flatnonzero(mask.any(axis=(1, 2)))
    if occupied.size == 0:
        print(f"  ! empty mask for {uid[-12:]}")
        return None
    levels = [occupied[int(f * (occupied.size - 1))] for f in (0.25, 0.5, 0.75)]

    # Crop to the organ plus a margin, so the reader sees the organ rather than the room.
    ys, xs = np.where(mask.any(axis=0))
    pad = 60
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad, mask.shape[1])
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, mask.shape[2])

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    for ax, z in zip(axes, levels, strict=True):
        ax.imshow(volume[z, y0:y1, x0:x1], cmap="gray", vmin=WINDOW[0], vmax=WINDOW[1],
                  interpolation="nearest")
        ax.contour(mask[z, y0:y1, x0:x1].astype(float), levels=[0.5],
                   colors=CONTOUR, linewidths=1.1)
        ax.set_title(f"slice {z}", fontsize=7.5, color=INK_SECONDARY, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        # DICOM axial display: column 0 is the patient's right.
        ax.text(0.015, 0.5, "R", transform=ax.transAxes, color="#ffffff", fontsize=7,
                va="center", ha="left")
        ax.text(0.985, 0.5, "L", transform=ax.transAxes, color="#ffffff", fontsize=7,
                va="center", ha="right")

    ratio = row["ratio"]
    fig.suptitle(
        f"#{index}  {organ}  {o['mass_g']:.0f} g  ({ratio:.2f}x ICRP 89)   ·   "
        f"{o['volume_cm3']:.0f} cm3, mean {o['mean_hu']:.0f} HU   ·   "
        f"{series_row['vendor']} {series_row['model_name']} · {series_row['collection']}",
        fontsize=8, x=0.01, ha="left", y=1.02,
    )
    fig.text(0.01, -0.03, f"Series Instance UID {uid}", fontsize=6, color=INK_SECONDARY)

    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{organ}_{index:02d}_{ratio:.2f}x_{uid[-12:]}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name}  ({series_row['vendor']}, {o['mass_g']:.0f} g, {ratio:.2f}x)")
    return path


def _local_path(uid: str) -> str:
    provenance = json.loads((REPO / "data" / "PROVENANCE.json").read_text(encoding="utf-8"))
    for row in provenance["series"]:
        if row["series_instance_uid"] == uid:
            return row["local_path"]
    raise KeyError(uid)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--organ", default="spleen")
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--tag", default="1.5mm")
    ap.add_argument("--results", type=Path, default=REPO / "results")
    ap.add_argument("--out", type=Path, default=REPO / "paper" / "figures" / "review")
    args = ap.parse_args()

    payload = json.loads(
        (args.results / f"organ_dose_{args.tag}.json").read_text(encoding="utf-8")
    )
    rows = rank(payload, args.organ)
    if not rows:
        raise SystemExit(f"no untruncated {args.organ} records found")

    apply_style(plt)
    print(f"{args.organ}: {len(rows)} whole organs, reviewing the {args.top} largest")
    for i, row in enumerate(rows[: args.top], 1):
        draw(row, args.organ, args.tag, args.out, i)

    ratios = [r["ratio"] for r in rows]
    print(f"\n  full range {min(ratios):.2f}x - {max(ratios):.2f}x, "
          f"median {float(np.median(ratios)):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
