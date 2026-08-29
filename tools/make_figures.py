"""Draw the manuscript figures from ``results/``.

Usage::

    python tools/make_figures.py --tag 1.5mm

Writes ``paper/figures/fig1..fig4.{png,pdf}`` and a ``captions.md`` beside them. Every
value is read from the shipped results, so a figure cannot disagree with the text.
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

from ctsegdose_core.analysis import ICRP89_REFERENCE_MASS_G, VENDORS  # noqa: E402
from ctsegdose_core.figures import (  # noqa: E402
    AVAILABILITY_STYLE,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
    VENDOR_COLOURS,
    VENDOR_HATCHES,
    VENDOR_MARKERS,
    apply_style,
    caption,
    jitter,
    strip_chrome,
)

#: Colours for the segmentation overlay. Identity is carried by position and by the
#: direct labels on each panel; colour is a supporting channel, because no twelve-way
#: categorical palette is separable under colour-vision deficiency. Neighbouring organs
#: are given well-separated hues so that adjacency, which is where confusion would
#: actually occur, stays legible.
ORGAN_COLOUR = {
    "liver": "#2a78d6",
    "spleen": "#eb6834",
    "kidney_left": "#1baf7a",
    "kidney_right": "#4a3aa7",
    "stomach": "#eda100",
    "pancreas": "#e34948",
    "gallbladder": "#00a3a3",
    "adrenal_gland_left": "#b45bd6",
    "adrenal_gland_right": "#7a5c2e",
    "small_bowel": "#e87ba4",
    "colon": "#5b8c00",
    "urinary_bladder": "#0f7fa8",
}

ORGAN_LABEL = {
    "liver": "liver", "spleen": "spleen", "kidney_left": "kidney (L)",
    "kidney_right": "kidney (R)", "pancreas": "pancreas", "stomach": "stomach",
    "gallbladder": "gallbladder", "colon": "colon", "small_bowel": "small bowel",
    "urinary_bladder": "bladder", "adrenal_gland_left": "adrenal (L)",
    "adrenal_gland_right": "adrenal (R)",
}


def vendor_legend(ax, *, loc="upper right", title=None):
    handles = [
        plt.Line2D([], [], color=VENDOR_COLOURS[v], marker=VENDOR_MARKERS[v],
                   linestyle="none", markersize=4.5, markeredgewidth=0, label=v)
        for v in VENDORS
    ]
    ax.legend(handles=handles, loc=loc, title=title, handletextpad=0.4, borderpad=0.3)


# --- figure 1: the segmentation the measurement chain rests on ---------------------------

#: Soft-tissue window, level 40 / width 400 HU: how the abdomen is read clinically.
CT_WINDOW = (-160.0, 240.0)
#: Overlay opacity. Enough to identify the structure, light enough that the anatomy
#: underneath -- which is what the reader is being asked to judge the mask against --
#: remains visible.
OVERLAY_ALPHA = 0.38

#: Which organs to draw on each panel. Showing all twelve everywhere would be unreadable
#: and would tell the reader less, not more.
PANEL_ORGANS = {
    "coronal": ("liver", "spleen", "kidney_left", "kidney_right", "urinary_bladder"),
    "upper": ("liver", "spleen", "stomach", "pancreas",
              "adrenal_gland_left", "adrenal_gland_right"),
    "renal": ("kidney_left", "kidney_right", "liver", "spleen", "small_bowel"),
    "pelvic": ("colon", "small_bowel", "urinary_bladder"),
}


def _overlay(ax, image, masks, organs, *, aspect=1.0, extent=None) -> set[str]:
    """Draw one CT image with its organ masks, filled and outlined.

    Returns the organs actually rendered, so the legend can be built from what is on the
    page rather than from what was requested -- an organ whose mask does not intersect
    the chosen plane must not appear in the key.
    """
    ax.imshow(image, cmap="gray", vmin=CT_WINDOW[0], vmax=CT_WINDOW[1],
              interpolation="bilinear", aspect=aspect, extent=extent)
    drawn: set[str] = set()
    for organ in organs:
        mask = masks.get(organ)
        if mask is None or not mask.any():
            continue
        drawn.add(organ)
        colour = ORGAN_COLOUR[organ]
        rgba = np.zeros((*mask.shape, 4))
        rgba[..., :3] = matplotlib.colors.to_rgb(colour)
        rgba[..., 3] = mask * OVERLAY_ALPHA
        ax.imshow(rgba, interpolation="nearest", aspect=aspect, extent=extent)
        ax.contour(mask.astype(float), levels=[0.5], colors=colour, linewidths=0.9,
                   extent=extent)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return drawn


def figure_segmentation(payload, tag: str, out: Path, series_uid: str) -> dict[str, Any]:
    """Representative TotalSegmentator output on the acquisition used end to end.

    Real imaging and the masks this study actually generated: no schematic, no redrawn
    contour, nothing rendered by hand. The panels exist to let a reader judge the
    anatomical basis of the measurement chain, which is why the overlay is translucent
    and the underlying anatomy stays visible.
    """
    from ctsegdose_core.pipeline import load_on_uniform_grid
    from ctsegdose_core.segment import load_mask
    from ctsegdose_core.volume import load_volume_hu

    provenance = json.loads((REPO / "data" / "PROVENANCE.json").read_text(encoding="utf-8"))
    row = next(r for r in provenance["series"] if r["series_instance_uid"] == series_uid)
    series, grid = load_on_uniform_grid(REPO / row["local_path"], series_uid)
    volume, _ = load_volume_hu(series)
    shape = tuple(int(n) for n in volume.shape)

    mask_dir = REPO / "segmentations" / series_uid / tag / "masks"
    masks = {}
    for organ in ORGAN_COLOUR:
        path = mask_dir / f"{organ}.nii.gz"
        if path.exists():
            m = load_mask(path, shape)
            if m.any():
                masks[organ] = m

    def centroid_z(organ, fallback):
        m = masks.get(organ)
        if m is None:
            return fallback
        occupied = np.flatnonzero(m.any(axis=(1, 2)))
        return int(occupied.mean())

    levels = {
        "upper": centroid_z("liver", shape[0] * 3 // 4),
        "renal": centroid_z("kidney_left", shape[0] // 2),
        "pelvic": centroid_z("urinary_bladder", shape[0] // 6),
    }

    py, px = series.pixel_spacing_mm or (1.0, 1.0)
    dz = grid.spacing_mm

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.6),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.04})

    # -- (a) coronal, reconstructed from the axial stack.
    # The plane is chosen to intersect as many of the panel's organs as possible rather
    # than by geometry: a mid-body coronal cut passes anterior to the kidneys and shows
    # only the liver, which would tell the reader nothing about the rest of the set.
    coronal_targets = [o for o in PANEL_ORGANS["coronal"] if o in masks]
    counts = np.zeros(shape[1])
    for organ in coronal_targets:
        present = masks[organ].any(axis=(0, 2))
        counts += present.astype(float)
    y_index = int(np.argmax(counts)) if counts.any() else shape[1] // 2
    coronal = volume[:, y_index, :][::-1]
    coronal_masks = {k: m[:, y_index, :][::-1] for k, m in masks.items()}
    drawn = _overlay(axes[0, 0], coronal, coronal_masks, PANEL_ORGANS["coronal"],
                     aspect=dz / px)

    # -- (b-d) three axial levels
    for ax, key, title in (
        (axes[0, 1], "upper", "(b) upper abdomen"),
        (axes[1, 0], "renal", "(c) renal level"),
        (axes[1, 1], "pelvic", "(d) lower abdomen"),
    ):
        z = levels[key]
        drawn |= _overlay(ax, volume[z], {k: m[z] for k, m in masks.items()},
                          PANEL_ORGANS[key], aspect=py / px)
        ax.set_title(title, loc="left", fontsize=8, color=INK_SECONDARY, pad=3)
        ax.text(0.015, 0.5, "R", transform=ax.transAxes, color="#ffffff", fontsize=7,
                va="center", ha="left")
        ax.text(0.985, 0.5, "L", transform=ax.transAxes, color="#ffffff", fontsize=7,
                va="center", ha="right")

    axes[0, 0].set_title("(a) coronal", loc="left", fontsize=8, color=INK_SECONDARY, pad=3)
    # The coronal panel carries left-right anatomy too, so it is marked like the axials.
    axes[0, 0].text(0.02, 0.5, "R", transform=axes[0, 0].transAxes, color="#ffffff",
                    fontsize=7, va="center", ha="left")
    axes[0, 0].text(0.98, 0.5, "L", transform=axes[0, 0].transAxes, color="#ffffff",
                    fontsize=7, va="center", ha="right")
    # A scale bar on the coronal panel, drawn in image coordinates.
    bar_mm = 100.0
    axes[0, 0].plot([shape[2] * 0.06, shape[2] * 0.06 + bar_mm / px],
                    [shape[0] * 0.94, shape[0] * 0.94], color="#ffffff", linewidth=2.0)
    axes[0, 0].text(shape[2] * 0.06, shape[0] * 0.90, "10 cm", color="#ffffff", fontsize=7)

    shown = sorted(drawn)
    handles = [
        plt.Line2D([], [], marker="s", linestyle="none", markersize=6,
                   markerfacecolor=ORGAN_COLOUR[o], markeredgecolor=ORGAN_COLOUR[o],
                   alpha=0.85, label=ORGAN_LABEL.get(o, o))
        for o in shown
    ]
    fig.legend(handles=handles, loc="lower center", ncols=6, frameon=False, fontsize=7.5,
               handletextpad=0.4, columnspacing=1.1, bbox_to_anchor=(0.5, -0.035))
    save(fig, out, "fig2_segmentation")
    return {"series": row, "levels": levels, "organs_shown": shown}


# --- figure 4: organ mass against the ICRP 89 reference ----------------------------------


def figure_mass(organs: list[dict[str, Any]], tables: dict[str, Any], out: Path) -> None:
    """Mass relative to the reference, per organ, per vendor.

    A ratio rather than raw grams, because the five organs differ by a factor of twelve
    and raw mass on one axis would compress everything but the liver into the baseline.
    The reference line at 1.0 is the whole point of the figure.
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    order = ["liver", "spleen", "kidney_left", "kidney_right", "pancreas"]

    # The axis is set from the data, never to a round number: a fixed limit silently
    # drops the outliers, and the spleen -- the organ with the largest offset and the
    # one the paper discusses -- is exactly what would disappear.
    ratios = [
        o["mass_g"] / ICRP89_REFERENCE_MASS_G[o["organ"]] for o in organs
        if o["organ"] in order and not o["truncated"]
    ]
    top_of_axis = max(ratios) * 1.08
    # The same argument applies at the bottom, where a floor at zero was doing what
    # the paragraph above warns against: the smallest ratio in the cohort is a mask
    # the quality control flagged, and it is the point a reader checking that control
    # goes looking for. At a floor of zero its marker is drawn half outside the axes.
    bottom_of_axis = min(ratios) - max(ratios) * 0.025

    ax.axhline(1.0, color=INK_SECONDARY, linewidth=1.0, zorder=2)
    ax.text(len(order) - 0.42, 1.0, "ICRP 89\nreference", fontsize=7,
            color=INK_SECONDARY, va="center", ha="left", linespacing=1.3)

    for x, organ in enumerate(order):
        reference = ICRP89_REFERENCE_MASS_G[organ]
        for vendor in VENDORS:
            values = [
                o["mass_g"] / reference for o in organs
                if o["organ"] == organ and o["vendor"] == vendor and not o["truncated"]
            ]
            if not values:
                continue
            offset = {"GE": -0.24, "Siemens": -0.08, "Canon/Toshiba": 0.08, "Philips": 0.24}[vendor]
            xs = [x + offset + j * 0.5 for j in jitter(len(values))]
            ax.plot(xs, values, VENDOR_MARKERS[vendor], color=VENDOR_COLOURS[vendor],
                    markersize=3.6, markeredgewidth=0, alpha=0.75, zorder=3)
        # The median across all vendors: the number the text quotes.
        pooled = [
            o["mass_g"] / reference for o in organs
            if o["organ"] == organ and not o["truncated"]
        ]
        median = float(np.median(pooled))
        ax.plot([x - 0.36, x + 0.36], [median, median], color=INK_PRIMARY,
                linewidth=1.8, solid_capstyle="round", zorder=4)
        # Backed with the surface colour: the label sits on its own median bar and,
        # in the denser columns, over data points. The production editor asked
        # whether overlapping content here impedes reading; this removes the part
        # that does, without moving any point.
        ax.text(x, median, f" {median:.2f}", fontsize=7, color=INK_PRIMARY,
                va="bottom", ha="center", fontweight="bold", zorder=6,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8))

    # Name the failed mask rather than leaving it as an unexplained point near zero.
    # Reviewer 2 asked what the segmentation quality control can and cannot catch, and
    # this is the one case in the cohort where it caught something: a left kidney
    # segmented at a few cubic centimetres, which no patient has. It is annotated, not
    # removed -- Section 2.6 reports what it does to the result.
    lowest = min(
        (
            (o["mass_g"] / ICRP89_REFERENCE_MASS_G[o["organ"]], o)
            for o in organs
            if o["organ"] in order and not o["truncated"]
        ),
        key=lambda pair: pair[0],
    )
    ratio, record = lowest
    ax.annotate(
        "flagged by\nquality control",
        xy=(order.index(record["organ"]) + 0.24, ratio),
        # Left and low: the space between two clusters, rather than over the next
        # organ's points, which is where a label to the right of this one lands.
        xytext=(order.index(record["organ"]) - 0.36, ratio + top_of_axis * 0.05),
        fontsize=6.5, color=INK_SECONDARY, va="center", ha="right", linespacing=1.3,
        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8),
        arrowprops=dict(arrowstyle="-", color=INK_SECONDARY, linewidth=0.7,
                        shrinkA=0, shrinkB=3),
        zorder=6,
    )

    ax.set_xticks(range(len(order)), [ORGAN_LABEL[o] for o in order])
    # Names both sides of the ratio: what was measured is an attenuation-derived
    # estimate, and what it is placed beside is a published reference-adult value, not a
    # ground truth for these subjects.
    ax.set_ylabel("estimated mass / ICRP 89 reference adult male mass")
    ax.set_ylim(bottom_of_axis, top_of_axis)
    ax.set_yticks([t for t in ax.get_yticks() if t >= 0 and t <= top_of_axis])
    ax.set_xlim(-0.6, len(order) - 0.15)
    ax.grid(axis="x", visible=False)
    strip_chrome(ax)
    vendor_legend(ax, loc="upper left")
    ax.set_title("Estimated organ mass beside a published reference value", loc="left", pad=8)
    save(fig, out, "fig5_organ_mass_vs_icrp89")


# --- figure 2: dose-index availability ---------------------------------------------------


def figure_availability(tables: dict[str, Any], out: Path) -> None:
    """Recorded / reconstructed / unrecoverable per vendor.

    Availability is an ordered state, not four identities, so it takes one hue light to
    dark plus a hatch -- which is also what makes the order legible in grayscale.
    """
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    blocks = tables["availability"]["by_vendor"]
    states = ["recorded", "reconstructed", "unrecoverable"]

    for y, vendor in enumerate(reversed(VENDORS)):
        left = 0.0
        for state in states:
            n = blocks[vendor][state]
            if not n:
                continue
            colour, hatch = AVAILABILITY_STYLE[state]
            ax.barh(y, n, left=left, height=0.58, color=colour, hatch=hatch,
                    edgecolor="#fcfcfb", linewidth=1.6, zorder=3)
            ax.text(left + n / 2, y, str(n), ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if state == "recorded" else INK_PRIMARY,
                    fontweight="bold", zorder=4)
            left += n

    ax.set_yticks(range(len(VENDORS)), list(reversed(VENDORS)))
    ax.set_xlabel("series")
    ax.set_xlim(0, 10.4)
    ax.set_xticks(range(0, 11, 2))
    ax.grid(axis="y", visible=False)
    strip_chrome(ax)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=AVAILABILITY_STYLE[s][0],
                      hatch=AVAILABILITY_STYLE[s][1], edgecolor="#fcfcfb", linewidth=1.2,
                      label=s)
        for s in states
    ]
    # Below the axis, not inside it: at ten series per vendor the bars fill the plot and
    # an inset legend lands on the Philips row.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncols=3, handlelength=1.6, handletextpad=0.5, columnspacing=1.6)

    t = tables["availability"]["ge_vs_rest_recorded"]["table"]
    ge_n = t["ge_recorded"] + t["ge_not_recorded"]
    note = (
        f"none of the {ge_n} sampled GE series retained a recorded CTDIvol "
        "(descriptive; this archive sample only)"
    )
    ax.set_title("Whole-scan dose index in the archived headers", loc="left", pad=16)
    ax.text(0, 1.06, note, transform=ax.transAxes, fontsize=7.5, color=INK_SECONDARY)
    save(fig, out, "fig4_dose_index_availability")


# --- figure 3: one acquisition, end to end ------------------------------------------------


def figure_demonstration(
    payload: dict[str, Any], out: Path, ineligible: set[str] | None = None
) -> None:
    """The method on a single series: I(z), organ extents, and the resulting index.

    Chosen automatically as the completed series with a recorded CTDIvol, the most
    untruncated organs and the widest weight spread -- i.e. the one where the modulation
    actually does something, which is what the figure is for.
    """
    blocked = ineligible or set()

    def score(s):
        organs = [o for o in s["organs"] if not o["truncated"]]
        if s["series_instance_uid"] in blocked:
            return -1  # a series the constancy criterion excludes cannot be the exemplar
        if not organs or s["ctdivol_mgy"] is None or "recorded" not in s["ctdivol_source"]:
            return -1
        w = [o["relative_weight"] for o in organs]
        return len(organs) + 10 * (max(w) - min(w))

    series = max(payload["series"], key=score)
    organs = sorted(
        [o for o in series["organs"] if not o["truncated"]],
        key=lambda o: o["relative_weight"], reverse=True,
    )[:8]

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.4, 5.2), height_ratios=[1.15, 1.0],
        gridspec_kw={"hspace": 0.55},
    )
    colour = VENDOR_COLOURS[series["vendor"]]

    # -- (a) each organ's z extent, against the mean tube current it saw
    y = np.arange(len(organs))
    for i, o in enumerate(organs):
        lo, hi = o["slice_span"]
        top.plot([lo, hi], [i, i], color=colour, linewidth=4.5, solid_capstyle="round",
                 alpha=0.85, zorder=3)
        top.text(hi + series["n_slices"] * 0.03, i, f"{o['mean_tube_current_ma']:.0f} mA",
                 fontsize=7, color=INK_SECONDARY, va="center")
    scan_mean = series["scan_mean_tube_current_ma"]
    top.set_yticks(y, [ORGAN_LABEL.get(o["organ"], o["organ"]) for o in organs])
    top.invert_yaxis()
    top.set_xlabel("slice index (0 = most inferior)")
    top.set_xlim(0, series["n_slices"] * 1.28)
    top.grid(axis="y", visible=False)
    strip_chrome(top)
    top.set_title(
        f"(a) organ extent and the tube current over it "
        f"(scan mean {scan_mean:.0f} mA)", loc="left", pad=6,
    )

    # -- (b) the organ-specific weighted CTDIvol the modulation produces
    values = [o["organ_weighted_ctdivol_mgy"] for o in organs]
    bars = bottom.barh(y, values, height=0.6, color=colour, alpha=0.85,
                       hatch=VENDOR_HATCHES[series["vendor"]], edgecolor="#fcfcfb",
                       linewidth=1.2, zorder=3)
    bottom.axvline(series["ctdivol_mgy"], color=INK_SECONDARY, linewidth=1.0,
                   linestyle=(0, (4, 2)), zorder=4)
    # Below the last bar, not above the first: above collides with the panel title.
    # Below is not enough on its own -- at journal column width the label started
    # where the last bar's weight label ended and the two ran together, which is
    # what the production editor saw. The axis is extended to make a clear row for
    # it, and the label sits in that row rather than beside the bars.
    # Set before invert_yaxis below, which flips whatever limits are in place.
    bottom.set_ylim(-0.65, len(organs) + 0.15)
    bottom.text(series["ctdivol_mgy"], len(organs) - 0.35,
                f" scan CTDIvol {series['ctdivol_mgy']:.1f} mGy",
                fontsize=7, color=INK_SECONDARY, va="center")
    for bar, o in zip(bars, organs, strict=True):
        # Backed with the surface colour: where an organ's index sits close to the
        # scan CTDIvol, the dashed line runs through the middle of this label and
        # strikes the digits out. Two organs did that, and it is the second half of
        # what the production editor flagged as overlap in this panel.
        bottom.text(bar.get_width() + max(values) * 0.015, bar.get_y() + bar.get_height() / 2,
                    f"×{o['relative_weight']:.2f}", fontsize=7, color=INK_PRIMARY, va="center",
                    zorder=5,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8))
    bottom.set_yticks(y, [ORGAN_LABEL.get(o["organ"], o["organ"]) for o in organs])
    bottom.invert_yaxis()
    bottom.set_xlabel("organ-weighted CTDIvol index (mGy)")
    bottom.set_xlim(0, max(values) * 1.22)
    bottom.grid(axis="y", visible=False)
    strip_chrome(bottom)
    bottom.set_title("(b) the anatomy-weighted CTDIvol index, and the weight producing it",
                     loc="left", pad=6)

    fig.suptitle(
        f"{series['vendor']} · {series['model_name']} · {series['collection']} · "
        f"{series['n_slices']} slices",
        x=0.005, y=1.0, ha="left", fontsize=8, color=INK_SECONDARY, fontweight="normal",
    )
    save(fig, out, "fig3_demonstration_case")
    return series


# --- figure 4: what limits the study -----------------------------------------------------


def figure_limits(tables: dict[str, Any], series_rows: list[dict[str, Any]], out: Path) -> None:
    """Truncation rate and organ-weight spread, per vendor."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.4, 2.6), width_ratios=[1, 1.15],
                                      gridspec_kw={"wspace": 0.35})

    trunc = tables["study_limits"]["truncation"]["by_vendor"]
    x = np.arange(len(VENDORS))
    for i, vendor in enumerate(VENDORS):
        b = trunc[vendor]
        left.bar(i, b["truncated_rate"] * 100, width=0.62, color=VENDOR_COLOURS[vendor],
                 hatch=VENDOR_HATCHES[vendor], edgecolor="#fcfcfb", linewidth=1.2,
                 alpha=0.85, zorder=3)
        left.text(i, b["truncated_rate"] * 100 + 0.5,
                  f"{b['n_truncated']}/{b['n_organ_records']}", ha="center", fontsize=7,
                  color=INK_PRIMARY, fontweight="bold")
    left.set_xticks(x, [v.split("/")[0] for v in VENDORS], rotation=0)
    left.set_ylabel("organ records truncated (%)")
    left.set_ylim(0, 24)
    left.grid(axis="x", visible=False)
    strip_chrome(left)
    left.set_title("(a) truncation by the scan boundary", loc="left", pad=6, fontsize=8)

    for i, vendor in enumerate(VENDORS):
        spreads = [r["weight_spread"] for r in series_rows
                   if r["vendor"] == vendor and r["weight_spread"] is not None]
        xs = [i + j for j in jitter(len(spreads))]
        right.plot(xs, spreads, VENDOR_MARKERS[vendor], color=VENDOR_COLOURS[vendor],
                   markersize=4.2, markeredgewidth=0, alpha=0.8, zorder=3)
        if spreads:
            median = float(np.median(spreads))
            right.plot([i - 0.3, i + 0.3], [median, median], color=INK_PRIMARY,
                       linewidth=1.6, solid_capstyle="round", zorder=4)
    threshold = 0.02
    right.axhline(threshold, color=INK_SECONDARY, linewidth=0.9, linestyle=(0, (4, 2)), zorder=2)
    right.text(len(VENDORS) - 0.45, threshold, " flat", fontsize=7, color=INK_SECONDARY,
               va="bottom", ha="left")
    right.set_xticks(x, [v.split("/")[0] for v in VENDORS])
    right.set_ylabel("organ weight spread (peak-to-peak)")
    right.set_xlim(-0.55, len(VENDORS) - 0.3)
    right.grid(axis="x", visible=False)
    strip_chrome(right)
    right.set_title("(b) does the modulation vary across the organs?", loc="left",
                    pad=6, fontsize=8)
    save(fig, out, "fig6_study_limits")


# --- driver ------------------------------------------------------------------------------


def save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{suffix}")
    plt.close(fig)
    print(f"  wrote {out / stem}.png / .pdf")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="1.5mm")
    ap.add_argument("--results", type=Path, default=REPO / "results")
    ap.add_argument("--out", type=Path, default=REPO / "paper" / "figures")
    args = ap.parse_args()

    payload = json.loads((args.results / f"organ_dose_{args.tag}.json").read_text(encoding="utf-8"))
    tables = json.loads((args.results / f"analysis_{args.tag}.json").read_text(encoding="utf-8"))
    organs = [
        {**o, "vendor": s["vendor"]} for s in payload["series"] for o in s.get("organs", [])
    ]

    # Panels that describe modulation obey the acquisition-constancy criterion; panels
    # that describe the archive or the segmentation cover the whole cohort.
    eligibility = tables.get("modulation_eligibility") or {}
    ineligible = {r["series_instance_uid"] for r in eligibility.get("ineligible_series", [])}
    series_rows = [
        {
            "vendor": s["vendor"],
            "weight_spread": (
                max(w) - min(w)
                if len(w := [o["relative_weight"] for o in s.get("organs", [])]) > 1
                else None
            ),
        }
        for s in payload["series"]
        if s["series_instance_uid"] not in ineligible
    ]

    apply_style(plt)
    # Drawn in any order; *numbered* in order of first mention in the manuscript, which is
    # segmentation, demonstration, availability, mass, limits.
    figure_mass(organs, tables, args.out)
    figure_availability(tables, args.out)
    demo = figure_demonstration(payload, args.out, ineligible)
    figure_segmentation(payload, args.tag, args.out, demo["series_instance_uid"])
    figure_limits(tables, series_rows, args.out)

    captions = "\n\n".join(
        f"**Figure {i}.** {caption(kind, tables)}"
        for i, kind in enumerate(
            ("segmentation", "demonstration", "availability", "mass", "limits"), 1
        )
    )
    captions += (
        f"\n\nFigure 3 demonstration series: {demo['vendor']}, {demo['model_name']}, "
        f"collection {demo['collection']}, Series Instance UID {demo['series_instance_uid']}."
    )
    (args.out / "captions.md").write_text(captions + "\n", encoding="utf-8")
    print(f"  wrote {args.out / 'captions.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

