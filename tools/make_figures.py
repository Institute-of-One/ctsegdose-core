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
    VENDOR_COLOURS,
    VENDOR_HATCHES,
    VENDOR_MARKERS,
    apply_style,
    caption,
    jitter,
    strip_chrome,
)

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


# --- figure 1: organ mass against the ICRP 89 reference ----------------------------------


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
        ax.text(x, median, f" {median:.2f}", fontsize=7, color=INK_PRIMARY,
                va="bottom", ha="center", fontweight="bold", zorder=5)

    ax.set_xticks(range(len(order)), [ORGAN_LABEL[o] for o in order])
    # Names both sides of the ratio: what was measured is an attenuation-derived
    # estimate, and what it is placed beside is a published reference-adult value, not a
    # ground truth for these subjects.
    ax.set_ylabel("estimated mass / ICRP 89 reference adult male mass")
    ax.set_ylim(0, top_of_axis)
    ax.set_xlim(-0.6, len(order) - 0.15)
    ax.grid(axis="x", visible=False)
    strip_chrome(ax)
    vendor_legend(ax, loc="upper left")
    ax.set_title("Estimated organ mass beside a published reference value", loc="left", pad=8)
    save(fig, out, "fig3_organ_mass_vs_icrp89")


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
    save(fig, out, "fig2_dose_index_availability")


# --- figure 3: one acquisition, end to end ------------------------------------------------


def figure_demonstration(payload: dict[str, Any], out: Path) -> None:
    """The method on a single series: I(z), organ extents, and the resulting index.

    Chosen automatically as the completed series with a recorded CTDIvol, the most
    untruncated organs and the widest weight spread -- i.e. the one where the modulation
    actually does something, which is what the figure is for.
    """
    def score(s):
        organs = [o for o in s["organs"] if not o["truncated"]]
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
    bottom.text(series["ctdivol_mgy"], len(organs) - 0.35,
                f" scan CTDIvol {series['ctdivol_mgy']:.1f} mGy",
                fontsize=7, color=INK_SECONDARY, va="center")
    for bar, o in zip(bars, organs, strict=True):
        bottom.text(bar.get_width() + max(values) * 0.015, bar.get_y() + bar.get_height() / 2,
                    f"×{o['relative_weight']:.2f}", fontsize=7, color=INK_PRIMARY, va="center")
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
    save(fig, out, "fig1_demonstration_case")
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
    save(fig, out, "fig4_study_limits")


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
    ]

    apply_style(plt)
    # Drawn in any order; *numbered* in order of first mention in the manuscript, which
    # is demonstration, availability, mass, limits.
    figure_mass(organs, tables, args.out)
    figure_availability(tables, args.out)
    demo = figure_demonstration(payload, args.out)
    figure_limits(tables, series_rows, args.out)

    captions = "\n\n".join(
        f"**Figure {i}.** {caption(kind, tables)}"
        for i, kind in enumerate(("demonstration", "availability", "mass", "limits"), 1)
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
