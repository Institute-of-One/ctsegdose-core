"""The manuscript figures, drawn from ``results/`` and from nothing else.

Every figure reads the shipped analysis tables or the per-series records, so a figure
cannot disagree with the numbers in the text: there is no second copy of the data to
drift from.

Three constraints shape the styling, and they are constraints rather than taste.

**Print, and grayscale.** A reader may see these in a black-and-white print or a
photocopy, so vendor identity is carried by *marker shape and hatch first*, with colour
as a redundant channel. Nothing in these figures is distinguishable by colour alone.

**Colour-vision deficiency.** The four vendor hues were checked with the palette
validator over all pairs: worst normal-vision separation ΔE 16.3, worst CVD 9.2 (OKLab
×100), both clear of their floors. Aqua sits below 3:1 against the surface, which is why
every series is also directly labelled.

**Recessive chrome.** Hairline grid, no top or right spine, no chartjunk. The marks
carry the ink.
"""

from __future__ import annotations

from typing import Any

#: Vendor hues, validated all-pairs in light mode. The order is the CVD-safety
#: mechanism, not a preference: re-ordering requires re-running the validator.
VENDOR_COLOURS: dict[str, str] = {
    "GE": "#2a78d6",
    "Siemens": "#eb6834",
    "Canon/Toshiba": "#1baf7a",
    "Philips": "#4a3aa7",
}
#: The channel that survives a grayscale print. Distinct silhouettes, not sizes.
VENDOR_MARKERS: dict[str, str] = {
    "GE": "o",
    "Siemens": "s",
    "Canon/Toshiba": "^",
    "Philips": "D",
}
#: The channel that survives grayscale in filled areas.
VENDOR_HATCHES: dict[str, str] = {
    "GE": "",
    "Siemens": "///",
    "Canon/Toshiba": "...",
    "Philips": "xxx",
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

#: Availability is an ordered state, not an identity, so it takes one hue light to dark
#: rather than three unrelated hues -- plus a hatch, so the order survives grayscale.
AVAILABILITY_STYLE: dict[str, tuple[str, str]] = {
    "recorded": ("#184f95", ""),
    "reconstructed": ("#6da7ec", "///"),
    "unrecoverable": ("#e1e0d9", "xxx"),
}


def apply_style(plt) -> None:
    """House style: hairline chrome, sans figures, tabular ticks."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "lines.linewidth": 1.4,
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
    })


def strip_chrome(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.8)


def jitter(n: int, width: float = 0.16, seed: int = 20260804) -> list[float]:
    """Deterministic horizontal offsets, so a figure is reproducible byte for byte.

    Random jitter would make the figure irreproducible, which for a paper that ships
    its figure script is a defect rather than a detail.
    """
    import numpy as np

    rng = np.random.default_rng(seed + n)
    return list(rng.uniform(-width, width, size=n))


def caption(kind: str, tables: dict[str, Any]) -> str:
    """One-sentence caption naming what the figure shows and what it excludes."""
    cohort = tables["cohort"]
    stem = (
        f"{cohort['n_series']} abdominal CT series, "
        f"{cohort['n_organ_records']} organ records, "
        f"{len(VENDOR_COLOURS)} manufacturers"
    )
    return {
        "mass": (
            f"Attenuation-derived estimated organ mass relative to the ICRP 89 reference "
            f"adult male value. {stem}. Organs truncated by the scan boundary are "
            "excluded, since their mass is that of the scanned part only. The reference "
            "is an external anchor, not a ground truth for these subjects."
        ),
        "availability": (
            f"Availability of a whole-scan dose index in the archived DICOM headers, by "
            f"manufacturer. {stem}. A series counted unrecoverable retained no CTDIvol in "
            "its header and its scanner lies outside the open coefficient database, so no "
            "anatomy-weighted index can be formed from it at all."
        ),
        "demonstration": (
            "One acquisition end to end: the segmented organs, the recorded per-slice tube "
            "current over each organ's longitudinal extent, and the anatomy-weighted "
            "CTDIvol index the modulation produces."
        ),
        "limits": (
            f"What limits an organ-level modulation analysis. {stem}. Truncation is the "
            "fraction of organ records the scan boundary cuts through; the weight spread is "
            "the peak-to-peak range of the organ weights within a series."
        ),
    }[kind]
