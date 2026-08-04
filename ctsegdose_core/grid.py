"""The slice grid a series actually sits on, which is not always the one it claims.

Organ *mass* is voxel count times voxel volume times density, so the slice spacing is a
direct multiplier on every mass this project reports. Taking it as
``(z_last - z_first) / (n_slices - 1)`` is wrong whenever the file count is not the
position count, and public archives supply that case routinely:

* **duplicate positions** — the same z appearing more than once, which a PET/CT series
  or a re-sent study produces. The naive spacing is then too small *and* the stacked
  volume repeats anatomy, so an organ occupies more slices than it physically does. Both
  errors inflate organ volume, and they do it silently: the masks look right, the
  Hounsfield values look right, and only the numbers are wrong.
* **gaps** — a missing instance, or two acquisitions with different spacing concatenated
  into one series. A single affine cannot describe that volume at all.

So the grid is resolved explicitly: one slice per position, spacing from the median step
between neighbouring positions, and a uniformity check that fails loudly rather than
handing a wrong multiplier to everything downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: A step may differ from the median by this fraction before the grid is called
#: non-uniform. Reconstruction rounds positions to 0.01 mm, so a few tenths of a percent
#: is normal; a missing slice is a 100% deviation and must not pass.
UNIFORMITY_TOLERANCE = 0.02


class NonUniformGrid(ValueError):
    """The slices do not lie on a single regular grid, so one affine cannot describe them."""


@dataclass
class SliceGrid:
    """Which slices to use, and the spacing they sit on."""

    keep: list[int]
    spacing_mm: float
    n_input: int
    n_duplicates: int
    max_step_deviation: float
    warnings: list[str] = field(default_factory=list)

    @property
    def deduplicated(self) -> bool:
        return self.n_duplicates > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_input_slices": self.n_input,
            "n_slices_used": len(self.keep),
            "n_duplicate_positions_dropped": self.n_duplicates,
            "slice_spacing_mm": round(self.spacing_mm, 4),
            "max_step_deviation": round(self.max_step_deviation, 4),
            "uniformity_tolerance": UNIFORMITY_TOLERANCE,
            "warnings": list(self.warnings),
        }


def resolve_grid(
    z_positions: np.ndarray | list[float],
    *,
    tolerance: float = UNIFORMITY_TOLERANCE,
    fallback_spacing_mm: float | None = None,
) -> SliceGrid:
    """Resolve the usable slice grid from the longitudinal positions of a series.

    Args:
        z_positions: one per image, in file order.
        fallback_spacing_mm: used only when a single slice makes the step undefined.

    Returns:
        The indices to keep — one per distinct position, ordered from the most inferior
        to the most superior — with the spacing they sit on.

    Raises:
        NonUniformGrid: when the surviving positions are not evenly spaced, or when
            every slice reports the same position (a de-identification artefact seen in
            public archives, which makes the z axis unrecoverable rather than imprecise).
    """
    z = np.asarray(z_positions, dtype=float)
    if z.size == 0:
        raise NonUniformGrid("the series carries no slice positions")
    warnings: list[str] = []

    # One slice per position, keeping the first occurrence, ordered along +z.
    order = np.argsort(z, kind="stable")
    keep: list[int] = []
    seen: set[float] = set()
    for idx in order:
        key = round(float(z[idx]), 3)
        if key in seen:
            continue
        seen.add(key)
        keep.append(int(idx))

    n_duplicates = int(z.size - len(keep))
    if n_duplicates:
        warnings.append(
            f"{n_duplicates} of {z.size} images repeat a slice position; one image per "
            "position is used, because stacking duplicates repeats anatomy and inflates "
            "every organ volume derived from the stack"
        )

    if len(keep) == 1:
        if n_duplicates:
            # Seen in public archives: de-identification flattens the z coordinate, so
            # every slice reports one position. The images are fine; the axis is gone.
            raise NonUniformGrid(
                f"all {z.size} images report the same longitudinal position "
                f"({z[keep[0]]:g} mm); the axis is unrecoverable rather than imprecise, "
                "so no volume, no organ extent and no I(z) weighting can be built"
            )
        if fallback_spacing_mm and fallback_spacing_mm > 0:
            warnings.append("single slice; spacing taken from Slice Thickness (0018,0050)")
            return SliceGrid([keep[0]], float(fallback_spacing_mm), int(z.size), n_duplicates, 0.0, warnings)
        raise NonUniformGrid("a single slice position gives no spacing and no volume")

    steps = np.diff(z[keep])
    median = float(np.median(steps))
    if median <= 0:
        raise NonUniformGrid(
            f"all {z.size} images report the same or a non-increasing position; the "
            "longitudinal axis is unrecoverable and no volume can be built from it"
        )
    deviation = float(np.max(np.abs(steps - median)) / median)
    if deviation > tolerance:
        raise NonUniformGrid(
            f"slice steps vary by {deviation:.1%} around {median:.3f} mm (tolerance "
            f"{tolerance:.0%}); the series is not on one regular grid — it has a gap, or "
            "two acquisitions were concatenated. A single affine cannot describe it."
        )
    return SliceGrid(keep, median, int(z.size), n_duplicates, deviation, warnings)
