"""From a dose index to an absorbed organ dose, one organ at a time.

The chain, and which project owns each link::

    CTDIvol                          recorded (0018,9345), or reconstructed   [ctdose-core]
      x  w_o                         organ-specific tube-current weighting    [ctdose-core]
      =  CTDIvol,o                   organ-specific weighted CTDIvol -- an *index*
      x  h_o(D_w,o)                  CTDIvol-normalised organ dose coefficient,
                                     size-corrected at the organ's own slices  [here]
      =  D_o                         absorbed organ dose, mGy                  [here]

Everything before the third line is IORN-004's result and is reproduced, not re-derived.
The contribution here is the last two lines, and the patient specificity in them: the
size correction uses a water-equivalent diameter measured from *this patient's* segmented
body contour over *this organ's* slices, and the organ mass reported alongside comes from
this patient's own attenuation rather than from a reference body.

Three properties are enforced rather than intended:

* an absorbed dose is never produced without a cited coefficient table
  (:mod:`ctsegdose_core.coefficients` refuses), so the index layer degrades gracefully
  instead of guessing;
* every reported dose carries its formula, its inputs and its uncertainty budget;
* the uncertainty budget lists only components with a stated source, so it can be read
  as a claim about what is known rather than a decorative error bar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .coefficients import CoefficientTable, OrganCoefficient
from .density import OrganMass

#: Displayed CTDIvol is required to agree with measurement to within this fraction, so a
#: recorded value carries at least this much uncertainty into anything derived from it.
IEC_CTDIVOL_TOLERANCE = 0.20
IEC_REFERENCE = (
    "IEC 60601-2-44: the displayed CTDIvol must agree with the measured value to "
    "within 20%."
)


@dataclass
class UncertaintyComponent:
    """One contribution to the combined uncertainty, with the source that justifies it."""

    name: str
    relative: float
    reference: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "relative": round(self.relative, 4), "reference": self.reference}


def combine(components: list[UncertaintyComponent]) -> float:
    """Combined relative standard uncertainty, components added in quadrature."""
    return math.sqrt(sum(c.relative**2 for c in components))


@dataclass
class OrganDose:
    """Everything computed for one organ of one series."""

    organ: str
    # -- anatomy, from this patient
    n_voxels: int
    volume_cm3: float
    mass_g: float
    mean_density_g_cm3: float
    mean_hu: float
    slice_span: tuple[int, int]
    #: Mask centre of mass in image indices ``(z, y, x)``. Recorded because it is what
    #: makes a left/right or head/foot flip detectable from the results alone: a mirrored
    #: segmentation produces correct-looking volumes and masses and wrong anatomy.
    centroid_zyx: tuple[float, float, float] | None
    water_equivalent_diameter_cm: float | None
    # -- modulation, from the recorded tube current
    mean_tube_current_ma: float
    relative_weight: float
    # -- dose
    #: ``None`` when the series carries no dose index at all — neither recorded nor
    #: reconstructable. Not a NaN: a missing value must serialise as JSON ``null``, so
    #: that a reader parsing the results cannot mistake it for a number, and so that the
    #: file stays valid JSON rather than the ``NaN`` extension.
    ctdivol_mgy: float | None
    ctdivol_source: str
    organ_weighted_ctdivol_mgy: float | None
    #: The organ's mask reaches the first or last slice, so the organ continues beyond
    #: the scan. Its mass is the mass of the scanned part, and its modulation weighting
    #: describes only the exposed part -- neither is the organ's.
    truncated: bool = False
    coefficient: float | None = None
    absorbed_dose_mgy: float | None = None
    combined_relative_uncertainty: float | None = None
    uncertainty_components: list[UncertaintyComponent] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "organ": self.organ,
            "n_voxels": self.n_voxels,
            "volume_cm3": round(self.volume_cm3, 2),
            "mass_g": round(self.mass_g, 1),
            "mean_density_g_cm3": round(self.mean_density_g_cm3, 4),
            "mean_hu": round(self.mean_hu, 1),
            "slice_span": list(self.slice_span),
            "centroid_zyx": (
                None if self.centroid_zyx is None else [round(v, 1) for v in self.centroid_zyx]
            ),
            "water_equivalent_diameter_cm": (
                None if self.water_equivalent_diameter_cm is None
                else round(self.water_equivalent_diameter_cm, 2)
            ),
            "mean_tube_current_ma": round(self.mean_tube_current_ma, 1),
            "relative_weight": round(self.relative_weight, 4),
            "truncated": self.truncated,
            "ctdivol_mgy": None if self.ctdivol_mgy is None else round(self.ctdivol_mgy, 3),
            "ctdivol_source": self.ctdivol_source,
            "organ_weighted_ctdivol_mgy": (
                None if self.organ_weighted_ctdivol_mgy is None
                else round(self.organ_weighted_ctdivol_mgy, 3)
            ),
            "coefficient_mgy_per_mgy": (
                None if self.coefficient is None else round(self.coefficient, 4)
            ),
            "absorbed_dose_mgy": (
                None if self.absorbed_dose_mgy is None else round(self.absorbed_dose_mgy, 3)
            ),
            "combined_relative_uncertainty": (
                None if self.combined_relative_uncertainty is None
                else round(self.combined_relative_uncertainty, 4)
            ),
            "uncertainty_components": [c.to_dict() for c in self.uncertainty_components],
            "formulae": {
                "organ_weighted_ctdivol": "CTDIvol_o = CTDIvol * w_o",
                "w_o": "sum_z n_o(z) I(z) / sum_z n_o(z), divided by mean_z I(z)",
                "absorbed_dose": "D_o = CTDIvol_o * h_o(D_w,o)",
                "size_correction": "h_o(D_w) = h_ref * exp(-alpha * (D_w - D_w_ref))",
                "organ_mass": "m_o = sum over mask voxels of rho(HU) * V_voxel",
            },
            "notes": list(self.notes),
        }


def organ_dose(
    mass: OrganMass,
    *,
    slice_span: tuple[int, int],
    mean_tube_current_ma: float,
    relative_weight: float,
    ctdivol_mgy: float | None,
    ctdivol_source: str,
    ctdivol_relative_uncertainty: float = IEC_CTDIVOL_TOLERANCE,
    ctdivol_uncertainty_reference: str = IEC_REFERENCE,
    water_equivalent_diameter_cm: float | None = None,
    centroid_zyx: tuple[float, float, float] | None = None,
    truncated: bool = False,
    coefficient: OrganCoefficient | None = None,
) -> OrganDose:
    """Assemble one organ's result, converting to mGy only when a coefficient exists."""
    weighted = None if ctdivol_mgy is None else ctdivol_mgy * relative_weight
    result = OrganDose(
        organ=mass.organ,
        n_voxels=mass.n_voxels,
        volume_cm3=mass.volume_cm3,
        mass_g=mass.mass_g,
        mean_density_g_cm3=mass.mean_density_g_cm3,
        mean_hu=mass.mean_hu,
        slice_span=slice_span,
        centroid_zyx=centroid_zyx,
        water_equivalent_diameter_cm=water_equivalent_diameter_cm,
        mean_tube_current_ma=mean_tube_current_ma,
        relative_weight=relative_weight,
        ctdivol_mgy=ctdivol_mgy,
        ctdivol_source=ctdivol_source,
        organ_weighted_ctdivol_mgy=weighted,
        truncated=truncated,
        notes=list(mass.warnings),
    )
    if truncated:
        result.notes.append(
            "the organ reaches the edge of the scan and continues beyond it; the mass "
            "is that of the scanned part and the weighting describes only the exposed "
            "part, so this organ must be excluded from a whole-organ dose"
        )

    if weighted is None:
        result.notes.append(
            "the series carries no CTDIvol, recorded or reconstructable, so no dose "
            "index exists for this organ; its volume, mass and modulation weight do"
        )
        return result

    if coefficient is None:
        result.notes.append(
            "no CTDIvol-normalised coefficient available for this organ; reported to the "
            "organ-specific weighted CTDIvol only, which is a dose index and not an "
            "absorbed dose"
        )
        return result

    if water_equivalent_diameter_cm is None:
        result.notes.append(
            "no water-equivalent diameter could be measured over this organ's slices; "
            "the coefficient is used at its reference size, so the result is not "
            "patient-size-corrected"
        )
        h = coefficient.h_ref
    else:
        h = coefficient.at(water_equivalent_diameter_cm)

    components = [
        UncertaintyComponent("coefficient", coefficient.relative_uncertainty, "coefficient table"),
        UncertaintyComponent("ctdivol", ctdivol_relative_uncertainty, ctdivol_uncertainty_reference),
    ]
    result.coefficient = h
    result.absorbed_dose_mgy = weighted * h
    result.uncertainty_components = components
    result.combined_relative_uncertainty = combine(components)
    return result


def water_equivalent_diameter_over(
    volume_hu: np.ndarray,
    body_mask: np.ndarray,
    pixel_spacing_mm: tuple[float, float],
    slice_indices: range | list[int],
) -> float | None:
    """Median water-equivalent diameter over a set of slices, from a segmented body mask.

    Delegates the measurement itself to ctdose-core, which implements the AAPM TG-220
    definition; what is added here is *which* slices to measure over — an organ's own
    extent rather than a single mid-scan slice, so a size correction applied to the
    liver is not taken from the pelvis.
    """
    from ctdose_core.organ import water_equivalent_diameter_from_masks

    try:
        q = water_equivalent_diameter_from_masks(
            volume_hu, body_mask, pixel_spacing_mm, slice_indices=slice_indices
        )
    except ValueError:
        return None
    return float(q.value)


def summarise(doses: list[OrganDose], table: CoefficientTable | None) -> dict[str, Any]:
    """Series-level summary: what was computed, for how many organs, and how far it got."""
    absorbed = [d for d in doses if d.absorbed_dose_mgy is not None]
    return {
        "n_organs": len(doses),
        "n_organs_with_absorbed_dose": len(absorbed),
        "reached": "absorbed organ dose (mGy)" if absorbed else "organ-specific weighted CTDIvol",
        "total_organ_mass_g": round(sum(d.mass_g for d in doses), 1),
        "coefficient_table": table.provenance() if table else None,
        "highest_weighted_organ": (
            max(doses, key=lambda d: d.relative_weight).organ if doses else None
        ),
    }
