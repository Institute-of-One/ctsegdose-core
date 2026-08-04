"""Hounsfield units to mass density.

Organ *mass* is what makes an absorbed dose patient-specific rather than
phantom-specific, and mass is density times volume. The density has to come from the
patient, which in a CT scan means from the measured attenuation.

The default calibration here is a **piecewise-linear** HU -> density curve through
reference-tissue anchor points, the standard construction in radiotherapy treatment
planning since Schneider et al. (1996). It is a *model*, not a measurement, and it is
treated as one throughout: the anchor points are explicit, the curve is replaceable by a
scanner-specific calibration measured on a density phantom, and whichever curve was used
travels into the provenance of every mass derived from it.

**Why the choice matters less than it looks.** Abdominal organs are soft tissue: liver,
spleen, kidney and pancreas sit within roughly -20 to +80 HU, where every reasonable
calibration agrees to about one percent, because they all pass through water at 0 HU.
The calibration matters for bone and lung, not for the organs this work reports. That is
an argument for stating the calibration, not for pretending it is unimportant --
:func:`sensitivity_to_slope` quantifies it per organ so the claim is checked rather than
asserted.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Reference tissue densities in g/cm^3 at nominal Hounsfield values for a 120 kVp scan.
#:
#: The densities are ICRU Report 44 reference tissues -- physical properties of the
#: tissues themselves, not a scanner-specific table. The Hounsfield values are nominal:
#: they depend on tube voltage, reconstruction kernel and contrast phase, which is
#: exactly why this curve is a documented default rather than a constant of nature.
ICRU44_ANCHORS: tuple[tuple[float, float, str], ...] = (
    (-1000.0, 0.00121, "air"),
    (-700.0, 0.26, "lung, inflated"),
    (-98.0, 0.95, "adipose tissue"),
    (0.0, 1.000, "water"),
    (52.0, 1.05, "skeletal muscle"),
    (260.0, 1.16, "trabecular bone"),
    (1524.0, 1.92, "cortical bone"),
)

#: Below this the voxel is air in the scanner bore, not tissue, and contributes no mass.
AIR_HU_FLOOR = -950.0


@dataclass(frozen=True)
class DensityCalibration:
    """A piecewise-linear HU -> mass density curve, with its own provenance.

    Args:
        anchors: ``(HU, density_g_cm3, tissue)`` in ascending HU order.
        name: short identifier recorded with every derived mass.
        reference: where the anchor densities come from. Required — a mass computed
            from an uncited calibration cannot be audited.
        note: anything a reader needs in order to judge transferability.
    """

    anchors: tuple[tuple[float, float, str], ...] = ICRU44_ANCHORS
    name: str = "icru44-piecewise-linear"
    reference: str = (
        "Anchor densities: ICRU Report 44 (1989), Tissue Substitutes in Radiation "
        "Dosimetry and Measurement. Piecewise-linear HU-to-density construction after "
        "Schneider U, Pedroni E, Lomax A. The calibration of CT Hounsfield units for "
        "radiotherapy treatment planning. Phys Med Biol. 1996;41(1):111-124."
    )
    note: str = (
        "Hounsfield anchor values are nominal for a 120 kVp abdominal acquisition. A "
        "scanner- and protocol-specific calibration measured on a density phantom should "
        "replace this curve where one is available; abdominal soft-tissue organs are "
        "insensitive to the choice because every such curve passes through water at 0 HU."
    )

    def __post_init__(self) -> None:
        hus = [a[0] for a in self.anchors]
        if len(self.anchors) < 2:
            raise ValueError("a calibration needs at least two anchor points")
        if hus != sorted(hus):
            raise ValueError("calibration anchors must be in ascending HU order")
        if any(a[1] < 0 for a in self.anchors):
            raise ValueError("densities must be non-negative")
        if not self.reference.strip():
            raise ValueError(
                "a calibration must name its source; an uncited density curve makes "
                "every mass derived from it unauditable"
            )

    @property
    def hu(self) -> np.ndarray:
        return np.asarray([a[0] for a in self.anchors], dtype=float)

    @property
    def density(self) -> np.ndarray:
        return np.asarray([a[1] for a in self.anchors], dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "reference": self.reference,
            "note": self.note,
            "anchors": [
                {"hu": hu, "density_g_cm3": rho, "tissue": tissue}
                for hu, rho, tissue in self.anchors
            ],
            "air_hu_floor": AIR_HU_FLOOR,
        }

    def slope_at(self, hu: float) -> float:
        """Local d(density)/d(HU), in g/cm^3 per HU."""
        hus, rhos = self.hu, self.density
        i = min(max(bisect_right(list(hus), hu) - 1, 0), len(hus) - 2)
        return float((rhos[i + 1] - rhos[i]) / (hus[i + 1] - hus[i]))


DEFAULT_CALIBRATION = DensityCalibration()


def density_g_cm3(
    hu: np.ndarray | float, calibration: DensityCalibration = DEFAULT_CALIBRATION
) -> np.ndarray:
    """Mass density for Hounsfield values, by linear interpolation between anchors.

    Outside the anchor range the curve is held flat rather than extrapolated: beyond
    cortical bone the linear trend has no physical warrant, and metal artefact would
    otherwise produce densities no tissue has. Anything below :data:`AIR_HU_FLOOR` is
    air and returns the air density, so bowel gas and the bore contribute no mass.
    """
    arr = np.asarray(hu, dtype=float)
    out = np.interp(arr, calibration.hu, calibration.density)
    return np.where(arr <= AIR_HU_FLOOR, calibration.density[0], out)


@dataclass
class OrganMass:
    """Volume, mass and mean density of one segmented organ."""

    organ: str
    n_voxels: int
    volume_cm3: float
    mass_g: float
    mean_density_g_cm3: float
    mean_hu: float
    voxel_volume_mm3: float
    calibration: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "organ": self.organ,
            "n_voxels": self.n_voxels,
            "volume_cm3": round(self.volume_cm3, 3),
            "mass_g": round(self.mass_g, 2),
            "mean_density_g_cm3": round(self.mean_density_g_cm3, 4),
            "mean_hu": round(self.mean_hu, 1),
            "voxel_volume_mm3": round(self.voxel_volume_mm3, 5),
            "calibration": self.calibration,
            "formula": "m = sum over mask voxels of rho(HU) * V_voxel",
            "warnings": list(self.warnings),
        }


def organ_mass(
    organ: str,
    volume_hu: np.ndarray,
    mask: np.ndarray,
    voxel_volume_mm3: float,
    *,
    calibration: DensityCalibration = DEFAULT_CALIBRATION,
) -> OrganMass:
    """Mass and volume of one organ from its mask and the patient's own attenuation.

    Args:
        volume_hu: the series volume in Hounsfield units, shaped ``(z, y, x)``.
        mask: boolean array of the same shape.
        voxel_volume_mm3: in-plane area times slice spacing.

    The mass is the sum over mask voxels of the local density, not the organ volume
    times a nominal tissue density: a fatty liver and a normal liver of equal volume do
    not have equal mass, and it is that difference the segmentation is there to capture.
    """
    if volume_hu.shape != mask.shape:
        raise ValueError(
            f"organ {organ!r}: mask shape {mask.shape} does not match volume {volume_hu.shape}"
        )
    if voxel_volume_mm3 <= 0:
        raise ValueError(f"organ {organ!r}: voxel volume must be positive")
    selected = np.asarray(mask, dtype=bool)
    n = int(selected.sum())
    if n == 0:
        return OrganMass(organ, 0, 0.0, 0.0, 0.0, 0.0, voxel_volume_mm3, calibration.name,
                         ["mask is empty in this series"])

    hu = volume_hu[selected]
    rho = density_g_cm3(hu, calibration)
    voxel_cm3 = voxel_volume_mm3 / 1000.0
    warnings: list[str] = []
    air_fraction = float(np.mean(hu <= AIR_HU_FLOOR))
    if air_fraction > 0.05:
        warnings.append(
            f"{air_fraction:.0%} of the mask is at or below {AIR_HU_FLOOR:g} HU (gas or "
            "out-of-body); those voxels carry air density and almost no mass"
        )
    return OrganMass(
        organ=organ,
        n_voxels=n,
        volume_cm3=n * voxel_cm3,
        mass_g=float(rho.sum() * voxel_cm3),
        mean_density_g_cm3=float(rho.mean()),
        mean_hu=float(hu.mean()),
        voxel_volume_mm3=voxel_volume_mm3,
        calibration=calibration.name,
        warnings=warnings,
    )


def sensitivity_to_slope(
    volume_hu: np.ndarray, mask: np.ndarray, voxel_volume_mm3: float, *, perturbation: float = 0.10
) -> float:
    """Relative change in organ mass when the soft-tissue slope is perturbed.

    Answers the question the calibration choice actually raises: *how much of the
    reported mass is an artefact of the curve?* Returns the fractional mass change for a
    ``perturbation`` relative change in the water-to-muscle slope. For abdominal soft
    tissue this is small, and reporting it is what turns that from a claim into a
    measurement.
    """
    base = organ_mass("probe", volume_hu, mask, voxel_volume_mm3)
    if base.mass_g == 0:
        return 0.0
    anchors = list(DEFAULT_CALIBRATION.anchors)
    hu_m, rho_m, tissue = anchors[4]
    anchors[4] = (hu_m, 1.0 + (rho_m - 1.0) * (1.0 + perturbation), tissue)
    perturbed = organ_mass(
        "probe",
        volume_hu,
        mask,
        voxel_volume_mm3,
        calibration=DensityCalibration(
            anchors=tuple(anchors),
            name="perturbed",
            reference=DEFAULT_CALIBRATION.reference,
        ),
    )
    return abs(perturbed.mass_g - base.mass_g) / base.mass_g
