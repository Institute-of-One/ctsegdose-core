"""One series in, one auditable organ-dose record out.

The stages are deliberately separable — load, resolve CTDIvol, segment, measure, weight,
convert — because they fail for different reasons and a batch has to say *which* stage
failed for a given series rather than dropping it. A series whose scanner is outside the
open CTDI database still yields organ volumes and masses; a series with no coefficient
table still yields organ-specific weighted CTDIvol. Nothing is silently skipped.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

from . import __version__
from .coefficients import CoefficientTable
from .density import DEFAULT_CALIBRATION, DensityCalibration, organ_mass
from .dose import OrganDose, organ_dose, summarise, water_equivalent_diameter_over
from .grid import NonUniformGrid, SliceGrid, resolve_grid
from .segment import (
    ABDOMINAL_ORGANS,
    BODY_STRUCTURE,
    SegmentationUnavailable,
    dicom_affine,
    load_masks,
    segment_volume,
)
from .volume import load_volume_hu


@dataclass
class SeriesResult:
    """The record for one series: what was computed, from what, and what was not."""

    series_uid: str
    vendor: str
    manufacturer: str
    model_name: str
    collection: str
    patient_id: str
    n_slices: int
    kvp: float | None
    ctdivol_mgy: float | None
    ctdivol_source: str
    scan_mean_tube_current_ma: float | None
    organs: list[OrganDose] = field(default_factory=list)
    segmentation: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    slice_grid: dict[str, Any] = field(default_factory=dict)
    stage_reached: str = ""
    error: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self, table: CoefficientTable | None = None) -> dict[str, Any]:
        return {
            "series_instance_uid": self.series_uid,
            "vendor": self.vendor,
            "manufacturer": self.manufacturer,
            "model_name": self.model_name,
            "collection": self.collection,
            "patient_id": self.patient_id,
            "n_slices": self.n_slices,
            "kvp": self.kvp,
            "ctdivol_mgy": self.ctdivol_mgy,
            "ctdivol_source": self.ctdivol_source,
            "scan_mean_tube_current_ma": self.scan_mean_tube_current_ma,
            "stage_reached": self.stage_reached,
            "error": self.error,
            "warnings": list(self.warnings),
            "slice_grid": self.slice_grid,
            "segmentation": self.segmentation,
            "density_calibration": self.calibration,
            "summary": summarise(self.organs, table) if self.organs else None,
            "organs": [o.to_dict() for o in self.organs],
            "ctsegdose_core_version": __version__,
        }


#: A recorded CTDIvol outside this range is not a dose, it is a corrupt attribute. One
#: Philips series in this sample records -3.7e19 mGy, which is a 64-bit sentinel that
#: reached the header; taking it at face value would produce an absurd organ dose with
#: full provenance attached, which is worse than having none.
CTDIVOL_PLAUSIBLE_MGY = (0.01, 1000.0)


def resolve_ctdivol(series) -> tuple[float | None, str, list[str]]:
    """CTDIvol for a series: the recorded value, else the open physics reconstruction.

    Reproduces IORN-004's resolution rather than reimplementing it, and keeps the two
    apart in the record: a recorded and a reconstructed CTDIvol must never become
    interchangeable by accident, because the uncertainty they carry differs. A recorded
    value is used only if it is a physically possible dose -- an archive header is not
    an oracle, and a corrupt one must fall through to reconstruction rather than
    propagate.
    """
    warnings: list[str] = []
    tag = series.ctdivol_tag
    if tag is not None and tag.value:
        value = float(tag.value)
        lo, hi = CTDIVOL_PLAUSIBLE_MGY
        if math.isfinite(value) and lo <= value <= hi:
            return value, "recorded (0018,9345)", warnings
        warnings.append(
            f"recorded CTDIvol (0018,9345) is {value:g} mGy, outside the physically "
            f"possible range {lo}-{hi} mGy; the attribute is corrupt and is discarded "
            "in favour of the physics reconstruction"
        )

    from ctdose_core.ctdi_table import CoefficientNotAvailable, resolve_model
    from ctdose_core.metrics import acquisition_from_series, estimate_ctdivol_open

    try:
        vendor, model = resolve_model(series.manufacturer, series.model_name)
        acq = acquisition_from_series(series)
        q = estimate_ctdivol_open(acq, vendor=vendor, model=model, phantom_cm=32)
        return float(q.value), f"reconstructed (open MIRDct table, {vendor} {model})", warnings
    except (CoefficientNotAvailable, ValueError) as exc:
        warnings.append(f"no CTDIvol: {exc}")
        return None, "unavailable", warnings


def slice_positions(files: list[Path]) -> np.ndarray:
    """Longitudinal position of each image, from Image Position (Patient)."""
    out = []
    for path in files:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        out.append(float(ds.ImagePositionPatient[2]))
    return np.asarray(out, dtype=float)


def load_on_uniform_grid(directory: Path | str, series_uid: str | None):
    """Load a series onto one regular slice grid, ordered from feet to head.

    Two things are fixed here, and both are invisible in the images themselves.

    **Repeated positions.** Public archives supply series whose file count is not their
    position count. Loading those as-is gives a spacing that is too small *and* a stack
    that repeats anatomy; both inflate every organ volume.

    **Slice order.** ctdose-core sorts by Slice Location (0020,1041), which on three of
    the four vendors in this sample carries the opposite sign to Image Position
    (Patient). Those volumes arrive head-first, so slice index 0 is the most *superior*
    slice. Nothing about the images looks wrong, and the segmentation is unaffected --
    the affine is built from the patient coordinates, so the anatomy is right -- but the
    index stops meaning "towards the head", which makes every reported slice span read
    backwards and any downstream reasoning about superior/inferior wrong.

    So the array is canonicalised: index 0 is the most inferior slice, always. The tube
    current is reordered with it, because ``I(z)`` is paired to the slice axis by index.
    """
    from ctdose_core.io import load_series

    series = load_series(directory, series_uid=series_uid)
    grid = resolve_grid(
        slice_positions(series.files), fallback_spacing_mm=series.slice_thickness_mm
    )
    order = grid.keep
    if order != list(range(len(series.files))):
        if series.tube_current_ma.size != len(series.files):
            raise NonUniformGrid(
                f"tube current is present on {series.tube_current_ma.size} of "
                f"{len(series.files)} slices, so it cannot be reordered with them "
                "without misaligning I(z)"
            )
        if slice_positions(series.files)[0] > slice_positions(series.files)[-1]:
            grid.warnings.append(
                "slices were stored head-first (Slice Location runs opposite to Image "
                "Position (Patient)); reordered so index 0 is the most inferior slice"
            )
        series.tube_current_ma = series.tube_current_ma[order]
        series.files = [series.files[i] for i in order]
        if series.slice_locations_mm.size == grid.n_input:
            series.slice_locations_mm = series.slice_locations_mm[order]
        grid.keep = list(range(len(order)))
    return series, grid


def series_geometry(series, grid: SliceGrid) -> tuple[np.ndarray, tuple[float, float, float]]:
    """The NIfTI affine for a loaded series, and its voxel spacing in ``(z, y, x)``.

    The slice *direction* is measured from the first and last Image Position (Patient)
    rather than assumed, so a series reconstructed in descending order is not silently
    flipped — which would pair every organ with the tube current of the wrong end of the
    patient. The slice *spacing* comes from the resolved grid, not from dividing the
    extent by the file count.
    """
    first = pydicom.dcmread(str(series.files[0]), stop_before_pixels=True)
    last = pydicom.dcmread(str(series.files[-1]), stop_before_pixels=True)
    orientation = tuple(float(v) for v in first.ImageOrientationPatient)
    origin = tuple(float(v) for v in first.ImagePositionPatient)
    step = np.asarray([float(v) for v in last.ImagePositionPatient]) - np.asarray(origin)
    norm = float(np.linalg.norm(step))
    direction = tuple(step / norm) if norm > 0 else (0.0, 0.0, 1.0)
    py, px = series.pixel_spacing_mm or (1.0, 1.0)
    affine = dicom_affine(orientation, origin, (py, px), direction, grid.spacing_mm)
    return affine, (grid.spacing_mm, float(py), float(px))


def threshold_body_mask(
    volume_hu: np.ndarray, pixel_spacing_mm: tuple[float, float]
) -> tuple[np.ndarray, str]:
    """Patient outline for every slice, by ctdose-core's deterministic threshold contour.

    Air/tissue threshold, largest connected component, hole fill: the AAPM TG-220
    construction. It needs no model weights, gives the same answer on every platform,
    and is what the water-equivalent diameter is defined on. A deep-learning body
    contour is more robust when arms or immobilisation hardware are in the field of
    view, and can be substituted by running TotalSegmentator's ``body`` task -- which is
    why the source is recorded rather than assumed.
    """
    from ctdose_core.segmentation import body_mask

    stack = np.zeros(volume_hu.shape, dtype=bool)
    for i in range(volume_hu.shape[0]):
        stack[i] = body_mask(volume_hu[i], pixel_spacing_mm)
    return stack, "ctdose-core threshold contour (AAPM TG-220)"


def run_series(
    directory: Path | str,
    *,
    work_dir: Path | str,
    series_uid: str | None = None,
    metadata: dict[str, Any] | None = None,
    table: CoefficientTable | None = None,
    calibration: DensityCalibration = DEFAULT_CALIBRATION,
    organs: tuple[str, ...] = ABDOMINAL_ORGANS,
    fast: bool = True,
    device: str = "",
    python_executable: str = "",
) -> SeriesResult:
    """Compute the organ record for one downloaded series."""
    from ctdose_core.organ import organ_weights_from_masks

    meta = metadata or {}
    result = SeriesResult(
        series_uid=series_uid or str(meta.get("series_instance_uid", "")),
        vendor=str(meta.get("vendor", "")),
        manufacturer=str(meta.get("manufacturer", "")),
        model_name=str(meta.get("model_name", "")),
        collection=str(meta.get("collection", "")),
        patient_id=str(meta.get("patient_id", "")),
        n_slices=0,
        kvp=None,
        ctdivol_mgy=None,
        ctdivol_source="",
        scan_mean_tube_current_ma=None,
        calibration=calibration.to_dict(),
    )

    try:
        series, grid = load_on_uniform_grid(directory, series_uid)
        result.stage_reached = "loaded"
    except Exception as exc:
        result.error = f"load failed: {type(exc).__name__}: {exc}"
        return result

    result.slice_grid = grid.to_dict()
    result.warnings.extend(grid.warnings)
    result.n_slices = series.n_slices
    result.kvp = series.kvp
    result.manufacturer = result.manufacturer or series.manufacturer
    result.model_name = result.model_name or series.model_name
    result.warnings.extend(series.warnings)
    if series.tube_current_ma.size != series.n_slices:
        result.error = (
            f"tube current on {series.tube_current_ma.size}/{series.n_slices} slices; "
            "I(z) is incomplete and the modulation weighting would be misaligned"
        )
        return result
    result.scan_mean_tube_current_ma = float(series.tube_current_ma.mean())

    ctdivol, source, warns = resolve_ctdivol(series)
    result.ctdivol_mgy, result.ctdivol_source = ctdivol, source
    result.warnings.extend(warns)

    try:
        volume, volume_notes = load_volume_hu(series)
        result.warnings.extend(volume_notes)
        affine, spacing = series_geometry(series, grid)
        result.stage_reached = "volume"
    except Exception as exc:
        result.error = f"volume/geometry failed: {type(exc).__name__}: {exc}"
        return result

    try:
        run = segment_volume(
            volume, affine, work_dir,
            voxel_spacing_mm=spacing, organs=organs, fast=fast, device=device,
            python_executable=python_executable,
        )
    except SegmentationUnavailable as exc:
        if fast:
            result.error = f"segmentation failed: {exc}"
            return result
        # The full-resolution model resamples to 1.5 mm isotropic, which on a wide-FOV
        # series can exhaust host memory in nnU-Net's export worker. Dropping to the
        # 3 mm model keeps the series in the sample; which model produced a mask is
        # recorded with it, so the difference is visible rather than averaged away.
        result.warnings.append(
            f"the 1.5 mm model failed on this series ({exc}); it was segmented with the "
            "3 mm model instead, and its masks are coarser than the rest of the sample"
        )
        try:
            run = segment_volume(
                volume, affine, Path(work_dir).parent / "3mm-fallback",
                voxel_spacing_mm=spacing, organs=organs, fast=True, device=device,
                python_executable=python_executable,
            )
        except Exception as retry_exc:
            result.error = f"segmentation failed at both resolutions: {retry_exc}"
            return result
    except Exception as exc:
        result.error = f"segmentation failed: {type(exc).__name__}: {exc}"
        return result

    try:
        masks = load_masks(run)
        result.segmentation = run.to_dict()
        result.stage_reached = "segmented"
    except Exception as exc:
        result.error = f"masks unreadable: {type(exc).__name__}: {exc}"
        return result

    body = masks.pop(BODY_STRUCTURE, None)
    body_source = "TotalSegmentator body task"
    if body is None:
        body, body_source = threshold_body_mask(volume, series.pixel_spacing_mm or (1.0, 1.0))
    result.segmentation["body_contour_source"] = body_source
    organ_masks = {name: m for name, m in masks.items() if m.any()}
    if not organ_masks:
        result.error = "no organ mask contains any voxel in this series"
        return result

    profile = organ_weights_from_masks(
        organ_masks,
        series.tube_current_ma,
        ctdivol_mgy=ctdivol,
        source="TotalSegmentator (Apache-2.0, inference only), separate process",
    )
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    py, px = series.pixel_spacing_mm or (1.0, 1.0)

    for weight in profile.weights:
        mask = organ_masks[weight.organ]
        mass = organ_mass(weight.organ, volume, mask, voxel_volume_mm3, calibration=calibration)
        idx = np.argwhere(mask)
        centroid = tuple(float(v) for v in idx.mean(axis=0)) if idx.size else None
        d_w = None
        if body is not None:
            lo, hi = weight.slice_span
            d_w = water_equivalent_diameter_over(volume, body, (py, px), list(range(lo, hi + 1)))
        result.organs.append(
            organ_dose(
                mass,
                slice_span=weight.slice_span,
                mean_tube_current_ma=weight.mean_tube_current_ma,
                relative_weight=weight.relative_weight,
                truncated=weight.slice_span[0] == 0 or weight.slice_span[1] == series.n_slices - 1,
                ctdivol_mgy=ctdivol,
                ctdivol_source=source,
                water_equivalent_diameter_cm=d_w,
                centroid_zyx=centroid,
                coefficient=table.get(weight.organ) if table else None,
            )
        )
    if ctdivol is None:
        result.warnings.append(
            "CTDIvol is unavailable, so organ-weighted CTDIvol and absorbed dose are not "
            "defined for this series; the organ weights, volumes and masses are"
        )
    result.stage_reached = "organ dose" if table else "organ-weighted CTDIvol"
    return result


def write_result(path: Path | str, result: SeriesResult, table: CoefficientTable | None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(table), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path
