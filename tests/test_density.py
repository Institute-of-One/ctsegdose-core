"""The HU -> density calibration and the organ mass derived from it."""

from __future__ import annotations

import numpy as np
import pytest

from ctsegdose_core.density import (
    AIR_HU_FLOOR,
    DEFAULT_CALIBRATION,
    DensityCalibration,
    density_g_cm3,
    organ_mass,
    sensitivity_to_slope,
)


def test_water_is_one_gram_per_cubic_centimetre():
    """The anchor every calibration shares, and the reason soft tissue is insensitive."""
    assert density_g_cm3(0.0) == pytest.approx(1.000, abs=1e-6)


def test_the_curve_is_monotonic_across_the_diagnostic_range():
    hu = np.arange(-1000.0, 2000.0, 10.0)
    rho = density_g_cm3(hu)
    assert np.all(np.diff(rho) >= -1e-12)


def test_soft_tissue_lands_where_soft_tissue_should():
    assert density_g_cm3(50.0) == pytest.approx(1.05, abs=0.01)
    assert density_g_cm3(-90.0) == pytest.approx(0.95, abs=0.02)


def test_bowel_gas_carries_air_density_not_tissue_density():
    assert density_g_cm3(-1000.0) < 0.01
    assert density_g_cm3(AIR_HU_FLOOR - 1) < 0.01


def test_the_curve_is_held_flat_beyond_cortical_bone_rather_than_extrapolated():
    """Metal artefact reaches thousands of HU; linear extrapolation would invent density."""
    assert density_g_cm3(3000.0) == pytest.approx(density_g_cm3(1524.0))


def test_a_calibration_without_a_source_is_refused():
    with pytest.raises(ValueError, match="name its source"):
        DensityCalibration(reference="  ")


def test_anchors_out_of_order_are_refused():
    with pytest.raises(ValueError, match="ascending"):
        DensityCalibration(anchors=((0.0, 1.0, "water"), (-1000.0, 0.0012, "air")))


# --- organ mass ------------------------------------------------------------------------


def uniform_organ(hu_value: float, shape=(10, 20, 20)):
    volume = np.full(shape, -1000.0, dtype=np.float32)
    mask = np.zeros(shape, dtype=bool)
    mask[2:8, 5:15, 5:15] = True
    volume[mask] = hu_value
    return volume, mask


def test_a_water_filled_organ_weighs_its_volume():
    volume, mask = uniform_organ(0.0)
    m = organ_mass("phantom", volume, mask, voxel_volume_mm3=1.0)
    assert m.n_voxels == 600
    assert m.volume_cm3 == pytest.approx(0.6)
    assert m.mass_g == pytest.approx(0.6, rel=1e-6)
    assert m.mean_density_g_cm3 == pytest.approx(1.0)


def test_mass_uses_the_measured_attenuation_not_a_nominal_tissue_density():
    """A fatty liver and a normal liver of equal volume do not have equal mass."""
    fatty, mask = uniform_organ(-40.0)
    normal, _ = uniform_organ(60.0)
    m_fatty = organ_mass("liver", fatty, mask, voxel_volume_mm3=1.0)
    m_normal = organ_mass("liver", normal, mask, voxel_volume_mm3=1.0)
    assert m_fatty.volume_cm3 == m_normal.volume_cm3
    assert m_fatty.mass_g < m_normal.mass_g


def test_gas_inside_a_mask_is_flagged_and_carries_almost_no_mass():
    volume, mask = uniform_organ(30.0)
    volume[mask.nonzero()[0][:400], mask.nonzero()[1][:400], mask.nonzero()[2][:400]] = -1000.0
    m = organ_mass("stomach", volume, mask, voxel_volume_mm3=1.0)
    assert any("gas" in w for w in m.warnings)
    assert m.mass_g < 0.35


def test_an_empty_mask_reports_zero_rather_than_failing():
    volume = np.zeros((4, 4, 4), dtype=np.float32)
    m = organ_mass("pancreas", volume, np.zeros((4, 4, 4), dtype=bool), voxel_volume_mm3=1.0)
    assert m.mass_g == 0.0 and m.warnings


def test_a_mask_on_a_different_grid_is_an_error_not_a_reshape():
    volume = np.zeros((4, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="does not match"):
        organ_mass("liver", volume, np.zeros((5, 4, 4), dtype=bool), voxel_volume_mm3=1.0)


def test_voxel_volume_scales_the_mass_linearly():
    volume, mask = uniform_organ(0.0)
    a = organ_mass("liver", volume, mask, voxel_volume_mm3=1.0)
    b = organ_mass("liver", volume, mask, voxel_volume_mm3=2.0)
    assert b.mass_g == pytest.approx(2.0 * a.mass_g)


def test_the_calibration_choice_barely_moves_an_abdominal_organ_mass():
    """The claim the module makes about itself, checked rather than asserted."""
    volume, mask = uniform_organ(55.0)
    assert sensitivity_to_slope(volume, mask, voxel_volume_mm3=1.0, perturbation=0.10) < 0.01


def test_the_calibration_travels_with_the_mass_it_produced():
    volume, mask = uniform_organ(0.0)
    d = organ_mass("liver", volume, mask, voxel_volume_mm3=1.0).to_dict()
    assert d["calibration"] == DEFAULT_CALIBRATION.name
    assert "rho(HU)" in d["formula"]
