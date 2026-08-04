"""The dose chain, and the two refusals that keep it honest."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ctsegdose_core.coefficients import (
    InvalidCoefficientTable,
    MissingCoefficients,
    OrganCoefficient,
    load_table,
)
from ctsegdose_core.density import organ_mass
from ctsegdose_core.dose import IEC_CTDIVOL_TOLERANCE, UncertaintyComponent, combine, organ_dose
from ctsegdose_core.segment import dicom_affine


def a_mass(organ="liver", hu=55.0):
    volume = np.full((6, 10, 10), hu, dtype=np.float32)
    mask = np.zeros((6, 10, 10), dtype=bool)
    mask[1:5, 2:8, 2:8] = True
    return organ_mass(organ, volume, mask, voxel_volume_mm3=8.0)


def a_coefficient(**kw):
    fields = {
        "organ": "liver",
        "h_ref": 0.9,
        "alpha_per_cm": 0.04,
        "d_w_ref_cm": 30.0,
        "relative_uncertainty": 0.10,
    }
    fields.update(kw)
    return OrganCoefficient(**fields)


# --- the index layer ---------------------------------------------------------------------


def test_the_organ_weighted_ctdivol_is_ctdivol_times_the_organ_weight():
    d = organ_dose(
        a_mass(), slice_span=(1, 4), mean_tube_current_ma=260.0, relative_weight=1.3,
        ctdivol_mgy=10.0, ctdivol_source="recorded (0018,9345)",
    )
    assert d.organ_weighted_ctdivol_mgy == pytest.approx(13.0)


def test_without_a_coefficient_the_chain_stops_at_the_index_and_says_so():
    d = organ_dose(
        a_mass(), slice_span=(1, 4), mean_tube_current_ma=260.0, relative_weight=1.3,
        ctdivol_mgy=10.0, ctdivol_source="recorded (0018,9345)",
    )
    assert d.absorbed_dose_mgy is None
    assert any("not an absorbed dose" in n for n in d.notes)


# --- the dose layer ----------------------------------------------------------------------


def test_absorbed_dose_is_the_weighted_index_times_the_size_corrected_coefficient():
    coeff = a_coefficient()
    d = organ_dose(
        a_mass(), slice_span=(1, 4), mean_tube_current_ma=260.0, relative_weight=1.2,
        ctdivol_mgy=10.0, ctdivol_source="recorded (0018,9345)",
        water_equivalent_diameter_cm=34.0, coefficient=coeff,
    )
    expected = 10.0 * 1.2 * 0.9 * np.exp(-0.04 * (34.0 - 30.0))
    assert d.absorbed_dose_mgy == pytest.approx(expected)


def test_a_larger_patient_receives_less_organ_dose_for_the_same_index():
    """The size correction is the patient-specific step; its direction is not optional."""
    coeff = a_coefficient()
    small = coeff.at(26.0)
    large = coeff.at(38.0)
    assert small > coeff.h_ref > large


def test_the_coefficient_falls_back_to_its_reference_size_and_records_that_it_did():
    d = organ_dose(
        a_mass(), slice_span=(1, 4), mean_tube_current_ma=260.0, relative_weight=1.0,
        ctdivol_mgy=10.0, ctdivol_source="recorded (0018,9345)",
        water_equivalent_diameter_cm=None, coefficient=a_coefficient(),
    )
    assert d.coefficient == pytest.approx(0.9)
    assert any("not patient-size-corrected" in n for n in d.notes)


def test_a_dose_carries_an_uncertainty_budget_whose_components_name_their_source():
    d = organ_dose(
        a_mass(), slice_span=(1, 4), mean_tube_current_ma=260.0, relative_weight=1.0,
        ctdivol_mgy=10.0, ctdivol_source="recorded (0018,9345)",
        water_equivalent_diameter_cm=30.0, coefficient=a_coefficient(),
    )
    assert d.combined_relative_uncertainty == pytest.approx(np.hypot(0.10, IEC_CTDIVOL_TOLERANCE))
    assert all(c.reference for c in d.uncertainty_components)


def test_components_combine_in_quadrature():
    assert combine([UncertaintyComponent("a", 0.3, "x"), UncertaintyComponent("b", 0.4, "y")]) == pytest.approx(0.5)


# --- the refusals ------------------------------------------------------------------------


def test_an_absent_coefficient_table_is_refused_with_an_actionable_message(tmp_path):
    with pytest.raises(MissingCoefficients, match="published Monte-Carlo"):
        load_table(tmp_path / "nope.json")


@pytest.mark.parametrize("dropped", ["citation", "doi", "license", "license_url", "source_sha256"])
def test_a_table_that_cannot_be_audited_is_refused(tmp_path, dropped):
    """The lesson from the companion project: transcribed values need their licence."""
    provenance = {
        "citation": "Someone et al. 2026",
        "doi": "10.0000/x",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "source_sha256": "abc123",
    }
    provenance.pop(dropped)
    path = tmp_path / "table.json"
    path.write_text(json.dumps({
        "provenance": provenance,
        "coefficients": [{
            "organ": "liver", "h_ref_mgy_per_mgy": 0.9, "alpha_per_cm": 0.04,
            "d_w_ref_cm": 30.0, "relative_uncertainty": 0.1,
        }],
    }), encoding="utf-8")
    with pytest.raises(InvalidCoefficientTable, match=dropped):
        load_table(path)


def test_a_well_formed_table_loads_and_keeps_its_provenance(tmp_path):
    path = tmp_path / "table.json"
    path.write_text(json.dumps({
        "provenance": {
            "citation": "Someone et al. 2026", "doi": "10.0000/x", "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "source_sha256": "abc123",
        },
        "coefficients": [{
            "organ": "liver", "h_ref_mgy_per_mgy": 0.9, "alpha_per_cm": 0.04,
            "d_w_ref_cm": 30.0, "relative_uncertainty": 0.1,
        }],
    }), encoding="utf-8")
    table = load_table(path)
    assert table.get("liver").h_ref == 0.9
    assert table.get("spleen") is None
    assert table.provenance()["license"] == "CC BY 4.0"


def test_no_coefficient_table_is_shipped_with_the_package():
    """Values are not invented here, and an accidental default would be the worst kind."""
    from pathlib import Path

    pkg = Path(__file__).resolve().parents[1] / "ctsegdose_core"
    assert not list(pkg.glob("**/*coefficient*.json"))


# --- geometry ----------------------------------------------------------------------------


def test_the_affine_maps_dicom_lps_to_nifti_ras():
    """A sign error here mirrors left and right kidney without any other symptom."""
    affine = dicom_affine(
        orientation=(1, 0, 0, 0, 1, 0),
        origin=(-250.0, -250.0, -100.0),
        pixel_spacing_mm=(0.7, 0.7),
        slice_direction=(0.0, 0.0, 1.0),
        slice_spacing_mm=5.0,
    )
    assert affine[0, 0] == pytest.approx(-0.7)   # DICOM +x (left) -> NIfTI -x
    assert affine[1, 1] == pytest.approx(-0.7)   # DICOM +y (posterior) -> NIfTI -y
    assert affine[2, 2] == pytest.approx(5.0)    # z (superior) keeps its sign
    assert affine[:3, 3] == pytest.approx([250.0, 250.0, -100.0])


def test_a_descending_series_keeps_its_true_slice_direction():
    """Assuming +z would pair every organ with the tube current of the wrong end."""
    affine = dicom_affine(
        orientation=(1, 0, 0, 0, 1, 0), origin=(0.0, 0.0, 100.0),
        pixel_spacing_mm=(1.0, 1.0), slice_direction=(0.0, 0.0, -1.0), slice_spacing_mm=2.5,
    )
    assert affine[2, 2] == pytest.approx(-2.5)
