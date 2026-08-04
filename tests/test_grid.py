"""Slice-grid resolution.

These exist because the naive spacing -- extent divided by file count -- was wrong on
the very first real series it met, and wrong in a way nothing else would have caught:
the masks looked right, the Hounsfield values looked right, and only the organ volumes
were inflated.
"""

from __future__ import annotations

import numpy as np
import pytest

from ctsegdose_core.grid import NonUniformGrid, resolve_grid


def test_a_clean_series_keeps_every_slice_and_its_spacing():
    grid = resolve_grid(np.arange(0.0, 100.0, 2.5))
    assert len(grid.keep) == 40
    assert grid.spacing_mm == pytest.approx(2.5)
    assert not grid.deduplicated


def test_slices_are_ordered_from_inferior_to_superior_whatever_the_file_order():
    grid = resolve_grid([10.0, 0.0, 5.0, 15.0])
    assert grid.keep == [1, 2, 0, 3]
    assert grid.spacing_mm == pytest.approx(5.0)


def test_duplicate_positions_are_dropped_and_the_true_spacing_survives():
    """The defect found on the first real series: 160 files, 119 positions, 5 mm steps.

    Taking (last - first) / (n - 1) gave 3.71 mm -- a 26% error on every voxel volume --
    and stacking the duplicates repeated anatomy on top of that.
    """
    z = np.repeat(np.arange(0.0, 595.0, 5.0), 1).tolist()
    z += z[:41]  # 41 images resent at positions the series already has
    grid = resolve_grid(z)
    assert grid.n_duplicates == 41
    assert len(grid.keep) == 119
    assert grid.spacing_mm == pytest.approx(5.0)
    assert any("repeats anatomy" in w for w in grid.warnings)


def test_the_naive_spacing_would_have_been_wrong_by_a_quarter():
    """Names the size of the error the resolution prevents."""
    z = np.arange(0.0, 595.0, 5.0).tolist()
    z += z[:41]
    naive = (max(z) - min(z)) / (len(z) - 1)
    grid = resolve_grid(z)
    assert naive == pytest.approx(3.71, abs=0.01)
    assert abs(naive - grid.spacing_mm) / grid.spacing_mm > 0.25


def test_the_first_image_at_each_position_is_the_one_kept():
    grid = resolve_grid([0.0, 5.0, 5.0, 10.0])
    assert grid.keep == [0, 1, 3]


def test_positions_are_matched_to_micrometre_rounding_not_exactly():
    grid = resolve_grid([0.0, 5.0, 5.0001, 10.0])
    assert grid.n_duplicates == 1


def test_a_gap_is_refused_because_one_affine_cannot_describe_it():
    with pytest.raises(NonUniformGrid, match="no regular sub-grid"):
        resolve_grid([0.0, 5.0, 10.0, 20.0, 25.0])


def test_two_interleaved_reconstructions_are_recovered_as_one_regular_volume():
    """The real GE case: 459 files whose steps read 0.75, 1.75, 2.5 repeating -- two
    2.5 mm reconstructions offset by 0.75 mm, sharing one Series Instance UID."""
    a = np.arange(0.0, 100.0, 2.5)
    b = a + 0.75
    grid = resolve_grid(np.sort(np.concatenate([a, b])))
    assert grid.spacing_mm == pytest.approx(2.5)
    assert len(grid.keep) == len(a)
    assert any("interleaved" in w for w in grid.warnings)


def test_the_recovered_sub_grid_is_actually_regular():
    a = np.arange(0.0, 60.0, 3.0)
    z = np.sort(np.concatenate([a, a + 1.0]))
    grid = resolve_grid(z)
    steps = np.diff(np.asarray(z)[grid.keep])
    assert np.allclose(steps, steps[0], rtol=0.02)


def test_a_series_in_two_blocks_with_a_gap_is_still_refused():
    """Recovery must not turn a series with a real gap into a short usable one: the
    sub-grid is judged on the extent it covers, not on how many slices it keeps."""
    with pytest.raises(NonUniformGrid, match="no regular sub-grid"):
        resolve_grid([0.0, 2.0, 4.0, 6.0, 50.0, 52.0, 54.0, 56.0])


def test_a_flattened_z_axis_is_refused_rather_than_silently_zero():
    """De-identification sometimes writes one position to every slice."""
    with pytest.raises(NonUniformGrid, match="unrecoverable"):
        resolve_grid([12.0] * 50)


def test_a_descending_series_resolves_to_a_positive_spacing():
    grid = resolve_grid(np.arange(100.0, 0.0, -2.0))
    assert grid.spacing_mm == pytest.approx(2.0)
    assert len(grid.keep) == 50


def test_a_single_slice_falls_back_to_slice_thickness():
    grid = resolve_grid([7.0], fallback_spacing_mm=1.25)
    assert grid.spacing_mm == pytest.approx(1.25)
    with pytest.raises(NonUniformGrid, match="single slice"):
        resolve_grid([7.0])


def test_sub_percent_rounding_in_the_positions_is_tolerated():
    z = np.arange(0.0, 100.0, 2.5) + np.array([0.001, -0.001] * 20)
    assert resolve_grid(z).spacing_mm == pytest.approx(2.5, abs=0.01)


def test_the_grid_reports_what_it_did_for_the_record():
    z = np.arange(0.0, 50.0, 5.0).tolist()
    z += [0.0]
    d = resolve_grid(z).to_dict()
    assert d["n_input_slices"] == 11
    assert d["n_slices_used"] == 10
    assert d["n_duplicate_positions_dropped"] == 1
    assert d["slice_spacing_mm"] == 5.0
