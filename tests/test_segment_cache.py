"""The mask cache must notice when the volume it was built from has changed."""

from __future__ import annotations

import numpy as np

from ctsegdose_core.segment import dicom_affine, fingerprint


def a_volume(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 100.0, size=(8, 16, 16)).astype(np.float32)


def an_affine(spacing_z=5.0):
    return dicom_affine((1, 0, 0, 0, 1, 0), (-250.0, -250.0, -100.0), (0.7, 0.7), (0, 0, 1), spacing_z)


def test_the_same_volume_and_geometry_fingerprint_the_same():
    v, a = a_volume(), an_affine()
    assert fingerprint(v, a) == fingerprint(v.copy(), a.copy())


def test_reversing_the_slice_order_changes_the_fingerprint():
    """The defect this guards: canonicalising head-first series invalidated the cache,
    and stale masks would have mirrored every organ along z with no error at all."""
    v, a = a_volume(), an_affine()
    assert fingerprint(v[::-1], a) != fingerprint(v, a)


def test_a_different_slice_spacing_changes_the_fingerprint():
    v = a_volume()
    assert fingerprint(v, an_affine(5.0)) != fingerprint(v, an_affine(2.5))


def test_a_different_matrix_changes_the_fingerprint():
    a = an_affine()
    assert fingerprint(a_volume()[:, :8, :8], a) != fingerprint(a_volume(), a)


def test_different_pixel_data_on_the_same_grid_changes_the_fingerprint():
    a = an_affine()
    assert fingerprint(a_volume(1), a) != fingerprint(a_volume(2), a)
