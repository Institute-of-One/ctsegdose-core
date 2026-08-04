"""Pixel padding decoding.

These exist because a mis-decoded padding value blanked four of the ten GE series in
this sample to a uniform block of air, and the only symptom anywhere was that the
segmenter returned empty masks.
"""

from __future__ import annotations

import numpy as np
import pydicom
import pytest

from ctsegdose_core.volume import AIR_HU, signed_padding_value, slice_hu


def a_slice(*, padding=None, padding_vr="US", pixel_representation=1, limit=None):
    """A 64x64 slice: a disc of soft tissue on a background of raw -2000."""
    meta = pydicom.dataset.FileMetaDataset()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds = pydicom.dataset.FileDataset("slice.dcm", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = pydicom.uid.CTImageStorage
    ds.Rows = ds.Columns = 64
    ds.PixelRepresentation = pixel_representation
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.RescaleSlope = 1
    ds.RescaleIntercept = -1024

    raw = np.full((64, 64), -2000, dtype=np.int16)
    yy, xx = np.mgrid[0:64, 0:64]
    body = (yy - 32) ** 2 + (xx - 32) ** 2 < 20**2
    raw[body] = 1074  # -> +50 HU
    ds.PixelData = raw.tobytes()
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    if padding is not None:
        ds.add_new((0x0028, 0x0120), padding_vr, padding)
    if limit is not None:
        ds.add_new((0x0028, 0x0121), padding_vr, limit)
    return ds, body


# --- the decoding rule -------------------------------------------------------------------


def test_an_unsigned_encoded_padding_value_on_signed_data_is_read_as_negative():
    """63536 with VR US on signed pixels is -2000, which is what GE meant."""
    assert signed_padding_value(63536, pixel_representation=1) == -2000.0


def test_the_same_value_on_unsigned_data_is_left_alone():
    assert signed_padding_value(63536, pixel_representation=0) == 63536.0


def test_an_ordinary_negative_padding_value_passes_through():
    assert signed_padding_value(-2000, pixel_representation=1) == -2000.0


def test_a_missing_padding_value_is_not_invented():
    assert signed_padding_value(None, pixel_representation=1) is None


# --- what it does to an image ------------------------------------------------------------


def test_the_defect_the_whole_module_exists_for():
    """Read literally, 63536 blanks every voxel. Read correctly, it blanks the corners."""
    ds, body = a_slice(padding=63536, padding_vr="US")
    hu, note = slice_hu(ds)
    assert note == ""
    assert hu[body].mean() == pytest.approx(50.0)
    assert hu[~body].min() == AIR_HU
    assert hu.max() > AIR_HU, "the patient must survive padding removal"


def test_padding_declared_the_correct_way_round_works_the_same():
    ds, body = a_slice(padding=-2000, padding_vr="SS")
    hu, note = slice_hu(ds)
    assert note == ""
    assert hu[body].mean() == pytest.approx(50.0)
    assert hu[~body].min() == AIR_HU


def test_a_slice_with_no_padding_attribute_is_untouched():
    ds, body = a_slice()
    hu, note = slice_hu(ds)
    assert note == ""
    assert hu[~body].min() == pytest.approx(-3024.0)


def test_a_padding_rule_that_would_blank_the_whole_slice_is_refused():
    """The general defence: the next encoding quirk will not be the one already fixed."""
    ds, body = a_slice(padding=32000, padding_vr="SS", pixel_representation=1)
    hu, note = slice_hu(ds)
    assert "was not applied" in note
    assert hu[body].mean() == pytest.approx(50.0), "the image must survive a bad rule"


def test_a_padding_range_limit_blanks_the_range_not_just_the_value():
    ds, body = a_slice(padding=63536, padding_vr="US", limit=63036)
    hu, note = slice_hu(ds)
    assert note == ""
    assert hu[~body].min() == AIR_HU
    assert hu[body].mean() == pytest.approx(50.0)


def test_padded_voxels_land_on_air_so_they_carry_no_mass():
    ds, _ = a_slice(padding=63536, padding_vr="US")
    hu, _ = slice_hu(ds)
    assert set(np.unique(hu[hu < 0])) == {AIR_HU}
