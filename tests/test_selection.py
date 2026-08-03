"""The selection rules, exercised offline against synthesised index rows and headers.

None of this needs the network. That is deliberate: the screen decides what gets
downloaded, so it has to be testable without downloading anything.
"""

from __future__ import annotations

import pydicom
import pytest

from ctsegdose_core.nbia import vendor_of
from ctsegdose_core.selection import (
    Candidate,
    candidate_from_index_row,
    candidate_from_survey_row,
    diversify,
    is_abdominal,
    one_series_per_patient,
    read_probe,
    screen,
    screen_reason,
    verdict,
)

CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"


def index_row(**overrides):
    row = {
        "SeriesInstanceUID": "1.2.3.4",
        "StudyInstanceUID": "1.2.3",
        "Modality": "CT",
        "Collection": "C4KC-KiTS",
        "PatientID": "KiTS-00001",
        "Manufacturer": "GE MEDICAL SYSTEMS",
        "ManufacturerModelName": "LightSpeed16",
        "SeriesDescription": "ABDOMEN PORTAL VENOUS",
        "ProtocolName": "6.8 IDI ABD WO/W/DELAY.",
        "StudyDesc": "three_phase__abdomen",
        "ImageCount": 120,
        "FileSize": 60_000_000,
        "LicenseName": "Creative Commons Attribution 3.0 Unported License",
        "LicenseURI": "http://creativecommons.org/licenses/by/3.0/",
        "CollectionURI": "https://doi.org/10.7937/TCIA.2019.IX49E8NX",
    }
    row.update(overrides)
    return row


def candidate(**overrides) -> Candidate:
    cand = candidate_from_index_row(index_row(**overrides))
    assert cand is not None
    return cand


# --- vendor mapping --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("GE MEDICAL SYSTEMS", "GE"),
        ("SIEMENS", "Siemens"),
        ("TOSHIBA", "Canon/Toshiba"),
        ("Canon Medical Systems", "Canon/Toshiba"),
        ("Philips", "Philips"),
        ("Philips Medical Systems", "Philips"),
        ("", None),
        ("NeuroLogica", None),
    ],
)
def test_vendor_of_groups_the_spellings_one_vendor_uses(raw, expected):
    assert vendor_of(raw) == expected


def test_a_series_from_an_unlisted_vendor_is_dropped_not_miscounted():
    assert candidate_from_index_row(index_row(Manufacturer="NeuroLogica")) is None


# --- the metadata screen ---------------------------------------------------------------


def test_an_abdominal_reconstructed_series_survives_the_screen():
    assert screen_reason(candidate()) is None


@pytest.mark.parametrize(
    "description",
    [
        "Full Dose Projections",
        "Low Dose Projection data",
        "sinogram",
        "raw data",
    ],
)
def test_projection_and_raw_series_are_refused_before_any_download(description):
    """The 600 GB this project must never fetch is refused here, on metadata alone."""
    reason = screen_reason(candidate(SeriesDescription=description))
    assert reason is not None and reason.startswith("projection-or-raw")


@pytest.mark.parametrize(
    "description",
    ["Scout", "TOPOGRAM", "Dose Report", "Patient Protocol", "Segmentation"],
)
def test_non_diagnostic_series_are_refused(description):
    reason = screen_reason(candidate(SeriesDescription=description))
    assert reason is not None and reason.startswith("non-diagnostic")


@pytest.mark.parametrize(
    "collection",
    ["QIBA-CT-Liver-Phantom", "Pseudo-PHI-DICOM-Data", "MIDI-B-Curated-Test", "MIDI-B-Synthetic-Test"],
)
def test_phantom_and_de_identification_benchmark_collections_are_refused(collection):
    """A phantom has no patient-specific anatomy, and a de-identification benchmark is
    not a dosimetry cohort -- neither belongs in a patient-specific organ dose sample."""
    reason = screen_reason(candidate(Collection=collection))
    assert reason is not None and reason.startswith("non-patient-collection")


def test_a_short_keyword_does_not_fire_on_a_longer_unrelated_word():
    assert not is_abdominal(
        candidate(SeriesDescription="Screen Capture", ProtocolName="", StudyDesc="")
    )
    assert is_abdominal(candidate(SeriesDescription="CT CAP", ProtocolName="", StudyDesc=""))


def test_a_series_too_short_to_hold_an_organ_is_refused():
    assert screen_reason(candidate(ImageCount=12)).startswith("too-few-images")


def test_a_whole_body_concatenation_is_refused():
    assert screen_reason(candidate(ImageCount=5000)).startswith("too-many-images")


def test_a_head_series_is_refused():
    reason = screen_reason(
        candidate(
            SeriesDescription="HEAD WO", ProtocolName="BRAIN", StudyDesc="head", BodyPartExamined="HEAD"
        )
    )
    assert reason == "not-abdominal"


def test_body_part_examined_wins_over_the_description_when_present():
    assert is_abdominal(candidate(BodyPartExamined="LIVER", SeriesDescription="series 3"))
    assert not is_abdominal(candidate(BodyPartExamined="CHEST", SeriesDescription="abdomen"))


def test_the_description_is_read_when_body_part_is_empty():
    assert is_abdominal(candidate(SeriesDescription="Liver 3-phase"))
    assert not is_abdominal(
        candidate(SeriesDescription="THORAX 1.0 B31f", ProtocolName="", StudyDesc="")
    )


def test_screen_tallies_every_rejection_so_the_counts_partition_the_pool():
    pool = [
        candidate(),
        candidate(SeriesDescription="Full Dose Projections"),
        candidate(SeriesDescription="Scout"),
        candidate(ImageCount=3),
    ]
    kept, excluded = screen(pool)
    assert len(kept) == 1
    assert sum(excluded.values()) == 3
    assert set(excluded) == {"projection-or-raw", "non-diagnostic", "too-few-images"}


# --- sampling shape --------------------------------------------------------------------


def test_two_reconstructions_of_one_patient_count_as_one_subject():
    pool = [
        candidate(SeriesInstanceUID="a", ImageCount=80),
        candidate(SeriesInstanceUID="b", ImageCount=300),
        candidate(SeriesInstanceUID="c", PatientID="KiTS-00002", ImageCount=90),
    ]
    kept = one_series_per_patient(pool)
    assert len(kept) == 2
    assert {c.series_uid for c in kept} == {"b", "c"}  # the longer series wins


def test_a_vendor_quota_is_spread_over_its_collections_not_taken_from_one():
    pool = [
        candidate(SeriesInstanceUID=f"a{i}", PatientID=f"p{i}", Collection="COLL-A") for i in range(5)
    ] + [candidate(SeriesInstanceUID=f"b{i}", PatientID=f"q{i}", Collection="COLL-B") for i in range(5)]
    ordered = diversify(pool, per_vendor=4)
    assert [c.collection for c in ordered[:4]] == ["COLL-A", "COLL-B", "COLL-A", "COLL-B"]


def test_a_series_the_iorn004_survey_already_screened_is_probed_first():
    a = candidate(SeriesInstanceUID="zzz", PatientID="p1")
    a.source = "iorn004-survey"
    b = candidate(SeriesInstanceUID="aaa", PatientID="p2")
    ordered = diversify([b, a], per_vendor=2)
    assert ordered[0].series_uid == "zzz"


def test_a_survey_row_becomes_a_candidate_carrying_its_licence():
    cand = candidate_from_survey_row(
        {
            "vendor": "Philips",
            "manufacturer_raw": "Philips",
            "collection": "TCGA-LIHC",
            "series_uid": "1.9.9",
            "model_name": "Brilliance 64",
            "n_images": 200,
            "licence": "Creative Commons Attribution 3.0 Unported License",
            "licence_url": "http://creativecommons.org/licenses/by/3.0/",
            "body_part": "LIVER",
        }
    )
    assert cand is not None
    assert cand.vendor == "Philips"
    assert cand.licence_uri.startswith("http")
    assert cand.source == "iorn004-survey"
    assert is_abdominal(cand)


# --- the header probe ------------------------------------------------------------------


def slice_header(*, z=0.0, ma=200.0, rescale=True, sop=CT_IMAGE_STORAGE, image_type="ORIGINAL\\PRIMARY\\AXIAL"):
    ds = pydicom.Dataset()
    ds.SOPClassUID = sop
    ds.ImageType = image_type.split("\\")
    ds.BodyPartExamined = "ABDOMEN"
    ds.KVP = 120.0
    ds.SliceThickness = 5.0
    ds.PixelSpacing = [0.7, 0.7]
    ds.ImagePositionPatient = [-250.0, -250.0, z]
    if ma is not None:
        ds.XRayTubeCurrent = ma
    if rescale:
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = -1024.0
    return ds


def modulated_series(n=6):
    currents = [140.0, 210.0, 260.0, 240.0, 190.0, 150.0][:n]
    return [slice_header(z=-300.0 + 60.0 * i, ma=ma) for i, ma in enumerate(currents)]


def probe_of(datasets, *, n_instances=200):
    return read_probe(datasets, series_uid="1.2.3", vendor="GE", n_instances=n_instances)


def test_a_modulated_abdominal_series_with_readable_hu_is_kept():
    probe = probe_of(modulated_series())
    assert probe.has_per_slice_tube_current
    assert probe.tube_current_is_modulated
    assert probe.has_rescale
    assert verdict(probe) == "keep"


def test_a_series_without_the_tube_current_tag_is_dropped_with_its_reason_named():
    """Vendors that omit (0018,1151) are a reported Limitation, not a silent gap."""
    probe = probe_of([slice_header(z=60.0 * i, ma=None) for i in range(6)])
    assert not probe.has_per_slice_tube_current
    assert verdict(probe) == "no-per-slice-tube-current"


def test_tube_current_recorded_on_only_some_slices_cannot_supply_i_of_z():
    datasets = modulated_series()
    del datasets[3].XRayTubeCurrent
    probe = probe_of(datasets)
    assert not probe.has_per_slice_tube_current
    assert verdict(probe) == "no-per-slice-tube-current"


def test_a_fixed_ma_acquisition_is_dropped_because_there_is_no_modulation_to_weight():
    probe = probe_of([slice_header(z=60.0 * i, ma=200.0) for i in range(6)])
    assert probe.tube_current_spread == 0.0
    assert verdict(probe).startswith("tube-current-not-modulated")


def test_rounding_in_the_header_is_not_mistaken_for_modulation():
    probe = probe_of([slice_header(z=60.0 * i, ma=ma) for i, ma in enumerate([200, 200, 201, 200, 200, 201])])
    assert not probe.tube_current_is_modulated


def test_a_series_without_hu_rescale_cannot_feed_the_density_step():
    probe = probe_of([slice_header(z=60.0 * i, ma=ma, rescale=False) for i, ma in enumerate([140, 210, 260, 240, 190, 150])])
    assert verdict(probe) == "no-hu-rescale"


def test_a_localizer_is_dropped_even_though_its_header_looks_complete():
    datasets = modulated_series()
    for ds in datasets:
        ds.ImageType = ["ORIGINAL", "PRIMARY", "LOCALIZER"]
    assert verdict(probe_of(datasets)) == "localizer"


def test_a_non_image_sop_class_is_dropped():
    datasets = modulated_series()
    for ds in datasets:
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.66"  # Raw Data Storage
    assert verdict(probe_of(datasets)).startswith("not-a-reconstructed-image")


def test_a_series_with_too_few_slices_is_dropped():
    assert verdict(probe_of(modulated_series(), n_instances=20)).startswith("too-few-slices")


def test_a_short_scan_is_dropped_even_when_the_slice_count_passes():
    datasets = [slice_header(z=0.5 * i, ma=ma) for i, ma in enumerate([140, 210, 260, 240, 190, 150])]
    for ds in datasets:
        ds.SliceThickness = 0.4
    assert verdict(probe_of(datasets, n_instances=60)).startswith("z-coverage-too-short")


def test_the_probe_extent_is_used_when_it_exceeds_the_nominal_one():
    probe = probe_of(modulated_series())
    assert probe.z_coverage_mm == pytest.approx(300.0)
    assert probe.axial_extent_mm >= probe.z_coverage_mm


def test_a_header_that_contradicts_the_index_body_part_is_dropped():
    datasets = modulated_series()
    for ds in datasets:
        ds.BodyPartExamined = "HEAD"
    assert verdict(probe_of(datasets)) == "not-abdominal-in-header:HEAD"


def test_a_failed_probe_reports_the_failure_rather_than_a_verdict():
    probe = probe_of([])
    assert verdict(probe).startswith("probe-failed")


def test_the_serialised_probe_carries_the_verdict_it_was_judged_on():
    d = probe_of(modulated_series()).to_dict()
    assert d["verdict"] == "keep"
    assert d["tube_current_spread"] > 0
    assert d["has_per_slice_tube_current"] is True
