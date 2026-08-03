"""The provenance record has to be enough to re-fetch the data, because the data is
never shipped. These tests hold it to that."""

from __future__ import annotations

from ctsegdose_core.provenance import SeriesProvenance, document


def record(**overrides) -> SeriesProvenance:
    fields = {
        "vendor": "GE",
        "manufacturer": "GE MEDICAL SYSTEMS",
        "model_name": "LightSpeed16",
        "collection": "C4KC-KiTS",
        "collection_uri": "https://doi.org/10.7937/TCIA.2019.IX49E8NX",
        "patient_id": "KiTS-00001",
        "series_instance_uid": "1.2.3.4",
        "study_instance_uid": "1.2.3",
        "series_description": "ABDOMEN PORTAL VENOUS",
        "body_part": "ABDOMEN",
        "n_instances": 120,
        "n_files_written": 120,
        "size_bytes": 60_000_000,
        "kvp": 120.0,
        "slice_thickness_mm": 5.0,
        "pixel_spacing_mm": [0.7, 0.7],
        "z_coverage_mm": 595.0,
        "tube_current_spread": 0.42,
        "licence": "Creative Commons Attribution 3.0 Unported License",
        "licence_uri": "http://creativecommons.org/licenses/by/3.0/",
        "retrieved_utc": "2026-08-03T00:00:00+00:00",
        "local_path": "data/GE/C4KC-KiTS__KiTS-00001/1.2.3.4",
        "selection_source": "tcia-index",
    }
    fields.update(overrides)
    return SeriesProvenance(**fields)


def build(series):
    return document(
        series=series,
        parameters={"modality": "CT"},
        generated_by="tests",
        package_version="0.1.0",
        started_utc="2026-08-03T00:00:00+00:00",
        finished_utc="2026-08-03T00:10:00+00:00",
    )


def test_every_series_can_be_re_fetched_and_cited_from_the_record_alone():
    doc = build([record()])
    entry = doc["series"][0]
    for key in ("collection", "series_instance_uid", "licence", "retrieved_utc"):
        assert entry[key], f"{key} is what makes the series re-fetchable; it must not be blank"


def test_the_record_states_that_no_imaging_is_redistributed():
    doc = build([record()])
    assert "No DICOM is redistributed" in doc["provenance"]["redistribution"]


def test_the_summary_is_the_sum_of_the_rows_it_summarises():
    series = [record(), record(vendor="Siemens", patient_id="p2", n_instances=200, size_bytes=90_000_000)]
    doc = build(series)
    assert doc["summary"]["n_series"] == 2
    assert doc["summary"]["n_instances"] == 320
    assert doc["summary"]["size_bytes"] == 150_000_000
    assert doc["summary"]["by_vendor"]["GE"]["n_series"] == 1
    assert doc["summary"]["by_vendor"]["Siemens"]["n_instances"] == 200


def test_the_licences_in_force_are_listed_once_each():
    doc = build([record(), record(patient_id="p2")])
    assert doc["summary"]["licences"] == ["Creative Commons Attribution 3.0 Unported License"]
