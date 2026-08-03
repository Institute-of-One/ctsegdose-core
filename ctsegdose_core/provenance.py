"""Provenance records for everything this repository derives or retrieves.

No imaging is redistributed here, so the provenance record *is* the dataset as far as a
reader is concerned: it has to name the archive, the collection, the collection DOI, the
Series Instance UID, the licence that series was published under and the date it was
retrieved, precisely enough that the same series can be fetched again and the same
numbers reproduced.

The same discipline applies to derived values later in the chain (density calibration,
organ mass, dose coefficients): a value ships with the formula, its inputs, its
assumptions and its source, never on its own.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

ARCHIVE = "The Cancer Imaging Archive (TCIA), public NBIA REST API"

REDISTRIBUTION_NOTE = (
    "No DICOM is redistributed by this repository. Each series is identified by its "
    "Series Instance UID, collection and licence so that a reader retrieves it from the "
    "archive directly; see docs/REPRODUCING_DATA.md."
)


@dataclass
class SeriesProvenance:
    """Everything needed to re-fetch one series and to cite it correctly."""

    vendor: str
    manufacturer: str
    model_name: str
    collection: str
    collection_uri: str
    patient_id: str
    series_instance_uid: str
    study_instance_uid: str
    series_description: str
    body_part: str
    n_instances: int
    n_files_written: int
    size_bytes: int
    kvp: float | None
    slice_thickness_mm: float | None
    pixel_spacing_mm: list[float] = field(default_factory=list)
    z_coverage_mm: float | None = None
    tube_current_spread: float | None = None
    licence: str = ""
    licence_uri: str = ""
    retrieved_utc: str = ""
    local_path: str = ""
    selection_source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def document(
    *,
    series: list[SeriesProvenance],
    parameters: dict[str, Any],
    generated_by: str,
    package_version: str,
    started_utc: str,
    finished_utc: str,
    exclusions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the ``data/PROVENANCE.json`` document.

    ``parameters`` records the query that produced the sample, so the selection is
    reproducible and not merely described.
    """
    by_vendor: dict[str, dict[str, Any]] = {}
    for s in series:
        block = by_vendor.setdefault(
            s.vendor, {"n_series": 0, "n_instances": 0, "size_bytes": 0, "collections": []}
        )
        block["n_series"] += 1
        block["n_instances"] += s.n_instances
        block["size_bytes"] += s.size_bytes
        if s.collection not in block["collections"]:
            block["collections"].append(s.collection)

    return {
        "provenance": {
            "layer": "1 - patient data acquisition for patient-specific organ dose",
            "generated_by": generated_by,
            "ctsegdose_core_version": package_version,
            "archive": ARCHIVE,
            "access_started_utc": started_utc,
            "access_finished_utc": finished_utc,
            "redistribution": REDISTRIBUTION_NOTE,
        },
        "parameters": parameters,
        "summary": {
            "n_series": len(series),
            "n_instances": sum(s.n_instances for s in series),
            "size_bytes": sum(s.size_bytes for s in series),
            "size_gb": round(sum(s.size_bytes for s in series) / 1e9, 3),
            "by_vendor": by_vendor,
            "licences": sorted({s.licence for s in series if s.licence}),
            "collections": sorted({s.collection for s in series if s.collection}),
        },
        "exclusions": exclusions or {},
        "series": [s.to_dict() for s in series],
    }
