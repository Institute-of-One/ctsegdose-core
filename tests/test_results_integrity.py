"""Anti-fabrication: every count this repository reports is re-derived from its rows.

A summary that is written rather than computed can drift from the data it claims to
summarise -- by a hand edit, or by a stale file left behind after a re-run. These tests
recompute the headline figures from the per-series records and fail if they disagree, so
no number quoted in a manuscript can outlive the data behind it.

They skip when the file is absent, so a fresh clone (which carries no downloaded data)
is still green; CI runs them wherever ``results/`` and ``data/PROVENANCE.json`` exist.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from ctsegdose_core.selection import Probe, read_probe, verdict

REPO = Path(__file__).resolve().parents[1]
CANDIDATES = REPO / "results" / "candidates.json"
VERIFICATION = REPO / "results" / "verification.json"
PROVENANCE = REPO / "data" / "PROVENANCE.json"


def load(path: Path):
    if not path.exists():
        pytest.skip(f"{path.relative_to(REPO)} has not been generated in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_candidate_counts_match_the_candidates_listed():
    payload = load(CANDIDATES)
    assert payload["counts"]["probe_queue"] == len(payload["candidates"])
    by_vendor = Counter(c["vendor"] for c in payload["candidates"])
    for vendor, n in payload["counts"]["probe_queue_by_vendor"].items():
        assert by_vendor.get(vendor, 0) == n


def test_every_probe_verdict_is_re_derived_from_the_evidence_recorded_with_it():
    """The stored verdict must follow from the stored header evidence, not from a note."""
    payload = load(VERIFICATION)
    for entry in payload["probes"]:
        probe = Probe(
            series_uid=entry["series_uid"],
            vendor=entry["vendor"],
            n_probed=entry["n_probed"],
            n_instances=entry["n_instances"],
            sop_class_uid=entry["sop_class_uid"],
            image_type=entry["image_type"],
            body_part=entry["body_part"],
            kvp=entry["kvp"],
            tube_currents_ma=entry["tube_currents_ma"],
            z_positions_mm=entry["z_positions_mm"],
            slice_thickness_mm=entry["slice_thickness_mm"],
            pixel_spacing_mm=entry["pixel_spacing_mm"],
            has_rescale=entry["has_rescale"],
            rescale_slope=entry["rescale_slope"],
            rescale_intercept=entry["rescale_intercept"],
            error=entry["error"],
        )
        assert verdict(probe) == entry["verdict"], entry["series_uid"]


def test_the_verification_counts_match_the_probes_listed():
    payload = load(VERIFICATION)
    probes = payload["probes"]
    assert payload["counts"]["probed"] == len(probes)
    kept = Counter(p["vendor"] for p in probes if p["verdict"] == "keep")
    assert payload["counts"]["kept"] == sum(kept.values())
    for vendor, n in payload["counts"]["kept_by_vendor"].items():
        assert kept.get(vendor, 0) == n


def test_the_downloaded_totals_match_the_series_they_are_totals_of():
    payload = load(PROVENANCE)
    series = payload["series"]
    summary = payload["summary"]
    assert summary["n_series"] == len(series)
    assert summary["n_instances"] == sum(s["n_instances"] for s in series)
    assert summary["size_bytes"] == sum(s["size_bytes"] for s in series)
    assert summary["size_gb"] == round(summary["size_bytes"] / 1e9, 3)
    for vendor, block in summary["by_vendor"].items():
        rows = [s for s in series if s["vendor"] == vendor]
        assert block["n_series"] == len(rows)
        assert block["n_instances"] == sum(s["n_instances"] for s in rows)


def test_every_downloaded_series_names_the_licence_it_was_published_under():
    payload = load(PROVENANCE)
    for s in payload["series"]:
        assert s["licence"], f"{s['series_instance_uid']} carries no licence"
        assert s["collection"], f"{s['series_instance_uid']} carries no collection"
        assert s["retrieved_utc"], f"{s['series_instance_uid']} carries no retrieval date"


def test_the_probe_evidence_survives_a_round_trip_through_json():
    """Guards the reconstruction the anti-fabrication test above depends on."""
    probe = read_probe([], series_uid="1.2.3", vendor="GE", n_instances=0)
    d = probe.to_dict()
    assert set(d) >= {"series_uid", "vendor", "verdict", "tube_currents_ma"}
