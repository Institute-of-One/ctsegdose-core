"""CTDIvol-normalised organ dose coefficients, and the rule that they must be cited.

An absorbed organ dose in mGy is only as defensible as the coefficient that converts a
dose index into it. Those coefficients are Monte-Carlo results from published studies;
this package computes with them and does not invent them. Nothing here ships a default
table, and every entry point refuses to produce a dose in mGy without one.

That is not caution for its own sake. Transcribed per-scanner coefficient *values* carry
copyright and database-right exposure even where the formulae are free to reimplement --
a lesson the companion project learned the expensive way, and the reason its shipped
table is a CC BY 4.0 source with its SHA-256 recorded. A table loaded here must name its
citation, DOI, licence and source hash, or it does not load.

The size correction is the other half. A coefficient measured on a reference phantom
does not apply unchanged to a patient of a different size; the standard correction is
exponential in the water-equivalent diameter,

.. math::  h_o(D_w) = h_o(D_{w,\\mathrm{ref}}) \\, e^{-\\alpha_o (D_w - D_{w,\\mathrm{ref}})}

with :math:`\\alpha_o` fitted per organ by the same study that supplies :math:`h_o`.
Here the patient's :math:`D_w` is measured over *that organ's own slices*, from the
segmented body contour, so the correction is local to the organ rather than taken from a
single mid-scan slice.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Provenance a coefficient table must carry before it may be used.
REQUIRED_PROVENANCE = ("citation", "doi", "license", "license_url", "source_sha256")


class MissingCoefficients(RuntimeError):
    """No licence-cleared coefficient table is available for this computation."""


class InvalidCoefficientTable(ValueError):
    """A table was supplied but does not carry what makes it auditable."""


@dataclass(frozen=True)
class OrganCoefficient:
    """One organ's CTDIvol-normalised dose coefficient and its size dependence.

    Attributes:
        h_ref: organ dose per unit CTDIvol, mGy/mGy, at ``d_w_ref``.
        alpha_per_cm: exponential size-correction rate, 1/cm.
        d_w_ref_cm: water-equivalent diameter the coefficient is referenced to.
        relative_uncertainty: 1-sigma relative standard uncertainty of ``h_ref`` as
            reported by the source. Required: a coefficient without a stated uncertainty
            cannot carry one into the result, and a dose reported without uncertainty
            invites being read as exact.
    """

    organ: str
    h_ref: float
    alpha_per_cm: float
    d_w_ref_cm: float
    relative_uncertainty: float
    scan_region: str = "abdomen"
    kvp: float | None = None

    def at(self, d_w_cm: float) -> float:
        """Size-corrected coefficient for a patient of water-equivalent diameter ``d_w_cm``."""
        if d_w_cm <= 0:
            raise ValueError("water-equivalent diameter must be positive")
        return self.h_ref * math.exp(-self.alpha_per_cm * (d_w_cm - self.d_w_ref_cm))

    def to_dict(self) -> dict[str, Any]:
        return {
            "organ": self.organ,
            "h_ref_mgy_per_mgy": self.h_ref,
            "alpha_per_cm": self.alpha_per_cm,
            "d_w_ref_cm": self.d_w_ref_cm,
            "relative_uncertainty": self.relative_uncertainty,
            "scan_region": self.scan_region,
            "kvp": self.kvp,
        }


@dataclass(frozen=True)
class CoefficientTable:
    """A set of organ coefficients that knows where it came from."""

    coefficients: dict[str, OrganCoefficient]
    citation: str
    doi: str
    license: str
    license_url: str
    source_sha256: str
    note: str = ""

    def get(self, organ: str) -> OrganCoefficient | None:
        return self.coefficients.get(organ)

    def provenance(self) -> dict[str, Any]:
        return {
            "citation": self.citation,
            "doi": self.doi,
            "license": self.license,
            "license_url": self.license_url,
            "source_sha256": self.source_sha256,
            "note": self.note,
            "n_organs": len(self.coefficients),
            "size_correction": "h(D_w) = h_ref * exp(-alpha * (D_w - D_w_ref))",
        }


def sha256_of(path: Path | str) -> str:
    """SHA-256 of a source file, so a shipped table can be tied to the file it came from."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_table(path: Path | str) -> CoefficientTable:
    """Load a coefficient table, refusing one that cannot be audited.

    The file is JSON with a ``provenance`` block carrying every field in
    :data:`REQUIRED_PROVENANCE`, and a ``coefficients`` list of organ entries.
    """
    path = Path(path)
    if not path.exists():
        raise MissingCoefficients(
            f"no coefficient table at {path}. Absorbed organ dose in mGy requires "
            "CTDIvol-normalised coefficients from a published Monte-Carlo study, with a "
            "licence that permits redistribution. Supply one with --coefficients; the "
            "pipeline reports organ-specific weighted CTDIvol, organ volume and organ "
            "mass without it."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    prov = payload.get("provenance", {})
    missing = [k for k in REQUIRED_PROVENANCE if not str(prov.get(k, "")).strip()]
    if missing:
        raise InvalidCoefficientTable(
            f"{path.name} is missing required provenance: {', '.join(missing)}. A "
            "coefficient table without its citation, licence and source hash cannot be "
            "shipped or reported from."
        )

    rows = payload.get("coefficients", [])
    if not rows:
        raise InvalidCoefficientTable(f"{path.name} carries no coefficients")

    coefficients: dict[str, OrganCoefficient] = {}
    for row in rows:
        try:
            coefficients[row["organ"]] = OrganCoefficient(
                organ=row["organ"],
                h_ref=float(row["h_ref_mgy_per_mgy"]),
                alpha_per_cm=float(row["alpha_per_cm"]),
                d_w_ref_cm=float(row["d_w_ref_cm"]),
                relative_uncertainty=float(row["relative_uncertainty"]),
                scan_region=str(row.get("scan_region", "abdomen")),
                kvp=(float(row["kvp"]) if row.get("kvp") is not None else None),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidCoefficientTable(f"{path.name}: bad coefficient row {row!r}") from exc
    return CoefficientTable(
        coefficients=coefficients,
        citation=str(prov["citation"]),
        doi=str(prov["doi"]),
        license=str(prov["license"]),
        license_url=str(prov["license_url"]),
        source_sha256=str(prov["source_sha256"]),
        note=str(prov.get("note", "")),
    )
