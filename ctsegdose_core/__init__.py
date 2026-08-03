"""ctsegdose-core: patient-specific absorbed organ dose from CT tube-current modulation.

The chain this package completes:

    recorded tube current I(z)  ->  organ-specific weighted CTDIvol   (ctdose-core)
    segmented organ masks       ->  organ volume and HU-derived mass  (here)
    CTDIvol-normalised organ-dose coefficients, scaled patient-specifically
                                ->  absorbed organ dose in mGy        (here)

ctdose-core (IORN-004) stops at the organ-specific weighted CTDIvol, which is a dose
*index*. This package takes that index to an absorbed organ dose made patient-specific
by the patient's own segmented anatomy and HU-derived density.

Phase 1, implemented here, is the data layer: selecting and retrieving a small, balanced
multi-vendor abdominal sample from a public archive without downloading the archive.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
