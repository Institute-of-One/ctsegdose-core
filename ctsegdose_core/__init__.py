"""ctsegdose-core: anatomy-weighted CTDIvol from routine CT metadata.

What this computes, and what it does not:

    recorded tube current I(z)  +  segmented organ masks
      ->  organ-specific modulation weight w_o            (dimensionless)
      ->  anatomy-weighted CTDIvol index = CTDIvol * w_o  (mGy)
      +   organ volume and attenuation-derived organ mass

The quantity is not new: it is the organ-specific weighted CTDIvol of Khatonabadi et al.
(2013) and Tian et al. (2015). What this package provides is an open, end-to-end
implementation of it that runs from routine archived DICOM alone, with organ contours
obtained at inference.

**It is not an estimate of absorbed organ dose.** The index describes *longitudinal*
tube-current modulation only, and does not account for scattered radiation, irradiation
originating outside an organ's segmented extent, angular modulation, organ depth and
attenuation, or radiation transport. Converting it to milligray needs CTDIvol-normalised
organ-dose coefficients, which :mod:`ctsegdose_core.coefficients` requires the caller to
supply, with citation, DOI, licence and source hash, and which this package never ships.

The whole-scan CTDIvol the weight scales -- recorded, or reconstructed from acquisition
physics -- comes from ctdose-core.
"""

from __future__ import annotations

__version__ = "0.1.2"

__all__ = ["__version__"]
