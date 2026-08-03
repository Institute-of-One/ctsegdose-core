# ctsegdose-core

**Patient-specific absorbed organ dose from CT tube-current modulation, using
deep-learning organ segmentation and HU-derived density.**

Institute of One, LISIT Co., Ltd., Tokyo, Japan · MIT licence

---

## What this is

A CT dose *index* is not a dose. [ctdose-core][ctdose] reconstructs CTDIvol, DLP, SSDE
and effective dose from DICOM, and takes the tube-current record as far as an
organ-specific **weighted CTDIvol** — still an index, and still tied to a reference
phantom rather than to the patient in the scanner.

This package completes the chain:

```
recorded tube current I(z)   ->  organ-specific weighted CTDIvol      (ctdose-core)
segmented organ masks        ->  organ volume, HU-derived organ mass  (here)
CTDIvol-normalised organ-dose coefficients, scaled patient-specifically
                             ->  absorbed organ dose in mGy           (here)
```

The patient specificity comes from the patient's own anatomy: organs are segmented from
the same series the dose is computed for, and the density used for the mass comes from
the measured Hounsfield units, not from a reference body.

## Status

**Phase 1 — data acquisition — is implemented.** The organ, density and dose layers are
next; the repository will say so plainly here until they exist.

## Phase 1: a balanced multi-vendor sample, without downloading the archive

Handing a collection manifest to NBIA Data Retriever downloads the whole collection.
For the low-dose CT collection that is roughly 600 GB, most of it raw projection data
this work has no use for. This repository never does that.

Instead the archive catalogue is read as metadata, screened, probed, and only then
fetched:

| stage | what moves over the wire | what it decides |
| --- | --- | --- |
| `plan` | series-level JSON (`getSeries`) | modality, vendor, collection, abdominal or not, projection or reconstructed, slice count |
| `verify` | a handful of image *headers* (`getSingleImage`) | per-slice tube current (0018,1151) present and modulated; Hounsfield rescale present; geometry long enough to hold an organ |
| `download` | only the series that passed (`getImage`) | nothing — the choice is already made |

```bash
python tools/select_and_download.py --stage plan
python tools/select_and_download.py --stage verify
python tools/select_and_download.py --stage download --dry-run   # prints the GB first
python tools/select_and_download.py --stage download
```

The candidate list starts from the IORN-004 public-archive survey — 400 series across
four manufacturers and 92 collections, itself built through this API without bulk
download — and is supplemented by direct queries on abdominal CT collections.

**A series with no per-slice tube current is dropped, and the drop is counted per
vendor.** Which vendors omit (0018,1151) is a result to report, not a gap to hide.

## Data policy

No DICOM is redistributed here. Every retrieved series is recorded in
`data/PROVENANCE.json` with its collection, collection DOI, Series Instance UID,
manufacturer, licence and retrieval date, which is what a reader needs to fetch exactly
the same data from the archive. `docs/REPRODUCING_DATA.md` gives the query and the
filters. Only openly licensed (CC BY) TCIA collections are used, and the downloaded
imaging is git-ignored.

## Provenance and anti-fabrication

Every reported figure is re-derived from the rows it summarises by the tests in
`tests/test_results_integrity.py`: the candidate counts from the candidate list, each
keep/drop verdict from the header evidence stored beside it, and the downloaded totals
from the per-series records. A summary edited by hand fails the suite.

## Install

```bash
python -m pip install -e ".[dev]"
```

`pip install -e ".[seg]"` adds TotalSegmentator (Apache-2.0, inference only). It is
always run as a separate process — never inside a long-lived parent such as a Streamlit
app, where process spawning deadlocks on Windows.

## Relationship to ctdose-core

This is a separate contribution with a new output, not a re-publication. It builds on
and cites the open dose engine of [ctdose-core][ctdose], which stops deliberately at the
dose index.

[ctdose]: https://github.com/Institute-of-One/ctdose-core

## Contact

Shuji Yamamoto — yamamoto@lisit.jp — ORCID
[0000-0001-9211-1071](https://orcid.org/0000-0001-9211-1071)
Institute of One, LISIT Co., Ltd., Tokyo, Japan
