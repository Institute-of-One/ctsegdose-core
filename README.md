# ctsegdose-core

**Anatomy-weighted CTDIvol from routine CT metadata: organ-specific longitudinal
tube-current modulation, using deep-learning organ segmentation.**

Institute of One, LISIT Co., Ltd., Tokyo, Japan · MIT licence

---

## What this is

A CT dose *index* is not a dose, and one value per series cannot express how the tube
current varied along the patient. This package computes an **anatomy-weighted CTDIvol
index**: the whole-scan CTDIvol scaled by a dimensionless, organ-specific weight formed
from the recorded per-slice tube current over that organ's segmented longitudinal extent.

```
recorded tube current I(z)  +  segmented organ masks
      ->  organ-specific modulation weight  w_o          (dimensionless)
      ->  anatomy-weighted CTDIvol index = CTDIvol · w_o (mGy)
      +   organ volume and attenuation-derived organ mass
```

**The quantity is not new.** It is the organ-specific weighted CTDIvol of Khatonabadi et
al. (2013) and Tian et al. (2015). What this package provides is an open, end-to-end
implementation of it that runs from routine archived DICOM alone, with organ contours
obtained at inference.

## What this is not

**It is not an estimate of absorbed organ dose.** The index describes *longitudinal*
tube-current modulation only. It does not account for scattered radiation, irradiation
originating outside an organ's segmented extent, angular modulation, organ depth and
attenuation, or radiation transport.

Converting it to a dose in milligray requires CTDIvol-normalised organ-dose coefficients
from a published Monte-Carlo study, under a licence permitting redistribution. **No such
table is shipped here**: `ctsegdose_core/coefficients.py` refuses to emit a dose without
one, and refuses to load a table lacking its citation, DOI, licence and source hash. See
*Coefficients* below.

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

## Phase 2: the organ layer

```bash
python tools/run_organ_dose.py --per-vendor 1 --full-resolution --device gpu \
    --python .venv-gpu/Scripts/python.exe        # cross-vendor check first
python tools/check_segmentation.py --tag 1.5mm   # anatomical sanity screens
python tools/run_organ_dose.py --full-resolution --device gpu \
    --python .venv-gpu/Scripts/python.exe        # the whole sample
```

Segmentation is TotalSegmentator (Apache-2.0, inference only), always in a **separate
child process** — nnU-Net spawns its own workers, and spawning those from a long-lived
parent leaks runaway processes on Windows. `--python` points at the CUDA environment, so
inference runs on the GPU while the analysis stays in the environment that has pydicom
and ctdose-core. Which interpreter, which torch and which GPU ran a mask is recorded
with it.

Three things this layer does that are easy to get wrong, and are therefore checked:

**The slice grid is resolved, not assumed.** Taking the spacing as
`(z_last − z_first) / (n − 1)` was wrong on the first real series it met: 160 files but
119 distinct positions, so the spacing came out 26% too small *and* the stack repeated
anatomy. Both errors inflate organ volume and neither leaves any other trace.

**The slice order is canonicalised.** On three of the four vendors in this sample, Slice
Location (0020,1041) runs opposite to Image Position (Patient), so the volume arrives
head-first. The segmentation is unaffected — the affine is built from patient
coordinates — but the array index stops meaning "towards the head". Index 0 is now
always the most inferior slice, and the tube current is reordered with it.

**Truncated organs are flagged.** An organ whose mask reaches the first or last slice
continues beyond the scan: its mass is the mass of the scanned part, and its modulation
weighting describes only the exposed part. Neither is the organ's, so it is marked and
excluded from whole-organ comparisons rather than reported as a small organ.

`tools/check_segmentation.py` screens each series against facts of gross anatomy — the
left kidney left of the right, the spleen left of the liver, the adrenals above the
kidneys, solid-organ masses within a wide band of the ICRP 89 reference adult. These
catch the failures that are otherwise silent: a mirrored or inverted segmentation
produces entirely plausible volumes, masses and Hounsfield values, and weights every
organ by the tube current of the wrong part of the patient.

## Coefficients, and why none are shipped

Converting the organ-specific weighted CTDIvol to an absorbed dose needs
`h_o(D_w) = h_ref · exp(−α · (D_w − D_w,ref))` with organ-specific `h_ref` and `α` from a
published Monte-Carlo study. Those values are **not** invented here, and no default
table is shipped: a coefficient table must carry citation, DOI, licence, licence URL and
source SHA-256 or it does not load.

That strictness is inherited. The companion project shipped per-scanner values of
CT-Expo lineage, and replacing them cost a re-derivation and a corrected archive —
formulae are free to reimplement, transcribed values are not. Until a licence-cleared
source is settled, the pipeline reports the index layer and says so in every record.

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
