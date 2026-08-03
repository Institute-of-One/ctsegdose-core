# Reproducing the data

No imaging is redistributed by this repository. This document is what replaces it: the
exact query, the exact filters and the exact per-series identifiers needed to retrieve
the same data from The Cancer Imaging Archive (TCIA) and reproduce the results.

Everything below runs against TCIA's public NBIA REST API. No credentials, no private
endpoints, nothing written back.

---

## The rule that shapes the whole procedure

**The archive is never bulk-downloaded.** Handing a `.tcia` manifest to NBIA Data
Retriever fetches an entire collection; for `LDCT-and-Projection-data` that is roughly
600 GB, dominated by raw projection data this work has no use for. Instead:

1. read the catalogue as **metadata** (`getSeries` — series-level JSON, no pixel data);
2. **screen** on that metadata;
3. **probe** a handful of image *headers* per surviving candidate (`getSingleImage`);
4. **download** only the series that passed (`getImage`).

The projection data is refused at step 2, before a byte of it is requested.

---

## Reproduce it

```bash
git clone https://github.com/Institute-of-One/ctsegdose-core
cd ctsegdose-core
python -m pip install -e ".[dev]"

python tools/select_and_download.py --stage plan       # writes results/candidates.json
python tools/select_and_download.py --stage verify     # writes results/verification.json
python tools/select_and_download.py --stage download   # writes data/PROVENANCE.json + DICOM
```

Add `--dry-run` to the download stage to print the series and the gigabytes without
transferring anything. Fetched objects are cached under `.tcia_work/` (git-ignored), so
a re-run costs no bandwidth.

To retrieve exactly the series this project used rather than re-running the selection,
take the `series_instance_uid` values from `data/PROVENANCE.json` and fetch each one:

```python
from ctsegdose_core.nbia import NbiaClient

client = NbiaClient()
client.download_series("<series_instance_uid>", "data/<vendor>/<subject>/<uid>")
```

The archive is a living resource: collections gain and lose series over time, so a
re-run of the *selection* may not reproduce the identical sample. Fetching the recorded
UIDs does.

---

## Stage 1 — the query (`--stage plan`)

**Source A, primary: the IORN-004 public-archive survey.**
`../ctdose-core/results/survey.json` catalogues 400 CT series across the four
manufacturers and 92 collections, with per-series licence, body part and header
attribute presence — itself built over this same API without bulk download. Its
abdominal rows seed the candidate list and are probed first, and the collections they
fall in are added to the set of collections queried.

**Source B, supplement: direct index queries.** `getSeries(Collection=…, Modality="CT")`
over the abdominal collections listed in `DEFAULT_COLLECTIONS` in
`tools/select_and_download.py`. The exact list used for a given run is recorded in
`results/candidates.json` under `parameters.collections_queried`, together with any
collection that could not be reached.

**Manufacturer grouping.** TCIA indexes the DICOM Manufacturer verbatim, so one vendor
appears under several spellings (`GE MEDICAL SYSTEMS`, `GE HEALTHCARE`, …). The mapping
to the four vendor labels is `VENDOR_ALIASES` in `ctsegdose_core/nbia.py`. A series from
any other manufacturer is dropped rather than counted as one of the four.

### The metadata screen

Applied in this order; a candidate is recorded against the first rule it fails, so the
exclusion counts in `results/candidates.json` partition the pool rather than
double-counting it.

| rule | rejects |
| --- | --- |
| non-patient collection | imaging phantoms; de-identification benchmark collections (synthetic or pseudo-PHI images) |
| projection or raw | descriptions containing `projection`, `sinogram`, `raw data` — **this is where the 600 GB is refused** |
| non-diagnostic | localiser/scout/topogram/surview, dose reports, patient protocols, screen captures, segmentation objects |
| too few images | fewer than 40 instances: too short to contain a whole organ |
| too many images | more than 1200 instances: whole-body or multi-phase concatenations |
| not abdominal | `BodyPartExamined` outside the abdominal list when present; otherwise no abdominal keyword in the series/protocol/study description |

Two further shaping steps, both recorded in `results/candidates.json`:

- **one series per patient per collection** — two reconstructions of one acquisition are
  not two subjects; the longest series wins;
- **collection round-robin** — a vendor's quota is drawn across as many of its
  collections as the index offers, so vendor is not confounded with a single site or
  protocol.

## Stage 2 — the header probe (`--stage verify`)

For each candidate, `getSOPInstanceUIDs` gives the instance list (metadata, free), and
six instances spread across it are fetched with `getSingleImage` and read header-only.
This is a few megabytes per series instead of hundreds, and it answers the three
questions that decide usability:

| requirement | tag | why it is required |
| --- | --- | --- |
| per-slice tube current, present on **every** probed slice | (0018,1151) | I(z) is the entire input to the modulation weighting; a series carrying it on some slices cannot supply it |
| tube current actually modulated | — | peak-to-peak / mean ≥ 0.02, so header rounding is not mistaken for modulation. A fixed-mA acquisition is a different acquisition, not a weaker case of the same one |
| readable Hounsfield units | (0028,1052), (0028,1053) | the HU → density conversion needs the rescale to be defined |

alongside: reconstructed-image SOP class, not a localiser, at least 40 instances, and an
axial extent of at least 120 mm.

`results/verification.json` stores the evidence — every probed tube current, the z
positions, the rescale — beside the verdict it produced, and
`tests/test_results_integrity.py` re-derives each verdict from that evidence, so a
verdict cannot drift from what the header actually said.

Probing stops for a vendor once its quota is filled, so the number of series probed
depends on how early the usable ones appear.

**Series without per-slice tube current are dropped and the drop is counted per vendor.**
Which vendors omit (0018,1151) is reported as a Limitation; it is not a failure of the
method and it is not hidden.

## Stage 3 — the download (`--stage download`)

Only the series that returned `keep` are fetched, one `getImage` call each, into
`data/<vendor>/<collection>__<patient>/<series-uid>/`. That directory is git-ignored.

`data/PROVENANCE.json` — which *is* committed — records for every series: collection,
collection DOI, Series Instance UID, Study Instance UID, patient identifier as published
by the archive, manufacturer and model, kVp, slice thickness, pixel spacing, axial
extent, tube-current spread, instance count, bytes on disk, **licence and licence URI**,
and the **retrieval timestamp in UTC**.

---

## Licensing and attribution

Only openly licensed TCIA collections are used, and each series' licence travels with it
into `data/PROVENANCE.json`. TCIA data are subject to the TCIA Data Usage Policy and to
each collection's own licence. Any publication using this sample must reproduce the
collection name, the licence and the access date; `data/PROVENANCE.json`
(`summary.licences`, `summary.collections`) carries exactly what a data-availability
statement needs.
