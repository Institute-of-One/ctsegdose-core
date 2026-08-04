# Patient-specific, multi-vendor organ-level CT dose characterisation from routine metadata using deep-learning segmentation

**Shuji Yamamoto**
Institute of One, LISIT Co., Ltd., Tokyo, Japan
ORCID 0000-0001-9211-1071 · yamamoto@lisit.jp

**Target venue:** Physica Medica / Medical Physics / Radiological Physics and Technology
(full Original Article)

**Status:** skeleton. Section content is sketched, not drafted. Every number below is
re-derived from `results/analysis_1.5mm.json` by `tests/test_manuscript_consistency.py`;
none is typed by hand, and a figure that drifts fails the suite.

---

## Abstract (structured, ~250 words — to draft)

**Purpose.** Whole-scan dose indices describe the acquisition, not the patient, and not
the organ. We characterise CT dose at the organ level from data a scanner already
records — the per-slice tube current — combined with organ anatomy segmented from the
same series, and we measure how far that can be taken across four manufacturers using
only openly licensed inputs.

**Methods.** Forty abdominal CT series (ten per manufacturer, 21 collections, 23 scanner
models) were selected from The Cancer Imaging Archive by a metadata-first procedure that
never downloads a collection. Organs were segmented with TotalSegmentator
(inference-only); organ mass was derived from the patient's own Hounsfield units through
a documented density calibration; and an organ-specific weighted CTDIvol was formed from
the recorded per-slice tube current over each organ's own extent.

**Results.** 455 organ records. Segmented mass tracked the ICRP 89 reference adult for
the liver (median ratio 1.06) and kidneys (1.15, 1.17), with two systematic offsets
reported rather than corrected: pancreas 0.61 and spleen 1.72. A whole-scan dose index
was recorded in 29 of 40 series, reconstructable in 5, and **absent in 6 — all GE**
(GE 0/10 recorded vs 29/30 for the other manufacturers, Fisher exact p ≈ 1e-08).

**Conclusions.** [to draft]

---

## 1. Introduction

- A dose index is not an organ dose, and a phantom is not a patient. Two gaps, usually
  conflated.
- Tube current modulation is recorded per slice in routine DICOM and is almost never
  used at the organ level, because using it needs anatomy.
- Deep-learning segmentation makes the anatomy free at inference time. What it does not
  make free is the **conversion to absorbed dose**, which requires Monte-Carlo
  coefficients that are not openly licensed. Naming that boundary precisely is part of
  the contribution.
- Prior work: whole-scan dose surveillance and the reconstruction of missing indices
  [IORN-004]; size-specific dose estimates (AAPM TG-204/220); organ dose coefficient
  libraries (Turner 2011, Tian 2013), all under restrictive licences.
- **Contributions.** (i) an anatomy-aware, patient-specific organ dose index computed
  end to end from routine metadata; (ii) its multi-vendor calibration against a
  published reference; (iii) the first organ-level measurement of dose-index
  availability by manufacturer; (iv) the practical limits — truncation, flat modulation —
  that an organ-level study must handle; (v) an openly licensed, provenance-carrying
  implementation, and an explicit statement of where openness runs out.

## 2. Materials and methods

### 2.1 Data selection without bulk download
Metadata-first over the NBIA REST API: index → screen → probe → download. 47,181 CT
series read as metadata; 398 projection/raw series refused at the screen; 62 series
probed at six image headers each; 40 kept. No DICOM is redistributed; provenance per
series (collection, DOI, UID, licence, retrieval date). → `docs/REPRODUCING_DATA.md`

### 2.2 Inclusion criteria
Per-slice tube current (0018,1151) present on every probed slice **and** genuinely
modulated; readable Hounsfield rescale; ≥ 40 slices; ≥ 120 mm axial extent; abdominal
anatomy. Balanced to ten series per manufacturer.

### 2.3 Geometry: the slice grid
One slice per longitudinal position; spacing from the median step, not from
extent/(n−1); slices canonicalised to ascending patient z. Series that are two
interleaved reconstructions under one UID are resolved to the largest regular sub-grid.
*(Section to state why: each of these silently multiplies every reported organ volume.)*

### 2.4 Segmentation
TotalSegmentator (Apache-2.0, inference only), 1.5 mm model, run as a separate process,
on the series' own grid so mask-to-voxel correspondence is the identity by construction
and is asserted. Twelve dose-relevant abdominal organs.

### 2.5 Organ mass from measured attenuation
Piecewise-linear HU → density through ICRU 44 reference tissues; mass is the sum of
local density over mask voxels, not volume times a nominal tissue density. Sensitivity
to the calibration slope is reported (< 1 % for abdominal soft tissue).

### 2.6 The organ-specific weighted CTDIvol
`w_o = [Σ_z n_o(z) I(z) / Σ_z n_o(z)] / mean_z I(z)`; organ index = CTDIvol · w_o.
CTDIvol is the recorded value (0018,9345) where present, else reconstructed from
acquisition physics against an openly licensed coefficient database; the two are never
merged. Implausible recorded values are rejected and reconstructed instead.

### 2.7 Verification
Anatomical screens (laterality, superior–inferior ordering, mass plausibility against
ICRP 89, weight variation) run on every series; a mirrored or inverted segmentation
produces entirely plausible volumes and masses, so it must be tested for directly.

### 2.8 Reproducibility
MIT engine; every reported value re-derived from the per-series records by the test
suite; results and figures regenerated by scripts in `tools/`.

## 3. Results

### 3.1 Cohort
40 series, 4 manufacturers, 455 organ records, 12 organs, 21 collections, 23 scanner
models.

### 3.2 Segmented organ mass against ICRP 89 — **Figure 1**
Liver 1.06, kidneys 1.15 / 1.17, spleen 1.72, pancreas 0.61 (median ratio, untruncated
organs). The pancreas and spleen offsets are stated as characteristics of the
segmentation and the cohort, not corrected.

### 3.3 Dose-index availability by manufacturer — **Figure 2**
Recorded 29/40; reconstructed 5/40; unrecoverable 6/40, all GE. GE 0/10 recorded.

### 3.4 The organ-level index — **Figure 3**
Organ weights span ≈ 0.59–1.69 across the cohort; a worked series end to end.

### 3.5 What limits an organ-level study — **Figure 4**
Truncation 5.2 % (GE) to 20.0 % (Philips) of organ records. Three series show no
usable variation in tube current across their organs.

## 4. Discussion

- What the organ-level index adds over a whole-scan index, and what it still is not.
- The pancreas and spleen offsets: segmentation behaviour vs genuine cohort anatomy;
  what would separate them.
- The availability finding at the organ level, and what it means for anyone attempting
  retrospective organ dosimetry on archive data.
- **The openness boundary.** Converting this index to absorbed dose in mGy requires
  CTDIvol-normalised organ dose coefficients. Every source located is subscription or
  NC-ND; none can be redistributed. This extends the CTDIvol-availability finding of
  [IORN-004] to the organ level and is stated as a limitation of the field, not of the
  method.
- Limitations: n = 10 per vendor; oncology cohort, not a reference population; a single
  segmentation model; no absorbed dose; no independent ground truth for organ mass.

## 5. Conclusion

[to draft]

---

## Data and code availability

Code: `https://github.com/Institute-of-One/ctsegdose-core` (MIT), archived at Zenodo.
No imaging is redistributed: every series is identified in `data/PROVENANCE.json` by
collection, collection DOI, Series Instance UID and licence, and is retrievable from The
Cancer Imaging Archive by following `docs/REPRODUCING_DATA.md`. All series are CC BY.

## Acknowledgement of prior work

This work builds on and cites the open dose engine of ctdose-core, which stops
deliberately at the whole-scan dose index.
