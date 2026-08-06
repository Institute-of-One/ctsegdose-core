<!--
MDPI *Tomography* — Article (regular submission, single-blind; author identity retained).
Every quantity is re-derived from results/analysis_1.5mm.json by
tests/test_manuscript_consistency.py; no number in this document is typed by hand.
-->

# Anatomy-Weighted CTDIvol from Routine CT Metadata: A Patient-Specific, Multi-Vendor Study Using Deep-Learning Segmentation

**Shuji Yamamoto**

Institute of One, LISIT Co., Ltd., Tokyo, Japan; yamamoto@lisit.jp; ORCID 0000-0001-9211-1071

**Correspondence:** yamamoto@lisit.jp

## Abstract

CTDIvol is a scanner-output index, not a patient or organ dose, and cannot express how
tube-current modulation varies along a patient. An organ-specific weighted CTDIvol
addressing this has been reported previously, but on single-institution cohorts and often
from data archives do not retain. We operationalised that quantity openly across
manufacturers, from routine DICOM metadata and automated segmentation alone; it is not an
estimate of absorbed organ dose. Forty abdominal CT series, ten per manufacturer, were
drawn from The Cancer Imaging Archive and twelve organs segmented with TotalSegmentator at
inference only. Of 480 requested organ–series combinations, 455 were produced; the
remaining 25 organs lay outside the scanned range. A prespecified acquisition-constancy
criterion admitted 39 of the 40 series. Modulation weights spanned 0.59 to 1.69, so the
index departs from the whole-scan CTDIvol by up to 70% within one acquisition. A recorded
CTDIvol was retained in the archived headers of 29 of 40 series, was reconstructable in 5
and unavailable in 6, differing markedly between manufacturers. Estimated organ mass was
broadly consistent with ICRP 89 values for liver and kidneys. Conversion to absorbed organ
dose requires Monte-Carlo coefficients this index does not replace. The implementation and
derived records are openly available.

**Keywords:** computed tomography; CTDIvol; tube-current modulation; deep-learning
segmentation; TotalSegmentator; image-based dosimetry indices; reproducibility; open data

## 1. Introduction

Two different things are routinely conflated when CT dose is discussed. The first is that
a *dose index* is not a *dose*: CTDIvol describes the output of a scanner into a standard
cylinder of acrylic, and is a property of the acquisition rather than of the patient in
it — a distinction set out explicitly by McCollough et al. [1]. The second is that a
single value per series cannot express variation along the patient: almost every modern
acquisition modulates the tube current longitudinally, so the conditions over the liver
and over the bladder are not the same, and one number for the series conceals that.

That second point is well established. Angel et al. showed that tube-current modulation
changes organ dose substantially and size-dependently [2]. Khatonabadi et al. demonstrated
that a *regional* or *organ-specific* CTDIvol, formed from the modulation profile over an
organ's own location, tracks Monte-Carlo organ dose far better than the whole-scan value,
raising the coefficient of determination for liver dose from 0.26 to 0.86 [3]. Tian et al.
formalised a **weighted organ-specific CTDIvol** computed from the modulation profile and
used it, with organ-dose coefficients, to predict organ dose prospectively [4]. Related
work has validated Monte-Carlo modelling of modulation [5], generalised organ-dose
estimation under modulation using patient-size descriptors [6], and recovered tube-current
profiles where they were not directly available [7].

**The quantity examined here is therefore not new, and no new index is proposed.** This
study takes the organ-specific weighted CTDIvol already reported in that literature and
asks a different question: what happens when it is operationalised openly, end to end, on
heterogeneous archived data from four manufacturers, using automated segmentation and
nothing but the metadata a scanner already writes?

The existing work does not answer that, and the gap is structural rather than incidental.
Those studies rest on single-institution cohorts and one or two scanner models; several
require raw projection data or vendor-supplied modulation profiles, which archived DICOM
does not retain; their organ contours are manual or semi-automatic; and neither their
implementations nor their coefficient tables are openly redistributable. Three practical
questions therefore remain open: whether the quantity can be computed at all from what
public archives actually keep, how it behaves across manufacturers when it can, and what
fraction of archived data supports it. Deep-learning segmentation removes the obstacle
that previously made the attempt impractical at scale — a general-purpose segmenter such
as TotalSegmentator [8], built on nnU-Net [9], produces abdominal organ masks from a
routine series in seconds at inference only, so the anatomy is now effectively free.

This study is accordingly an **open, multi-vendor operationalisation and empirical
characterisation** of a previously reported quantity: the whole-scan CTDIvol scaled by a
dimensionless, organ-specific weight formed from the recorded per-slice tube current over
that organ's segmented longitudinal extent, referred to here as the anatomy-weighted
CTDIvol index. It is explicitly *not* an estimate of absorbed organ dose, and Section 2.10
sets out what it does not account for.

Converting an index of this kind into an absorbed organ dose in milligray requires
CTDIvol-normalised organ-dose coefficients, computed by Monte-Carlo simulation over
anthropomorphic phantoms and corrected for patient size [10,11,12]. Such coefficient sets
exist, are well validated, and are in routine use; the index reported here does not
replace them and is not offered as a surrogate for their output.

The contributions are:

1. An open, end-to-end implementation of the organ-specific weighted CTDIvol computed
   entirely from data a scanner already records, requiring neither projection data nor
   manual contouring.
2. The first multi-vendor empirical characterisation of that quantity on public archive
   data: its range, its within-acquisition spread, and its behaviour across four
   manufacturers.
3. A description, for this archive sample, of how often the required inputs are actually
   retained in archived DICOM headers — a precondition for any retrospective study of
   this kind.
4. A prespecified acquisition-constancy criterion that makes the assumption underlying
   the weighting testable rather than implicit.
5. An external reference comparison of attenuation-derived estimated organ mass against
   ICRP 89 reference values [13], and the practical limits an organ-level modulation
   analysis must handle.

## 2. Materials and Methods

### 2.1. Data Selection Without Bulk Download

Handing a collection manifest to a bulk downloader fetches an entire collection, which for
the low-dose CT collection is of the order of 600 GB, most of it raw projection data
irrelevant to this work. We therefore used a metadata-first procedure over the public NBIA
REST API with four stages — index, screen, probe, download — in which only the last
transfers a series.

All imaging was drawn from The Cancer Imaging Archive [19]. The candidate index was built
from 47,181 CT series across 21 abdominal collections, read as series-level JSON with no
pixel data. Candidates were seeded from the public-archive
survey distributed with the companion software release [15] — a software record, not a
peer-reviewed study — and supplemented by direct collection queries. A metadata screen then rejected, in order and with each rejection
counted: non-patient collections (imaging phantoms and de-identification benchmarks);
projection and raw-data series, of which 398 were refused at this stage; non-diagnostic
series (localisers, dose reports, screen captures); series shorter than 40 or longer than
1200 images; and series outside the abdomen. One series was retained per patient per
collection, and each manufacturer's quota was drawn round-robin across its collections so
that manufacturer was not confounded with a single collection.

### 2.2. Inclusion Criteria and Header Probing

Sixty-two surviving candidates were probed by fetching six image headers each and judged
on four requirements: per-slice tube current present on *every* probed slice; that current
genuinely modulated (peak-to-peak over mean at least 0.02, so header rounding is not
mistaken for modulation); a defined Hounsfield rescale; and a reconstructed-image SOP
class with at least 40 slices spanning at least 120 mm. Forty series were kept, ten per
manufacturer.

No imaging is redistributed. Each retained series is identified in the shipped provenance
record by collection, collection DOI, Series Instance UID, manufacturer, model, licence
and retrieval date. All forty series were retrieved under Creative Commons Attribution
licences — 33 under CC BY 4.0 and 7 under CC BY 3.0, as recorded per series in
`data/PROVENANCE.json`; users must in every case observe the licence terms of the
originating collection and the TCIA Data Usage Policy.

### 2.3. Resolving the Slice Grid

Organ volume is a voxel count multiplied by a voxel volume, so slice spacing multiplies
every volume and every mass estimate. Two properties of archived series make the obvious
calculation wrong, and both are silent.

First, the file count is not always the position count. One series in this cohort contains
160 images at 119 distinct longitudinal positions; taking the spacing as the extent
divided by the number of images gives 3.71 mm where the true spacing is 5.0 mm, and
stacking the duplicated images repeats anatomy so that organs occupy more slices than they
physically do. We therefore resolve the grid explicitly: one image per position, spacing
from the median step between neighbouring positions, and a uniformity check that refuses a
series whose steps vary by more than 2%. Two series proved to be a pair of reconstructions
interleaved under a single Series Instance UID; for these the largest regular sub-grid was
taken, accepted only when it preserves the full longitudinal extent.

Second, the ordering of the slice axis is not guaranteed. Slice Location (0020,1041) runs
opposite in sign to Image Position (Patient) on three of the four manufacturers in this
sample, so a series sorted by the former arrives head-first. The segmentation is
unaffected, because the geometry handed to the segmenter is built from patient
coordinates; but the array index ceases to mean "towards the head", which reverses every
reported organ extent. Volumes are therefore canonicalised so that index zero is the most
inferior slice, with the tube current reordered alongside, since *I(z)* is paired to the
slice axis by index.

### 2.4. Reading Hounsfield Units

Outside the reconstruction circle an image carries a padding value rather than a
measurement, and that value must be replaced with air before anything is measured. Pixel
Padding Value (0028,0120) has a value representation that depends on Pixel Representation,
and this is not reliably honoured: in this sample one export writes 63,536 with an
unsigned representation on signed pixel data, which is the two's complement encoding of
the −2000 intended. Read literally, that places the padding threshold above every
Hounsfield value in the image, and the volume becomes uniform air. Four series were
affected, with no symptom other than a segmenter returning empty masks. We reinterpret the
padding value against Pixel Representation, and additionally refuse any padding rule that
would blank essentially the whole image.

### 2.5. Segmentation

Twelve abdominal organs were segmented with TotalSegmentator v2.17 [8], `total` task, 1.5
mm full-resolution model, at inference only; no weights were trained, modified or
redistributed. Inference runs in a separate child process, because nnU-Net spawns its own
workers and doing so from a long-lived parent leaks processes on Windows.

The series is written to NIfTI by our own code, with an affine constructed from the DICOM
patient coordinates, and the masks return on that same grid; the correspondence between
mask voxel and image voxel is therefore the identity by construction, and is asserted
rather than assumed. A mirrored segmentation would otherwise produce entirely plausible
volumes and Hounsfield values while pairing every organ with the wrong anatomy.

The patient outline, used for the water-equivalent diameter, was taken from a
deterministic threshold contour following AAPM Report 220 [16]. Figure 1 illustrates the
segmentation output and its correspondence with the CT anatomy in the representative
acquisition used for the end-to-end example.

### 2.6. Attenuation-Derived Estimated Organ Mass

Hounsfield units were converted to mass density by piecewise-linear interpolation through
reference tissue anchor points, taking the densities from ICRU Report 44 [17] and the
construction from Schneider et al. [18]. Estimated organ mass is the sum of local density
over mask voxels multiplied by the voxel volume.

These are **model-based estimates, not measurements**. Contrast enhancement, tube voltage,
reconstruction kernel and scanner-specific HU calibration may all affect
attenuation-derived density estimates, and none was controlled in this archive cohort:
contrast phase in particular varies between and within collections. The reported masses
should therefore be interpreted as model-based estimates rather than physical ground
truth. The HU-to-density curve itself is replaceable and travels into the provenance of
every estimate; abdominal soft tissue is relatively insensitive to the choice, since
perturbing the water-to-muscle slope by 10% changes an abdominal organ-mass estimate by
less than 1%.

### 2.7. The Anatomy-Weighted CTDIvol Index

For an organ *o* occupying slices with per-slice voxel counts *n_o(z)*, and a series with
per-slice tube current *I(z)*, the organ-specific modulation weight is

*w_o* = [ Σ_z *n_o(z) I(z)* / Σ_z *n_o(z)* ] / mean_z *I(z)*     (1)

and the anatomy-weighted CTDIvol index is CTDIvol · *w_o*. The weight is dimensionless and
is the transferable quantity: it expresses the recorded longitudinal tube-current
conditions over the organ relative to the scan mean, independently of the scanner's own
output.

The weighting assumes that, within each series, tube voltage, rotation or exposure time,
pitch and beam collimation remain fixed, so that longitudinal changes in scanner output
are proportional to the recorded tube current.

### 2.8. The Acquisition-Constancy Criterion

Rather than assume that assumption, we stated it as an eligibility rule in advance and
applied it mechanically:

> A series is eligible for quantitative anatomy-weighted CTDIvol analysis only when the
> acquisition parameters required for scanner output to remain proportional to the
> recorded tube current are constant within that series, to the extent verifiable from the
> archived DICOM headers.

Every slice header of every series was read and each output-governing attribute — tube
voltage, exposure time, rotation time, pitch and total collimation width — classified into
one of four states. *Verified constant*: one value throughout. *Absent*: never written to
the archived headers, so constancy can be neither confirmed nor refuted; absence alone
does not disqualify a series, since excluding on it would remove series for a property of
the de-identified export rather than of the acquisition. *Negligible variation*: varying
by less than a relative tolerance of 0.02, attributable to the numeric representation —
exposure time is written as an integer number of milliseconds, so a one-unit step on a
value of a few hundred is a rounding artefact. *Materially variable*: varying by at least
that tolerance, which disqualifies the series. The tolerance is not fitted to these data:
any value between roughly 1% and 50% classifies this cohort identically, because the only
material variation observed is a factor of two.

Tube voltage was verified constant in all 40 series, as were Image Type and convolution
kernel, so no series mixes acquisition or reconstruction types. Exposure time was verified
constant in 35 series, negligibly variable in 3, materially variable in 1 and absent in 1;
rotation time verified constant in 20, materially variable in 1 and absent in 19; pitch
verified constant in 32 and absent in 8; total collimation width verified constant in 31
and absent in 9. The full record is shipped as `results/acquisition_constancy.json` and
the eligibility decision for every series in `results/analysis_1.5mm.json`.

Series failing the criterion are retained for the archive-availability, segmentation,
estimated-mass and provenance analyses, and excluded only from quantitative
modulation-weight and anatomy-weighted-index summaries.

CTDIvol was taken from the image header (0018,9345) where present. Where absent, it was
reconstructed from acquisition physics against an openly licensed normalised-CTDI database
[14]; recorded and reconstructed values are never merged, and each series records which it
carries. A recorded value outside the physically possible range is treated as a corrupt
attribute and falls through to reconstruction — one series records CTDIvol as
−3.7 × 10^19 mGy.

An organ whose mask reaches the first or last slice of the series continues beyond the
scan; its estimated mass is that of the scanned part and its weight describes only the
exposed part. Such organs are flagged and excluded from whole-organ comparisons.

### 2.9. Organ Record Flow

Forty series and twelve requested organs give 480 organ–series combinations. Records were
produced for 455. The remaining 25 are organs that lay outside the scanned longitudinal
range, so their masks were empty and no record exists: urinary bladder in 10 series,
gallbladder in 8, and seven further organs in a single 41-slice pelvic acquisition that
does not reach the upper abdomen.

Two further conditions apply to the 455 records, and they are **independent axes rather
than nested subsets**. Truncation is a property of the organ: 408 records are untruncated
and 47 reach a scan boundary. Index availability is a property of the series: 386 records
carry an anatomy-weighted index, while the remaining 69 records, from 6 series, carry a
modulation weight but no index, because those series have no CTDIvol by either route. The
two conditions hold together for 345 records; 41 truncated records still carry an index,
and 63 untruncated records do not. The external reference-mass comparison uses the 177
untruncated records of the five solid organs. The full flow is in the shipped
`results/analysis_1.5mm.json`.

### 2.10. What the Index Does and Does Not Represent

The anatomy-weighted CTDIvol index describes organ-specific *longitudinal* tube-current
modulation relative to the whole-scan CTDIvol. It does not account for scattered
radiation; irradiation originating outside the organ's segmented longitudinal extent;
angular (in-plane) tube-current modulation; organ depth, position or attenuation;
patient-specific Monte-Carlo radiation transport; or absorbed organ dose in milligray. It
is therefore not a surrogate for absorbed organ dose and must not be read as one. What it
does provide is a dimensionless, patient-specific, organ-specific measure of longitudinal
modulation, and a derived index in the units of the parent CTDIvol.

### 2.11. Verification and Reproducibility

Every series is screened against facts of gross anatomy that hold for any adult: the left
kidney lies to the patient's left of the right kidney and the spleen to the left of the
liver; the liver lies superior to the bladder and the adrenal glands superior to the
kidneys; solid-organ mass estimates fall within a wide band of reference values; and the
organ weights vary within a series. These screens exist because the failures they catch
leave no other trace.

Analyses were run with Python 3.14 (the package supports 3.10 and later), TotalSegmentator
v2.17 (`total` task, 1.5 mm full-resolution model) on PyTorch 2.11 with CUDA 12.8, pydicom
3.0 and NumPy 2.5, using an NVIDIA RTX 3080. Every reported value is re-derived from the
per-series records by the test suite, including regenerating the complete analysis table
and comparing it. The acquisition, organ, analysis and figure layers are regenerated by
`tools/select_and_download.py`, `tools/run_organ_dose.py`, `tools/make_analysis.py` and
`tools/make_figures.py` respectively; the acquisition-parameter check by
`tools/verify_acquisition_constancy.py`. The repository, its release tag, commit hash and
archived version DOI are given in the Data Availability Statement.

## 3. Results

### 3.1. Cohort and Records

Forty series were analysed — ten from each of GE, Siemens, Canon/Toshiba and Philips —
drawn from 21 collections and 23 scanner models. Of 480 organ–series combinations, 455
organ records were produced across 12 organs, with the flow as given in Section 2.9.
Figure 1 shows the segmentation output for a representative acquisition.

Of the 40 segmented series, 39 met the acquisition-constancy criterion of Section 2.8. One
archived GE series contained two blocks with different exposure and rotation times; it was
retained for the segmentation, estimated-mass and archive-availability analyses and
excluded from the modulation-weight and anatomy-weighted-index summaries. The quantitative
modulation analysis therefore rests on 375 organ records from the 33 eligible series that
also carry a CTDIvol.

![](figures/fig1_segmentation.png){width=100%}

**Figure 1.** Representative TotalSegmentator output from one abdominal CT series used in
the analysis. (**a**) Coronal reformat with multi-organ overlays. (**b**–**d**) Axial
levels through the upper abdomen, the renal level and the lower abdomen. Masks were
generated with TotalSegmentator v2.17 using the 1.5 mm full-resolution `total` task and
returned to the native DICOM-derived image grid; overlays are translucent so the anatomy
they are drawn against remains visible, and the key lists the structures actually shown.
The same acquisition appears in Figure 2. Images are displayed in radiological convention,
with the patient's left on the viewer's right. Only de-identified imaging from The Cancer
Imaging Archive is shown.

### 3.2. Organ-Specific Modulation Weights

Across the eligible series, organ-specific modulation weights span 0.59 to 1.69. The
anatomy-weighted index therefore departs from the whole-scan CTDIvol by up to roughly 70%
in either direction within a single acquisition, which is the variation the index exists
to express.

Figure 2 shows one acquisition end to end: a Canon/Toshiba Aquilion PRIME series of 268
slices with a recorded CTDIvol of 16.1 mGy and a scan mean tube current of 265 mA. The
small bowel and colon, lying in the inferior abdomen where the modulation raised the
current to about 447 and 435 mA, take weights of 1.69 and 1.64 respectively, giving indices
near 27 mGy; the left kidney and stomach, higher in the scan, take weights of 0.94 and
indices near 15 mGy. Two organs in the same acquisition thus differ by a factor of 1.8 in
their anatomy-weighted index, a difference no whole-scan value can express.

![](figures/fig2_demonstration_case.png){width=100%}

**Figure 2.** One acquisition end to end. (**a**) Each organ's longitudinal extent,
annotated with the mean tube current recorded over it, against a scan mean of 265 mA.
(**b**) The resulting anatomy-weighted CTDIvol index, with the organ-specific modulation
weight beside each bar; the dashed line is the whole-scan CTDIvol of 16.1 mGy. The bars
are modulation-weighted indices, not absorbed doses.

### 3.3. Availability of a Dose Index in the Archived Headers

Across the cohort, 29 of 40 series retained a recorded CTDIvol in the archived DICOM
headers, 5 were reconstructable from acquisition physics, and 6 were neither (Figure 3).

All 6 unrecoverable series are GE, and none of the ten sampled GE series retained a
recorded CTDIvol in the archived headers; the other three manufacturers retained one in 29
of 30. These counts are reported descriptively. No significance test is applied: series
drawn from a curated archive are not independent with respect to collection, contributing
site, scanner model, export pathway or de-identification, and a p-value computed over that
structure would describe a sampling model the data do not satisfy. For the six
unrecoverable series the organ masks, volumes, mass estimates and modulation weights are
all computable and reported, but no anatomy-weighted index exists for them.

![](figures/fig3_dose_index_availability.png){width=100%}

**Figure 3.** Availability of a whole-scan dose index in the archived DICOM headers, by
manufacturer, in this TCIA sample. A series counted unrecoverable retained no CTDIvol in
its header and its scanner lies outside the open coefficient database. Segments are
distinguished by fill pattern as well as tone.

### 3.4. External Reference Comparison of Estimated Organ Mass

Table 1 and Figure 4 place attenuation-derived estimated organ mass beside the ICRP 89
reference adult male values [13], over the 177 untruncated records of the five solid
organs.

**Table 1.** Attenuation-derived estimated organ mass beside ICRP 89 reference adult male
values, untruncated organs only. The reference is an external anchor, not a subject-level
ground truth.

| organ | n | median estimated mass | ratio to ICRP 89 |
|---|---|---|---|
| liver | 34 | 1901 g | 1.06 |
| spleen | 38 | 259 g | 1.72 |
| kidney (left) | 34 | 179 g | 1.15 |
| kidney (right) | 35 | 182 g | 1.17 |
| pancreas | 36 | 86 g | 0.61 |

Estimates for the liver were broadly consistent with the reference value, within 6%, and
the kidneys within 17%. Two organs departed more substantially: the pancreas estimate was
39% below the reference and the spleen 72% above. Neither was adjusted; possible
explanations are examined in Section 4.

![](figures/fig4_organ_mass_vs_icrp89.png){width=100%}

**Figure 4.** Attenuation-derived estimated organ mass relative to the ICRP 89 reference
adult male mass, by manufacturer. Each marker is one organ in one series; the horizontal
bar is the median across all manufacturers. Organs truncated by the scan boundary are
excluded. Manufacturer is encoded by marker shape as well as colour, so the figure is
readable in greyscale.

### 3.5. What Limits an Organ-Level Modulation Analysis

Two conditions reduce what such an analysis can measure, and both differ across the
sampled manufacturers (Figure 5).

Truncation by the scan boundary affected 5.2% of organ records on GE, 5.5% on Siemens,
10.4% on Canon/Toshiba and 20.0% on Philips, reflecting the scan ranges of the sampled
acquisitions rather than any property of the scanners. The organs most often cut are the
colon and small bowel.

3 of the 40 series showed a peak-to-peak spread of organ weights below 0.02: their tube
current does not vary across the abdominal organs, so the weighting has nothing to
express. These series passed the modulation screen at selection, where the current varies
across the whole scan; the flatness is local to the abdomen. They are uninformative for
this analysis rather than faulty.

![](figures/fig5_study_limits.png){width=100%}

**Figure 5.** What limits an organ-level modulation analysis. (**a**) Percentage of organ
records truncated by the scan boundary, by manufacturer, annotated with the counts.
(**b**) Peak-to-peak spread of the organ-specific modulation weights within each series;
the dashed line is the threshold below which a series carries no usable variation.

## 4. Discussion

**Principal finding.** Organ-specific modulation weights span 0.59 to 1.69 across this
cohort, and within a single acquisition two organs differed by a factor of 1.8 in their
anatomy-weighted index. Longitudinal modulation therefore produces organ-specific exposure
conditions that a single whole-scan CTDIvol cannot represent, and the magnitude is large
enough to matter for any organ-level analysis built on that value.

**Interpretation.** The weight is a direct, dimensionless summary of how the recorded tube
current was distributed over an organ's own longitudinal extent in that patient. It
requires no phantom, no simulation and no additional acquisition — only metadata the
scanner already writes and a segmentation obtained at inference.

**Relation to previous work.** The quantity is that of Khatonabadi et al. [3] and Tian et
al. [4]; what is new here is the conditions under which it was obtained. Those studies,
and the related modulation dosimetry of Angel et al. [2] and Bostani et al. [5,6],
established the concept and validated it against Monte-Carlo dose on single-institution
cohorts, typically with one or two scanner models, manual or semi-automatic contours, and
in several cases raw projection data or vendor-supplied modulation profiles. Archived
DICOM retains none of the latter. This study consequently answers questions those designs
were not positioned to address: that the quantity is computable from archived headers
alone, on four manufacturers and 23 scanner models, with contours obtained at inference;
that its range across such a cohort is 0.59 to 1.69; and — the finding with the most
practical consequence — that the whole-scan CTDIvol it scales is missing from a
manufacturer-associated fraction of archived series, which bounds any retrospective study
of this kind before segmentation is even attempted. The contribution is therefore
operationalisation and empirical characterisation, not a new index, and the reported
values should be read as describing archived multi-vendor data rather than as improving
on the Monte-Carlo validations already published.

**What the index is, and what it is not.** As set out in Section 2.10, the index addresses
longitudinal modulation alone. It does not account for scatter, for irradiation
originating outside the organ's segmented extent, for angular modulation, for organ depth
and attenuation, or for radiation transport, and it is not an estimate of absorbed organ
dose in milligray. Its value lies in isolating one well-defined contribution — the
longitudinal one — and reporting it patient-specifically and reproducibly.

**External reference comparison of estimated mass.** Liver and kidney estimates were
broadly consistent with ICRP 89 reference values, which is the expected behaviour if the
segmentation and the density model are working. The pancreas estimate sits 39% below the
reference. The pancreas is the weakest of these organs in TotalSegmentator's own
validation (Dice 0.887, against 0.965 for the liver, 0.983 for the spleen and 0.953 and
0.939 for the kidneys [8]), which supports reduced boundary agreement; but Dice is
symmetric and does not establish the *direction* of a disagreement, so it does not by
itself demonstrate under-segmentation. Contrast phase, reconstruction kernel and genuine
anatomical variation in this cohort are alternative contributors that the present design
cannot separate. The source of the discrepancy cannot be determined without subject-level
reference contours or clinical ground truth.

The spleen estimate sits 72% above the reference. The four largest cases — 4.47, 3.74,
3.34 and 3.06 times the reference value — were reviewed slice by slice against their own
images: each contour follows the splenic boundary with correct laterality, tracks the notch
at the hilum, shows no leakage into liver, kidney or stomach, and forms a single connected
component, so no accessory spleen was included. Their mean densities, 1.052 to 1.078
g/cm³, are unremarkable for splenic tissue, so the elevation arises from segmented volume
rather than from the density model; and the spleen is the best-segmented of these organs
in the segmenter's validation. The cases arise on four different manufacturers. The cohort
is oncological — renal cell, colorectal and adrenal carcinoma — in which splenomegaly is
common, and the ICRP reference adult is not that population. The observed elevation was
most consistent with cohort anatomy among the explanations examined, although
subject-level pathological confirmation was unavailable.

**Availability in archived headers, and its confounders.** In this sample, availability of
a recorded CTDIvol differed markedly with manufacturer. This is an observation about
archived DICOM headers in one curated archive, not a statement about scanner
implementations. The present design cannot distinguish between scanner implementation,
scanner generation, acquisition site, DICOM export pathway, PACS processing,
de-identification, archive curation and collection composition as the origin of a missing
attribute; several of these are confounded with manufacturer through collection
membership. The practical consequence stands regardless of cause: a retrospective
organ-level analysis drawn from an archive will lose a manufacturer-associated fraction of
its cohort before segmentation is considered, and a study that does not report which
series were lost will under-represent that manufacturer silently.

**Why acquisition constancy has to be screened.** The acquisition-constancy criterion
identified one series in which tube current alone was not proportional to scanner output,
because the acquisition changed rotation and exposure time part-way through. Excluding it
prevents a mixed acquisition from entering the quantitative modulation analysis, and
illustrates why constancy must be verified rather than assumed: nothing in the images, the
segmentation or the weights themselves would have revealed it. A current–time product
would be the more faithful weighting in general; it is not adopted here because exposure
time is absent from the archived headers of one other series and could not be applied
uniformly across the cohort.

**Reproducibility and open implementation.** The implementation is MIT-licensed, no
imaging is redistributed, and every reported value is re-derived from the per-series
records by an automated test suite that regenerates the analysis tables and compares them.
The acquisition procedure, the analysis and the figures are each a single command. The
components differ in status and are not claimed as uniformly open: the imaging is publicly
accessible under collection-specific licences; the segmentation software and the `total`
task weights are openly redistributable; the normalised-CTDI database is CC BY; the
HU-to-density anchor values are used by citation to ICRU Report 44 [17] and Schneider et
al. [18] rather than redistributed; and no Monte-Carlo coefficient table is included at all.

**The boundary to absorbed organ dose.** Converting an anatomy-weighted index into an
absorbed organ dose requires CTDIvol-normalised coefficients with a patient-size
correction, of the kind established by Turner et al. [10] and extended over larger patient
model libraries by Tian et al. [11], together with the transport considerations listed in
Section 2.9. Those coefficient sets are published in subscription journals or distributed
with research software under terms granting use but not redistribution, which reflects a
publishing convention for Monte-Carlo reference data rather than any deficiency in the
coefficients themselves. Normalised-CTDI data of the kind used here to reconstruct a
missing whole-scan index has been published under CC BY [14], which shows the convention is
movable. The software accordingly refuses to emit a dose in milligray unless supplied with
a coefficient table carrying its citation, DOI, licence and source hash.

**Limitations.** Ten series per manufacturer supports a median and an interquartile range,
not a distributional claim, and series within the archive are not independent with respect
to collection, site, scanner model or export pathway. The cohort is oncological and not a
reference population. Contrast phase, tube voltage and reconstruction kernel were not
controlled, and all affect attenuation-derived mass estimates. A single segmentation model
was used, so segmentation behaviour and cohort anatomy cannot be separated. There is no
subject-level ground truth for organ mass. Rotation time, pitch and collimation are absent
from the archived headers of some series (19, 8 and 9 respectively), so their constancy
within those series could not be fully verified and is assumed from the single-acquisition
representation; the acquisition-constancy criterion is therefore a screen against
detectable violations rather than a guarantee. The index addresses longitudinal modulation
only, as set out in Section 2.10, and no absorbed dose is reported.

**Future work.** Coefficients computed with an open-source Monte-Carlo engine would carry
no licensing constraint and would permit the transport terms this index omits; an
independent segmentation model on the same series would separate segmentation behaviour
from cohort anatomy for the pancreas and spleen; and a larger archive cohort would allow
the availability observation to be examined with collection and site modelled explicitly
rather than confounded.

## 5. Conclusions

This study did not compute absorbed organ dose, and did not propose a new index. It took
an organ-specific weighted CTDIvol already reported in the literature and established what
it yields when operationalised openly across manufacturers, from metadata a scanner
already records and organ masks obtained at inference.
Organ-specific modulation weights spanned 0.59 to 1.69, and two organs within one
acquisition differed by a factor of 1.8 in their index, so the variation a single
whole-scan value conceals is substantial. The approach ran across four manufacturers using
publicly accessible imaging data and openly redistributable software components, and in
this archive cohort the availability of the whole-scan CTDIvol the index scales varied
markedly between manufacturers, which constrains any retrospective analysis of this kind.
The index is a step before conversion to absorbed organ dose, not a substitute for it:
that conversion requires Monte-Carlo coefficients and the transport terms this index
omits. The implementation and derived records are openly available, provenance-aware and
reproducible from the shipped results.

## Author Contributions

S.Y.: conceptualisation, methodology, software, validation, formal analysis,
investigation, data curation, writing — original draft, writing — review and editing,
visualisation. The author has read and agreed to the published version of the manuscript.

## Funding

This research received no external funding.

## Institutional Review Board Statement

Not applicable. This study used only publicly available, de-identified imaging from The
Cancer Imaging Archive.

## Informed Consent Statement

Not applicable. This study used only publicly available, de-identified imaging from The
Cancer Imaging Archive.

## Data Availability Statement

The software supporting this study, `ctsegdose-core`, is openly available under the MIT
licence at https://github.com/Institute-of-One/ctsegdose-core, release
{{RELEASE_TAG}} (commit {{COMMIT_HASH}}), archived at Zenodo under
{{ZENODO_VERSION_DOI}}. The machine-readable results the manuscript quotes, the per-organ
records, the analysis tables, the acquisition-parameter check and the figure scripts are
included in that repository, together with the mask-review overlays underlying Section 4.

No DICOM imaging is redistributed. Every series analysed is identified in
`data/PROVENANCE.json` by collection, collection DOI, Series Instance UID, manufacturer,
model, licence and retrieval date, and is retrievable directly from The Cancer Imaging
Archive by following `docs/REPRODUCING_DATA.md`. The forty series were retrieved under
Creative Commons Attribution licences (33 under CC BY 4.0, 7 under CC BY 3.0) as recorded
per series; the licence terms of each originating collection and the TCIA Data Usage
Policy apply.

Segmentation used TotalSegmentator [8]. Its software code is distributed under the Apache
2.0 licence, and the `total` task model weights used here are likewise stated by the
project to be openly available under Apache 2.0; other tasks in that project require a
separate licence and were not used. No model weights are redistributed with this work.

No organ-dose coefficient table is redistributed, for the licensing reasons set out in
Section 4.

## Conflicts of Interest

The author is the representative of LISIT Co., Ltd. (Tokyo, Japan) and Chief Executive
Officer of TexelCraft OÜ (Estonia), and has a commercial interest in downstream products
that may incorporate or build upon the methods described here. The software reported in
this article is released under the MIT licence. No patient data, customer data or
proprietary clinical data were used; all imaging is publicly available and de-identified.

## Use of Generative Artificial Intelligence

A generative artificial intelligence assistant (Claude, Anthropic) was used as a tool in
the development of the software and in drafting and editing the text of this manuscript.
All reported results are produced by executable code contained in the cited repository,
are re-derived from the underlying per-series records by an automated test suite, and were
verified by the author. All scientific judgements, interpretations and conclusions are the
author's, who takes full responsibility for the content of this article.

## References

1. McCollough, C.H.; Leng, S.; Yu, L.; Cody, D.D.; Boone, J.M.; McNitt-Gray, M.F. CT Dose
   Index and Patient Dose: They Are Not the Same Thing. *Radiology* **2011**, *259*,
   311–316. https://doi.org/10.1148/radiol.11101800
2. Angel, E.; Yaghmai, N.; Jude, C.M.; DeMarco, J.J.; Cagnon, C.H.; Goldin, J.G.; Primak,
   A.N.; Stevens, D.M.; Cody, D.D.; McCollough, C.H.; McNitt-Gray, M.F. Monte Carlo
   Simulations to Assess the Effects of Tube Current Modulation on Breast Dose for
   Multidetector CT. *Phys. Med. Biol.* **2009**, *54*, 497–512.
   https://doi.org/10.1088/0031-9155/54/3/003
3. Khatonabadi, M.; Kim, H.J.; Lu, P.; McMillan, K.L.; Cagnon, C.H.; DeMarco, J.J.;
   McNitt-Gray, M.F. The Feasibility of a Regional CTDIvol to Estimate Organ Dose from
   Tube Current Modulated CT Exams. *Med. Phys.* **2013**, *40*, 051903.
   https://doi.org/10.1118/1.4798561
4. Tian, X.; Li, X.; Segars, W.P.; Frush, D.P.; Samei, E. Prospective Estimation of Organ
   Dose in CT under Tube Current Modulation. *Med. Phys.* **2015**, *42*, 1575–1585.
   https://doi.org/10.1118/1.4907955
5. Bostani, M.; McMillan, K.; DeMarco, J.J.; Cagnon, C.H.; McNitt-Gray, M.F. Validation of
   a Monte Carlo Model Used for Simulating Tube Current Modulation in Computed Tomography
   over a Wide Range of Phantom Conditions/Challenges. *Med. Phys.* **2014**, *41*,
   112101. https://doi.org/10.1118/1.4887807
6. Bostani, M.; McMillan, K.; Lu, P.; Kim, G.H.J.; Cody, D.; Arbique, G.; Greenberg, S.B.;
   DeMarco, J.J.; Cagnon, C.H.; McNitt-Gray, M.F. Estimating Organ Doses from Tube Current
   Modulated CT Examinations Using a Generalized Linear Model. *Med. Phys.* **2017**, *44*,
   1500–1513. https://doi.org/10.1002/mp.12119
7. McMillan, K.; Bostani, M.; Cagnon, C.H.; Yu, L.; Leng, S.; McCollough, C.H.;
   McNitt-Gray, M.F. Estimating Patient Dose from CT Exams That Use Automatic Exposure
   Control: Development and Validation of Methods to Accurately Estimate Tube Current
   Values. *Med. Phys.* **2017**, *44*, 4262–4275. https://doi.org/10.1002/mp.12314
8. Wasserthal, J.; Breit, H.-C.; Meyer, M.T.; Pradella, M.; Hinck, D.; Sauter, A.W.; Heye,
   T.; Boll, D.T.; Cyriac, J.; Yang, S.; Bach, M.; Segeroth, M. TotalSegmentator: Robust
   Segmentation of 104 Anatomic Structures in CT Images. *Radiol. Artif. Intell.* **2023**,
   *5*, e230024. https://doi.org/10.1148/ryai.230024
9. Isensee, F.; Jaeger, P.F.; Kohl, S.A.A.; Petersen, J.; Maier-Hein, K.H. nnU-Net: A
   Self-Configuring Method for Deep Learning-Based Biomedical Image Segmentation. *Nat.
   Methods* **2021**, *18*, 203–211. https://doi.org/10.1038/s41592-020-01008-z
10. Turner, A.C.; Zhang, D.; Khatonabadi, M.; Zankl, M.; DeMarco, J.J.; Cagnon, C.H.; Cody,
    D.D.; Stevens, D.M.; McCollough, C.H.; McNitt-Gray, M.F. The Feasibility of Patient
    Size-Corrected, Scanner-Independent Organ Dose Estimates for Abdominal CT Exams. *Med.
    Phys.* **2011**, *38*, 820–829. https://doi.org/10.1118/1.3533897
11. Tian, X.; Li, X.; Segars, W.P.; Frush, D.P.; Paulson, E.K.; Samei, E. Dose Coefficients
    in Pediatric and Adult Abdominopelvic CT Based on 100 Patient Models. *Phys. Med.
    Biol.* **2013**, *58*, 8755–8768. https://doi.org/10.1088/0031-9155/58/24/8755
12. Sahbaee, P.; Segars, W.P.; Samei, E. Patient-Based Estimation of Organ Dose for a
    Population of 58 Adult Patients across 13 Protocol Categories. *Med. Phys.* **2014**,
    *41*, 072104. https://doi.org/10.1118/1.4883778
13. ICRP. Basic Anatomical and Physiological Data for Use in Radiological Protection:
    Reference Values. ICRP Publication 89. *Ann. ICRP* **2002**, *32* (3–4).
14. Dinwiddie, L.E.; Baggett, J.M.; Kofler, J.M.; et al. Survey of Normalized CTDIvol
    Values Across Four Major Computed Tomography Vendors for Use in the MIRDct Software.
    *J. Appl. Clin. Med. Phys.* **2026**, *27*, e70473. https://doi.org/10.1002/acm2.70473
15. Yamamoto, S. ctdose-core: open, auditable CT dose surveillance from DICOM, with a
    physics reconstruction when the dose attributes are missing (Version 0.1.1)
    [Software]. Zenodo, **2026**. https://doi.org/10.5281/zenodo.21636082
16. McCollough, C.; Bakalyar, D.M.; Bostani, M.; Brady, S.; Boedeker, K.; Boone, J.M.;
    Chen-Mayer, H.H.; Christianson, O.I.; Leng, S.; Li, B.; et al. Use of Water Equivalent
    Diameter for Calculating Patient Size and Size-Specific Dose Estimates (SSDE) in CT:
    The Report of AAPM Task Group 220. *AAPM Report No. 220*, **2014**.
    https://doi.org/10.37206/146
17. ICRU. Tissue Substitutes in Radiation Dosimetry and Measurement. ICRU Report 44; ICRU:
    Bethesda, MD, USA, **1989**.
18. Schneider, U.; Pedroni, E.; Lomax, A. The Calibration of CT Hounsfield Units for
    Radiotherapy Treatment Planning. *Phys. Med. Biol.* **1996**, *41*, 111–124.
    https://doi.org/10.1088/0031-9155/41/1/009
19. Clark, K.; Vendt, B.; Smith, K.; Freymann, J.; Kirby, J.; Koppel, P.; Moore, S.;
    Phillips, S.; Maffitt, D.; Pringle, M.; Tarbox, L.; Prior, F. The Cancer Imaging
    Archive (TCIA): Maintaining and Operating a Public Information Repository. *J. Digit.
    Imaging* **2013**, *26*, 1045–1057. https://doi.org/10.1007/s10278-013-9622-7
