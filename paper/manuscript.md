<!--
MDPI *Tomography* — Article (regular submission, single-blind; author identity retained).
Formatted to MDPI structure: single-paragraph abstract, IMRAD headings, MDPI back-matter,
numbered references. Every quantity is re-derived from results/analysis_1.5mm.json by
tests/test_manuscript_consistency.py; no number in this document is typed by hand.
-->

# Patient-Specific, Multi-Vendor Organ-Level CT Dose Characterisation from Routine Metadata Using Deep-Learning Segmentation

**Shuji Yamamoto**

Institute of One, LISIT Co., Ltd., Tokyo, Japan; yamamoto@lisit.jp; ORCID 0000-0001-9211-1071

**Correspondence:** yamamoto@lisit.jp

---

## Abstract

Whole-scan computed tomography dose indices describe an acquisition rather than the
patient, and cannot express how differently organs in one scan are exposed. We
characterised CT dose at the organ level from the per-slice tube current a scanner
already records, combined with anatomy segmented from the same series. Forty
abdominal CT series, ten from each of four manufacturers, were drawn from The Cancer
Imaging Archive without bulk download. Twelve organs were segmented with TotalSegmentator
at inference only; organ mass came from each patient's own Hounsfield units, and an
organ-specific weighted CTDIvol from the tube current over each organ's extent. Across
455 organ records, segmented mass agreed with ICRP 89 reference values for the liver
(ratio 1.06) and kidneys (1.15 and 1.17), with systematic offsets for pancreas (0.61) and
spleen (1.72), characterised and attributed rather than adjusted. Organ modulation
weights spanned 0.59 to 1.69, so organ-level and whole-scan indices differ by up to about
70% within one acquisition. A dose index was recorded in 29 of 40 series, reconstructable
in 5, and absent in 6 — all on GE, which recorded none. Extending the chain to absorbed
dose is limited by the licensing convention for Monte-Carlo organ-dose coefficients, not
by method.

**Keywords:** computed tomography; organ dose; tube-current modulation; deep-learning
segmentation; TotalSegmentator; size-specific dosimetry; reproducibility; open data

---

## 1. Introduction

Two different things are routinely conflated when CT dose is discussed. The first is that
a *dose index* is not a *dose*: CTDIvol describes the output of a scanner into a standard
cylinder of acrylic, and is a property of the acquisition rather than of the patient in
it. The second is that a whole-scan quantity is not an organ quantity: within one
abdominal acquisition the liver and the bladder do not receive the same exposure, and no
single number for the series can express that difference.

The information needed to close the second gap is already in the file. Almost every
modern CT acquisition uses tube current modulation, and the resulting per-slice current
is written into the image header as X-Ray Tube Current (0018,1151). What has been missing
is not the exposure record but the anatomy: to weight an organ by the current delivered
over its own extent, one must know where that organ is, slice by slice, in that patient.
Manual contouring of a dose-relevant organ set across a multi-vendor cohort has never
been practical at scale.

Deep-learning segmentation removes that obstacle. A general-purpose segmenter such as
TotalSegmentator [1] produces abdominal organ masks from a routine series in seconds on a
consumer GPU, at inference only and under a permissive licence. The anatomy is now
effectively free.

What is not free is the final conversion. Turning an organ-level dose *index* into an
absorbed organ dose in milligray requires CTDIvol-normalised organ-dose coefficients,
computed by Monte-Carlo simulation over anthropomorphic phantoms and corrected for
patient size [2,3]. Such coefficient sets exist, are well validated, and are in routine
use. They are, however, distributed under terms that do not permit onward distribution:
published in subscription journals, or bundled with research software licensed for use
rather than redistribution. Open licensing has not become the convention for this class
of reference data — in contrast to, for example, the normalised-CTDI database on which
the whole-scan index can now be rebuilt [4], which shows that the alternative is
practicable rather than hypothetical.

The consequence is structural, not incidental. An openly licensed measurement chain can
be assembled from the scanner to an organ-level dose index and no further, and that
boundary sits in the field rather than in any particular implementation. We treat it as
an observation this study reports, not as a limitation it apologises for.

This study therefore does three things. It computes a patient-specific, anatomy-aware
organ dose index end to end from routine metadata across four manufacturers. It
calibrates the segmented anatomy that index rests on against a published reference,
reporting the disagreements rather than tuning them away. And it measures, at the organ
level, how often the necessary inputs are actually present in archived clinical data,
which turns out to differ sharply by manufacturer. Specifically, the contributions are:

1. An anatomy-aware, patient-specific organ dose index computed entirely from data a
   scanner already records, implemented as an openly licensed, provenance-carrying engine.
2. A multi-vendor calibration of the segmented organ masses against ICRP 89 reference
   masses [5], with two systematic offsets characterised and attributed.
3. A per-manufacturer measurement, at the organ level, of whether a dose index is
   recorded, reconstructable, or absent altogether.
4. The practical limits an organ-level modulation study must handle: organ truncation at
   the scan boundary, and acquisitions whose modulation does not vary across the abdomen.
5. An explicit statement of where openness runs out, with the coefficient licensing
   convention documented as a result.

## 2. Materials and Methods

### 2.1. Data Selection Without Bulk Download

Public imaging archives are large, and the naive retrieval path is prohibitive: handing a
collection manifest to a bulk downloader fetches an entire collection, which for the
low-dose CT collection is of the order of 600 GB, most of it raw projection data
irrelevant to this work. We therefore used a metadata-first procedure over the public
NBIA REST API with four stages — index, screen, probe, download — in which only the last
transfers a series, and only for series already shown to carry what the method needs.

The candidate index was built from 47,181 CT series across 21 abdominal collections, read
as series-level JSON with no pixel data. Candidates were seeded from the public-archive
survey of the companion whole-scan study [6], which had catalogued 400 series over the
same four manufacturers through the same API, and supplemented by direct collection
queries. A metadata screen then rejected, in order and with each rejection counted:
non-patient collections (imaging phantoms and de-identification benchmarks); projection
and raw-data series, of which 398 were refused at this stage; non-diagnostic series
(localisers, dose reports, screen captures); series shorter than 40 or longer than 1200
images; and series outside the abdomen. One series was retained per patient per
collection, and each manufacturer's quota was drawn round-robin across its collections so
that manufacturer was not confounded with a single site or protocol.

### 2.2. Inclusion Criteria and Header Probing

Sixty-two surviving candidates were probed by fetching six image headers each — a few
megabytes per series rather than hundreds — and judged on four requirements: per-slice
tube current present on *every* probed slice; that current genuinely modulated
(peak-to-peak over mean at least 0.02, so header rounding is not mistaken for
modulation); a defined Hounsfield rescale; and a reconstructed-image SOP class with at
least 40 slices spanning at least 120 mm. Forty series were kept, ten per manufacturer.
Series failing on tube current were counted per manufacturer rather than discarded
silently.

No imaging is redistributed. Each retained series is identified in the shipped provenance
record by collection, collection DOI, Series Instance UID, manufacturer, model, licence
and retrieval date, which is what a reader needs to retrieve the identical data. All forty
series are published under Creative Commons Attribution licences.

### 2.3. Resolving the Slice Grid

Organ mass is a voxel count multiplied by a voxel volume, so slice spacing multiplies
every mass reported. Two properties of archived series make the obvious calculation
wrong, and both are silent.

First, the file count is not always the position count. One series in this cohort contains
160 images at 119 distinct longitudinal positions; taking the spacing as the extent
divided by the number of images gives 3.71 mm where the true spacing is 5.0 mm, a 26%
error, and stacking the duplicated images repeats anatomy so that organs occupy more
slices than they physically do. Both errors inflate organ volume, and neither leaves any
trace in the images. We therefore resolve the grid explicitly: one image per position,
spacing from the median step between neighbouring positions, and a uniformity check that
refuses a series whose steps vary by more than 2%. Two series proved to be a pair of
reconstructions interleaved under a single Series Instance UID; for these the largest
regular sub-grid was taken, accepted only when it preserves the full longitudinal extent,
which distinguishes an interleaved pair from a series with a genuine gap.

Second, the ordering of the slice axis is not guaranteed. Slice Location (0020,1041) runs
opposite in sign to Image Position (Patient) on three of the four manufacturers in this
cohort, so a series sorted by the former arrives head-first. The segmentation is
unaffected, because the geometry handed to the segmenter is built from patient
coordinates; but the array index ceases to mean "towards the head", which reverses every
reported organ extent and any downstream reasoning about superior and inferior. Volumes
are therefore canonicalised so that index zero is the most inferior slice, with the tube
current reordered alongside, since I(z) is paired to the slice axis by index.

### 2.4. Reading Hounsfield Units

Outside the reconstruction circle an image carries a padding value rather than a
measurement, and that value must be replaced with air before anything is measured. The
attribute naming it, Pixel Padding Value (0028,0120), has a value representation that
depends on Pixel Representation, and this is not reliably honoured: one manufacturer
writes 63,536 with an unsigned representation on signed pixel data, which is the two's
complement encoding of the −2000 intended. Read literally, that places the padding
threshold above every Hounsfield value in the image, and the volume becomes uniform air.
Four of the ten GE series in this cohort were destroyed this way, with no symptom other
than a segmenter returning empty masks. We reinterpret the padding value against Pixel
Representation, and additionally refuse any padding rule that would blank essentially the
whole image — the second safeguard being the more durable one.

### 2.5. Segmentation

Twelve dose-relevant abdominal organs were segmented with TotalSegmentator v2.17 [1]
using the 1.5 mm full-resolution model, at inference only; no weights were trained,
modified or redistributed. Inference runs in a separate child process, because nnU-Net
spawns its own workers and doing so from a long-lived parent leaks processes on Windows.

The series is written to NIfTI by our own code, with an affine constructed from the DICOM
patient coordinates, and the masks return on that same grid; the correspondence between
mask voxel and image voxel is therefore the identity by construction, and is asserted
rather than assumed. This matters because the alternative — inferring an axis permutation
after the fact — fails silently: a mirrored segmentation produces entirely plausible
volumes, masses and Hounsfield values while pairing every organ with the wrong anatomy.

The patient outline, needed for the water-equivalent diameter, was taken from a
deterministic threshold contour (air/tissue threshold, largest connected component, hole
filling) following AAPM Report 220 [7], which requires no model weights and is
reproducible across platforms.

### 2.6. Organ Mass from Measured Attenuation

Hounsfield units were converted to mass density by piecewise-linear interpolation through
reference tissue anchor points — air, inflated lung, adipose, water, skeletal muscle,
trabecular and cortical bone — taking the densities from ICRU Report 44 [8] and the
construction from Schneider et al. [9]. Organ mass is the sum of local density over mask
voxels multiplied by the voxel volume, not the organ volume multiplied by a nominal
tissue density: a steatotic and a normal liver of equal volume do not have equal mass, and
capturing that difference is the point of using the patient's own attenuation.

The calibration is a model and is treated as one. Its anchor Hounsfield values are nominal
for a 120 kVp acquisition, it is replaceable by a scanner-specific calibration, and
whichever curve was used travels into the provenance of every mass. Abdominal soft tissue
is in any case insensitive to the choice, because every reasonable curve passes through
water at 0 HU; perturbing the water-to-muscle slope by 10% changes an abdominal organ mass
by less than 1%.

### 2.7. The Organ-Specific Weighted CTDIvol

For an organ *o* occupying slices with per-slice voxel counts *n_o(z)*, and a series with
per-slice tube current *I(z)*, the modulation weight is

*w_o* = [ Σ_z *n_o(z) I(z)* / Σ_z *n_o(z)* ] / mean_z *I(z)*     (1)

and the organ-specific weighted CTDIvol is CTDIvol · *w_o*. The weight is dimensionless
and is the transferable quantity: it expresses what the modulation contributes,
independently of the scanner's own output.

CTDIvol was taken from the image header (0018,9345) where present. Where absent, it was
reconstructed from acquisition physics against an openly licensed normalised-CTDI
database [4]; recorded and reconstructed values are never merged, and each series records
which it carries, because the uncertainty attaching to them differs. A recorded value
outside the physically possible range is treated as a corrupt attribute and falls through
to reconstruction — one Philips series records CTDIvol as −3.7 × 10^19 mGy, and taking
that at face value would have produced an absurd organ dose carrying full provenance.

An organ whose mask reaches the first or last slice of the series continues beyond the
scan. Its mass is the mass of the scanned part and its weight describes only the exposed
part; neither is the organ's. Such organs are flagged and excluded from whole-organ
comparisons rather than reported as small organs.

### 2.8. Verification and Reproducibility

Every series is screened against facts of gross anatomy that hold for any adult: the left
kidney lies to the patient's left of the right kidney and the spleen to the left of the
liver; the liver lies superior to the bladder and the adrenal glands superior to the
kidneys; solid-organ masses fall within a wide band of reference values; and the organ
weights vary within a series. These screens exist because the failures they catch are the
ones that leave no other trace.

The engine is MIT-licensed. Every reported value is re-derived from the per-series records
by the test suite, including regenerating the complete analysis table and comparing it, so
that a hand-edited summary fails. Results and figures are regenerated by scripts in the
repository.

## 3. Results

### 3.1. Cohort

Forty series were analysed — ten from each of GE, Siemens, Canon/Toshiba and Philips —
drawn from 21 collections and 23 scanner models, yielding 455 organ records across 12
organs.

### 3.2. Segmented Organ Mass Against a Published Reference

Table 1 and Figure 1 give segmented organ mass relative to the ICRP 89 reference adult
male [5], over organs not truncated by the scan boundary.

**Table 1.** Segmented organ mass against ICRP 89 reference values, whole organs only.

| organ | n | median mass | ratio to ICRP 89 |
|---|---|---|---|
| liver | 34 | 1901 g | 1.06 |
| spleen | 38 | 259 g | 1.72 |
| kidney (left) | 34 | 179 g | 1.15 |
| kidney (right) | 35 | 182 g | 1.17 |
| pancreas | 36 | 86 g | 0.61 |

![**Figure 1.** Segmented organ mass relative to the ICRP 89 reference adult male, by manufacturer. Each marker is one organ in one series; the horizontal bar is the median across all manufacturers. Organs truncated by the scan boundary are excluded, since their mass is that of the scanned part only. Manufacturer is encoded by marker shape as well as colour, so the figure is readable in greyscale.](figures/fig1_organ_mass_vs_icrp89.png){width=100%}

The liver, the largest organ and the one contributing most mass, agrees with the reference
to within 6%, and the kidneys to within 17%. Two organs depart systematically: the
pancreas is low by 39% and the spleen high by 72%. Both are reported as measured; neither
was corrected. Their attribution is taken up in Section 4.

### 3.3. Availability of a Dose Index, by Manufacturer

Figure 2 gives the availability of a whole-scan dose index. Across the cohort, 29 of 40
series carried a recorded CTDIvol, 5 were reconstructable from acquisition physics, and 6
were neither.

![**Figure 2.** Availability of a whole-scan dose index by manufacturer. A series counted unrecoverable carries no CTDIvol in its header and its scanner lies outside the open coefficient database, so no organ-level index can be formed from it at all. Segments are distinguished by fill pattern as well as tone.](figures/fig2_dose_index_availability.png){width=100%}

All 6 unrecoverable series are GE, and GE recorded a CTDIvol in none of its ten series
(GE 0/10); the other three manufacturers recorded one in 29 of 30 (Fisher exact test,
two-sided, *p* = 1 × 10^−8). Four of the ten GE series were rescued by physics
reconstruction; the remaining six carry scanners outside the open coefficient database and
therefore admit no dose index at all. For those six, the organ masks, volumes, masses and
modulation weights are all computable and are reported — but no organ-level dose index
exists for them, and no coefficient set, however licensed, would change that.

### 3.4. The Organ-Level Index

Organ modulation weights across the cohort span 0.59 to 1.69. Within a single acquisition
the organ-level index therefore departs from the whole-scan index by up to roughly 70% in
either direction, which is the magnitude the method exists to capture.

Figure 3 shows one acquisition end to end: a Canon/Toshiba Aquilion PRIME series of 268
slices with a recorded CTDIvol of 16.1 mGy and a scan mean tube current of 265 mA. The
small bowel and colon, lying in the inferior abdomen where the modulation raised the
current to about 447 and 435 mA, take weights of 1.69 and 1.64 respectively, giving
organ-level indices near 27 mGy; the left kidney and stomach, higher in the scan, take
weights of 0.94 and indices near 15 mGy. Two organs in the same acquisition thus differ by
a factor of 1.8 in the exposure they receive, a difference no whole-scan number can
express.

![**Figure 3.** One acquisition end to end. (**a**) Each organ's longitudinal extent, annotated with the mean tube current recorded over it, against a scan mean of 265 mA. (**b**) The resulting organ-specific weighted CTDIvol, with the modulation weight beside each bar; the dashed line is the whole-scan CTDIvol of 16.1 mGy.](figures/fig3_demonstration_case.png){width=100%}

### 3.5. What Limits an Organ-Level Study

Two conditions, neither a defect in the data nor in the method, reduce what an organ-level
study can measure, and both are vendor-dependent in this cohort (Figure 4).

Truncation by the scan boundary affected 5.2% of organ records on GE, 5.5% on Siemens,
10.4% on Canon/Toshiba and 20.0% on Philips. The organs most often cut are the colon and
small bowel, which are longitudinally extensive, followed by the liver on Philips.

![**Figure 4.** What limits an organ-level modulation study. (**a**) Percentage of organ records truncated by the scan boundary, by manufacturer, annotated with the counts. (**b**) Peak-to-peak spread of the organ weights within each series; the dashed line is the threshold below which a series carries no usable variation across its organs.](figures/fig4_study_limits.png){width=100%}

3 of the 40 series showed a peak-to-peak spread of organ weights below 0.02: their tube
current does not vary across the abdominal organs, so the modulation weighting has nothing
to express. These are genuine acquisitions and passed the modulation screen at selection,
where the current varies across the whole scan; the flatness is local to the abdomen. They
are uninformative for an organ-level modulation study rather than faulty, and are counted
as such.

## 4. Discussion

**What the organ layer adds.** The organ modulation weights span 0.59 to 1.69, so
attributing the whole-scan index to every organ misstates the exposure of some organs by
tens of percent within a single acquisition. That is the quantitative case for the organ
layer, and it is obtainable from data every scanner already writes.

**The pancreas offset is attributed to the segmentation.** The pancreas is the weakest of
the solid abdominal organs in TotalSegmentator's own validation: Dice 0.887, against 0.965
for the liver, 0.983 for the spleen and 0.953 and 0.939 for the kidneys [1]. A thin,
low-contrast organ with a variable course is precisely the case a conservative boundary
under-segments, and a deficit of this size, consistent across four manufacturers and 36
whole organs, is what that produces. One caveat belongs in the open: Dice is symmetric and
does not establish the *sign* of a disagreement, so it supports lower boundary agreement
but not, by itself, under-segmentation. Separating model bias from cohort anatomy requires
a second segmentation model on the same series, which we defer to future work.

**The spleen offset is attributed to the cohort.** Three independent lines support this.
The four largest spleens — 4.47, 3.74, 3.34 and 3.06 times the reference mass — were
reviewed slice by slice against their own images: each contour follows the splenic
boundary with correct laterality, tracks the notch at the hilum, and shows no leakage into
liver, kidney or stomach, and in each case the mask is a single connected component, so no
accessory spleen was included. Their mean densities are 1.052 to 1.078 g/cm³ — that is,
normal splenic tissue — so the elevated mass is entirely volume-derived rather than an
artefact of the density calibration. And the spleen is the *best*-segmented organ in the
segmenter's validation (Dice 0.983), which makes a segmentation explanation the least
likely one available. The four arise on four different manufacturers, excluding a vendor
artefact. The cohort is oncological — renal cell, colorectal and adrenal carcinoma — in
which splenomegaly is common, and the ICRP reference adult is not that population. The
+72% median is cohort anatomy.

**Availability at the organ level.** That one manufacturer records no CTDIvol in any
series, while the others record it almost universally, is consequential for anyone
attempting retrospective organ dosimetry on archived data. Physics reconstruction rescues
some of those series, but only where the scanner appears in an open coefficient database;
in this cohort it rescued four of ten, leaving six with no dose index by any route. An
organ-level dosimetry study drawn from an archive will therefore lose a
manufacturer-dependent fraction of its cohort before segmentation is even considered, and
a study that does not report which series were lost will silently under-represent that
manufacturer.

**Where the open chain ends, and why.** Converting the organ-specific weighted CTDIvol
into an absorbed organ dose in milligray requires CTDIvol-normalised coefficients with a
patient-size correction, of the kind established by Turner et al. [2] and extended over
larger patient model libraries by Tian et al. [3]. The coefficient sets the field relies
on for this purpose are published in subscription journals or distributed with research
software under terms granting use but not redistribution. This reflects a publishing
convention for Monte-Carlo reference data rather than any deficiency in the coefficients
themselves, which are well validated and widely applied; it simply means that no such set
can be carried inside an openly licensed pipeline.

The contrast with the layer below is instructive. Normalised-CTDI data of the kind used
here to reconstruct a missing whole-scan index has been published under CC BY [4], and
that single change is what makes the reconstruction in Section 3.3 possible at all. The
same change at the organ layer would extend the open chain to absorbed dose without any
methodological advance. The boundary is therefore a property of how this class of
reference data is currently published, not of the method — and it is movable.

We accordingly stop at the index, and the software enforces that: it refuses to produce a
dose in milligray unless supplied with a coefficient table carrying its citation, DOI,
licence and source hash. This is the openness gap the companion study documented for the
whole-scan index [6], observed one level down: the measurement chain is open from the
scanner to the organ-level index, and closed for the final step. Computing coefficients
with an open-source Monte-Carlo engine would remove the barrier rather than work around
it, and is the natural continuation of this work.

**Limitations.** Ten series per manufacturer supports a median and an interquartile range,
not a distributional claim. The cohort is oncological and not a reference population, which
is directly relevant to the spleen finding and probably to the liver spread. A single
segmentation model was used, so segmentation bias and cohort anatomy cannot be fully
separated here. There is no independent ground truth for organ mass in these subjects — the
ICRP comparison is an anchor, not a gold standard. And no absorbed dose is reported, by the
deliberate choice set out above.

## 5. Conclusions

Patient-specific, organ-level CT dose characterisation is achievable across all four major
manufacturers from data that routine acquisitions already record, using inference-only
deep-learning segmentation and openly licensed inputs throughout. The modulation weights
span 0.59 to 1.69 within this cohort, so the organ layer is not a refinement of the
whole-scan index but a materially different quantity. The segmented anatomy underpinning
it agrees with published reference masses for the liver and kidneys, with two systematic
offsets that we characterise and attribute rather than adjust.

The chain stops at the index, and where it stops is worth stating plainly. Monte-Carlo
organ-dose coefficients are conventionally published under terms that permit use but not
redistribution, so they cannot be carried inside an openly licensed pipeline. That is a
property of current publishing practice for this class of reference data rather than of
the coefficients, which are sound, or of the method, which is complete up to that point.
The precedent one layer down — normalised-CTDI data released under CC BY, which is what
makes the whole-scan reconstruction reported here possible — shows the convention can
change. Until it does, the organ-level index is where an open chain ends, and it is worth
having on its own terms.

---

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
licence at https://github.com/Institute-of-One/ctsegdose-core and archived at Zenodo
(version DOI to be inserted at release). The machine-readable results the manuscript
quotes, the per-organ records, the analysis tables and the figure scripts are included in
that repository, together with the mask-review overlays underlying Section 4.

No DICOM imaging is redistributed. Every series analysed is identified in
`data/PROVENANCE.json` by collection, collection DOI, Series Instance UID, manufacturer,
model, licence and retrieval date, and is retrievable directly from The Cancer Imaging
Archive by following `docs/REPRODUCING_DATA.md`. All forty series are published under
Creative Commons Attribution licences. Segmentation used TotalSegmentator [1], which is
distributed under the Apache 2.0 licence and was used at inference only; no model weights
are redistributed here. No organ-dose coefficient table is redistributed, for the
licensing reasons set out in Section 4.

## Conflicts of Interest

The author is the representative of LISIT Co., Ltd. (Tokyo, Japan) and Chief Executive
Officer of TexelCraft OÜ (Estonia), and has a commercial interest in downstream products
that may incorporate or build upon the methods described here. The software reported in
this article is released under the MIT licence. No patient data, customer data or
proprietary clinical data were used in this study; all imaging is publicly available and
de-identified. The funders had no role in the design of the study, in the collection,
analysis or interpretation of data, in the writing of the manuscript, or in the decision
to publish the results.

## Use of Generative Artificial Intelligence

A generative AI assistant (Claude, Anthropic) was used in the development of the software
and in drafting and editing the text of this manuscript. All reported results are produced
by executable code contained in the cited repository, are re-derived from the underlying
per-series records by an automated test suite, and were verified by the author, who takes
full responsibility for the content of this article.

---

## References

1. Wasserthal, J.; Breit, H.-C.; Meyer, M.T.; Pradella, M.; Hinck, D.; Sauter, A.W.; Heye,
   T.; Boll, D.T.; Cyriac, J.; Yang, S.; Bach, M.; Segeroth, M. TotalSegmentator: Robust
   Segmentation of 104 Anatomic Structures in CT Images. *Radiol. Artif. Intell.* **2023**,
   *5*, e230024. https://doi.org/10.1148/ryai.230024
2. Turner, A.C.; Zhang, D.; Khatonabadi, M.; Zankl, M.; DeMarco, J.J.; Cagnon, C.H.; Cody,
   D.D.; Stevens, D.M.; McCollough, C.H.; McNitt-Gray, M.F. The Feasibility of Patient
   Size-Corrected, Scanner-Independent Organ Dose Estimates for Abdominal CT Exams. *Med.
   Phys.* **2011**, *38*, 820–829. https://doi.org/10.1118/1.3533897
3. Tian, X.; Li, X.; Segars, W.P.; Frush, D.P.; Paulson, E.K.; Samei, E. Dose Coefficients
   in Pediatric and Adult Abdominopelvic CT Based on 100 Patient Models. *Phys. Med. Biol.*
   **2013**, *58*, 8755–8768. https://doi.org/10.1088/0031-9155/58/24/8755
4. Dinwiddie, L.E.; Baggett, J.M.; Kofler, J.M.; et al. Survey of Normalized CTDIvol Values
   Across Four Major Computed Tomography Vendors for Use in the MIRDct Software. *J. Appl.
   Clin. Med. Phys.* **2026**, *27*, e70473. https://doi.org/10.1002/acm2.70473
5. ICRP. Basic Anatomical and Physiological Data for Use in Radiological Protection:
   Reference Values. ICRP Publication 89. *Ann. ICRP* **2002**, *32* (3–4).
6. Yamamoto, S. ctdose-core: Open, Auditable CT Dose Surveillance from DICOM, with a
   Physics Reconstruction When the Dose Attributes Are Missing (Software). Zenodo,
   **2026**. https://doi.org/10.5281/zenodo.21636082
7. McCollough, C.; Bakalyar, D.M.; Bostani, M.; Brady, S.; Boedeker, K.; Boone, J.M.;
   Chen-Mayer, H.H.; Christianson, O.I.; Leng, S.; Li, B.; et al. Use of Water Equivalent
   Diameter for Calculating Patient Size and Size-Specific Dose Estimates (SSDE) in CT: The
   Report of AAPM Task Group 220. *AAPM Report No. 220*, **2014**.
   https://doi.org/10.37206/146
8. ICRU. Tissue Substitutes in Radiation Dosimetry and Measurement. ICRU Report 44;
   International Commission on Radiation Units and Measurements: Bethesda, MD, USA, **1989**.
9. Schneider, U.; Pedroni, E.; Lomax, A. The Calibration of CT Hounsfield Units for
   Radiotherapy Treatment Planning. *Phys. Med. Biol.* **1996**, *41*, 111–124.
   https://doi.org/10.1088/0031-9155/41/1/009
