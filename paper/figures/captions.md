**Figure 1.** Representative TotalSegmentator output from one abdominal CT series used in the analysis, shown as translucent overlays on the acquisition's own images. The same acquisition appears in Figure 2. Images are displayed in radiological convention (patient left on the viewer's right); only de-identified imaging from The Cancer Imaging Archive is shown.

**Figure 2.** One acquisition end to end: the segmented organs, the recorded per-slice tube current over each organ's longitudinal extent, and the anatomy-weighted CTDIvol index the modulation produces.

**Figure 3.** Availability of a whole-scan dose index in the archived DICOM headers, by manufacturer. 40 abdominal CT series, 455 organ records, 4 manufacturers. A series counted unrecoverable retained no CTDIvol in its header and its scanner lies outside the open coefficient database, so no anatomy-weighted index can be formed from it at all.

**Figure 4.** Attenuation-derived estimated organ mass relative to the ICRP 89 reference adult male value. 40 abdominal CT series, 455 organ records, 4 manufacturers. Organs truncated by the scan boundary are excluded, since their mass is that of the scanned part only. The reference is an external anchor, not a ground truth for these subjects.

**Figure 5.** What limits an organ-level modulation analysis. 40 abdominal CT series, 455 organ records, 4 manufacturers. Truncation is the fraction of organ records the scan boundary cuts through; the weight spread is the peak-to-peak range of the organ weights within a series.

Figure 3 demonstration series: Canon/Toshiba, Aquilion PRIME, collection CMB-OV, Series Instance UID 1.3.6.1.4.1.14519.5.2.1.235815435552836713008925930681449087181.
