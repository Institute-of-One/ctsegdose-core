# Pre-submission checklist — IORN-006, *Tomography* (MDPI)

Everything the manuscript needs that cannot be determined from the repository alone.
Nothing on this list appears in the submission PDF: `tools/build_submission.py` **refuses
to build the submission variant** while any of it is outstanding, so a placeholder cannot
reach a reviewer by accident.

## 1. Release fields — author action required

Create the GitHub release and the Zenodo deposition, then record the values in
`paper/release_metadata.json`:

```json
{
  "RELEASE_TAG": "v0.1.0",
  "COMMIT_HASH": "<40-character hash of the tagged commit>",
  "ZENODO_VERSION_DOI": "10.5281/zenodo.XXXXXXXX"
}
```

Then:

```bash
python tools/build_submission.py          # submission variant; refuses if anything is missing
```

Until that file exists, build the working copy instead:

```bash
python tools/build_submission.py --internal
```

which marks each unresolved field in place as `[[ pending: NAME ]]`.

**Note on the Zenodo DOI.** The *version* DOI is required here, not the concept DOI. The
concept DOI `10.5281/zenodo.21636082` in reference [6] belongs to the companion project
`ctdose-core` and is correct as cited; do not reuse it for this work.

## 2. Confirmed and requiring no further action

| item | status |
|---|---|
| Repository URL | `https://github.com/Institute-of-One/ctsegdose-core` — must be **public** before submission |
| Software licence | MIT |
| Python | 3.14 used; package supports 3.10+ |
| TotalSegmentator | v2.17, `total` task, 1.5 mm full-resolution model |
| PyTorch / CUDA | 2.11 / 12.8, NVIDIA RTX 3080 |
| pydicom / NumPy | 3.0 / 2.5 |
| Regeneration commands | in Section 2.10 |
| TCIA licences | 33 × CC BY 4.0, 7 × CC BY 3.0, recorded per series in `data/PROVENANCE.json` |
| Reference [6] metadata | verified against the Zenodo record (title, author, type, year, version 0.1.1) |

## 3. Author decisions still open

**3.1 One series where the weighting assumption does not hold.**
The GE series ending `399991479763` (Anti-PD-1_Lung, Discovery CT750 HD) contains two
contiguous blocks at 0.4 s / 0.8 s rotation time and 400 ms / 800 ms exposure time. Its
scanner output is therefore not proportional to tube current alone. Re-weighting it by the
current–time product shifts its organ weights by up to about 35%.

It is currently **retained and disclosed** in Sections 2.7 and 4. Its weights (0.98–1.21)
are interior to the reported range, which is 0.59–1.69 with or without it, so no reported
result depends on the choice. The alternatives are to exclude the series (cohort becomes
39, GE 9) or to adopt a current–time weighting throughout — the latter is **not possible
uniformly**, because exposure time is absent from the archived headers of the Philips
series ending `687268266165`. Confirm the disclosure approach or choose otherwise.

**3.2 Optional sensitivity analysis, computed but not incorporated.**
Estimated mass from the HU-density model versus segmented volume × 1.05 g/cm³ nominal
soft-tissue density, over the 177 reference-comparison records: median difference **+1.0%**
(range −4.1% to +10.4%), with ICRP ratios essentially unchanged (liver 1.06→1.05, spleen
1.72→1.71, pancreas 0.61→0.61). This shows both offsets are volume-driven rather than
artefacts of the density model. Recommended for inclusion as it pre-empts an obvious
reviewer question; say so and it will be added to `analysis.py` with a test.

**3.3 Suggested reviewers.** Five proposed, with affiliations to be verified before entry
into SuSy. Not recorded here; see the working notes.

## 4. Submission logistics

- Journal: *Tomography* (MDPI), article type **Article**, regular submission (not a
  special issue). Single-blind review, so the manuscript is **not** masked.
- Files: `paper/manuscript_tomography.pdf` and `.docx`, plus the four figures if requested
  separately (`paper/figures/fig1..fig4.{png,pdf}`).
- APC 2,400 CHF, payable only on acceptance.
- ORCID linked; corresponding author yamamoto@lisit.jp.
