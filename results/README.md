# results/

Machine-readable outputs. Everything a manuscript quotes is computed into this
directory and re-derived from its own rows by `tests/test_results_integrity.py`.

| file | written by | contents |
| --- | --- | --- |
| `candidates.json` | `--stage plan` | the screened candidate pool, the collections queried, and a tally of every metadata rejection by reason |
| `verification.json` | `--stage verify` | per-series header evidence (tube currents probed, rescale, geometry) and the keep/drop verdict each implies |
| `organ_dose_<tag>.json` | `tools/run_organ_dose.py` | per organ: volume, HU-derived mass, mask centroid, modulation weight, organ-specific weighted CTDIvol, truncation flag, and the formulae behind each |
| `organ_dose/<tag>/<uid>.json` | same | one file per series, written as it completes so an interrupted batch resumes |
| `segmentation_checks_<tag>.json` | `tools/check_segmentation.py` | the anatomical screens per series, with what failed and what could not be checked |

`<tag>` is the segmentation model resolution (`1.5mm`, `3mm`), so runs at different
resolutions never overwrite each other and can be compared.

`data/PROVENANCE.json` completes the set: what was actually retrieved, from where, and
under which licence.
