# results/

Machine-readable outputs. Everything a manuscript quotes is computed into this
directory and re-derived from its own rows by `tests/test_results_integrity.py`.

| file | written by | contents |
| --- | --- | --- |
| `candidates.json` | `--stage plan` | the screened candidate pool, the collections queried, and a tally of every metadata rejection by reason |
| `verification.json` | `--stage verify` | per-series header evidence (tube currents probed, rescale, geometry) and the keep/drop verdict each implies |

`data/PROVENANCE.json` completes the set: what was actually retrieved, from where, and
under which licence.
