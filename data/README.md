# data/

Downloaded imaging lives here and **is not published**. `.gitignore` excludes everything
in this directory except this file and `PROVENANCE.json`.

Layout after `tools/select_and_download.py --stage download`:

```
data/
  PROVENANCE.json                     committed: what was fetched, from where, under which licence
  GE/<collection>__<patient>/<series-uid>/*.dcm       git-ignored
  Siemens/...
  Canon_Toshiba/...
  Philips/...
```

`PROVENANCE.json` is the dataset as far as a reader is concerned: it names the archive,
the collection and its DOI, the Series Instance UID, the manufacturer and model, the
licence, and the retrieval date for every series. That is enough to fetch exactly the
same imaging from The Cancer Imaging Archive. See `docs/REPRODUCING_DATA.md`.
