"""No imaging leaves this repository.

The rule is a standing condition of the openly-licensed data used here: process
locally, publish provenance, redistribute nothing. A test enforces it because a stray
``git add -A`` is all it takes to break it.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IMAGING_SUFFIXES = {".dcm", ".ima", ".nii", ".gz", ".zip"}

#: Working directories that hold derived or downloaded artefacts and are git-ignored:
#: the archive cache, the segmentation masks, the virtual environments. Imaging is
#: *expected* here; what must not happen is imaging appearing anywhere else.
WORKING_DIRS = {".git", ".tcia_work", "segmentations", ".venv", ".venv-gpu", "data"}


def test_no_imaging_file_sits_anywhere_in_the_working_tree_outside_data():
    strays = [
        p
        for p in REPO.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGING_SUFFIXES
        and not WORKING_DIRS.intersection(p.relative_to(REPO).parts)
    ]
    assert not strays, f"imaging outside the git-ignored working directories: {strays}"


def test_the_ignore_rules_keep_the_downloaded_series_out_of_the_repository():
    text = (REPO / ".gitignore").read_text(encoding="utf-8")
    for rule in ("data/**", "!data/PROVENANCE.json", "*.dcm", ".tcia_work/"):
        assert rule in text, f"missing .gitignore rule: {rule}"


def test_the_affiliation_convention_holds_in_the_published_metadata():
    """Institute of One, LISIT Co., Ltd. is the affiliation everywhere; no other appears."""
    for name in ("CITATION.cff", ".zenodo.json", "README.md"):
        text = (REPO / name).read_text(encoding="utf-8")
        assert "Institute of One, LISIT Co., Ltd." in text
        assert "National Cancer Center" not in text
        assert "NCC" not in text
