"""Make the second-proof changes in the manuscript itself, as the editor asked.

The first proof was answered with replies and a change list. The production editor
came back with "To avoid any potential errors, could you kindly make the necessary
changes directly in the manuscript?" and, for the abstract, "Please add them
directly to the main text of the manuscript." So the edits go into the document.

Two rules govern everything here.

*Nothing the English Editor changed is reverted.* The abstract is rebuilt, and it
is rebuilt on the text with their changes accepted -- the removed comma, "the"
for "The", the inserted "and", the inserted "an" -- not on the version in this
repository, which predates them.

*Every edit is asserted.* A replacement that does not match raises rather than
passing silently, because a silent miss here reaches print.

Usage::

    python tools/apply_second_proof_edits.py \\
        --proof docs/proof/tomography-4516935-for2ndproof.docx \\
        --out docs/proof/tomography-4516935-2nd-proofread.docx
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

#: The abstract, rebuilt. Built from the proof's text with the English Editor's
#: changes accepted, plus what production and the English Editor asked for: the
#: four subheadings, and CT, CTDIvol and ICRP defined at first use.
ABSTRACT = [
    ("b", "Background/Objectives: "),
    (
        "",
        "The volume computed tomography (CT) dose index (CTDIvol) is a scanner "
        "output, not an organ dose, and cannot express how tube-current modulation "
        "varies along a patient. An organ-specific weighted CTDIvol addressing this "
        "has been reported before, in single-institution cohorts and often from "
        "inputs routine archives do not retain. New here is not the quantity but "
        "what an open, multi-vendor operationalisation reveals: whether its inputs "
        "survive archive curation and what the fallback costs when they do not. ",
    ),
    ("b", "Methods: "),
    (
        "",
        "Forty abdominal CT series, ten per manufacturer, were drawn from the Cancer "
        "Imaging Archive and twelve organs segmented with TotalSegmentator at "
        "inference. Of 480 requested organ–series combinations, 455 were "
        "produced. A rule-based acquisition-constancy criterion admitted 39 series. ",
    ),
    ("b", "Results: "),
    (
        "",
        "Modulation weights spanned 0.59 to 1.69, so the index departs from the "
        "whole-scan CTDIvol by up to 70% within one acquisition. A recorded CTDIvol "
        "survived in 29 of 40 archived headers and was reconstructable in 5 and "
        "unavailable in 6, availability differing markedly between manufacturers. "
        "Forcing that reconstruction on series that did retain a value agreed to "
        "within 12% on three scanner models and diverged by 58% and 84% on two "
        "others. Estimated organ mass was broadly consistent with International "
        "Commission on Radiological Protection (ICRP) Publication 89 for liver and "
        "kidneys. ",
    ),
    ("b", "Conclusions: "),
    ("", "This index is not an absorbed dose; the implementation is open."),
]

#: (description, text as it stands in one run, replacement). Each must match exactly
#: once across the document.
REPLACEMENTS = [
    (
        "Introduction: define CT at first use",
        "conflated when CT dose is discussed",
        "conflated when computed tomography (CT) dose is discussed",
    ),
    (
        "Introduction: define CTDIvol at first use",
        ": CTDIvol describes the output of a scanner",
        ": the volume CT dose index (CTDIvol) describes the output of a scanner",
    ),
    (
        "Contribution 5: define ICRP at first use in the main text",
        "against ICRP 89 values",
        "against International Commission on Radiological Protection (ICRP) "
        "Publication 89 values",
    ),
    (
        "Table 1 caption: define ICRP in the first table",
        "beside ICRP 89 reference adult male",
        "beside International Commission on Radiological Protection (ICRP) "
        "Publication 89 reference adult male",
    ),
    (
        "Data availability: release tag",
        "1 (commit 46edee6423ddc19b47efae180d8edf527299d066), archived at Zenodo under ",
        "2 (commit 0ca9d57ef9fbc1a321cd5b671d21eb5857def518), archived at Zenodo under ",
    ),
    (
        "Data availability: version DOI",
        "10.5281/zenodo.21817307",
        "10.5281/zenodo.22143005",
    ),
]

#: Emphasis to drop, as answered on the first proof. Only these four words and three
#: phrases: the rest of the italics and bold are terminology, variables or run-in
#: headings, and the replies say why each is kept.
DROP_ITALIC = {"not", "every", "longitudinal", "direction"}
#: "weighted organ-specific CTDIvol" is two runs in the file, "weighted" and
#: "organ-specific CTDIvol", so it is listed as the two fragments it actually is.
DROP_BOLD = {
    "The quantity examined here is therefore not new, and no new index is proposed.",
    "open, multi-vendor operationalisation and empirical characterisation",
    "weighted",
    "organ-specific CTDIvol",
}


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _run(text: str, *, bold: bool) -> str:
    props = "<w:snapToGrid w:val=\"0\"/><w:szCs w:val=\"20\"/><w:lang w:val=\"en-GB\"/>"
    if bold:
        props = "<w:b/>" + props
    return (
        f'<w:r w:rsidRPr="00DA0038"><w:rPr>{props}</w:rPr>'
        f'<w:t xml:space="preserve">{_escape(text)}</w:t></w:r>'
    )


def rebuild_abstract(document: str) -> str:
    """Replace the abstract paragraph's runs, keeping its paragraph properties."""
    for match in re.finditer(r"<w:p\b[^>]*>.*?</w:p>", document, re.S):
        para = match.group(0)
        if "CTDIvol is a scanner-output index" not in re.sub(r"<[^>]+>", "", para):
            continue
        open_tag = para[: para.index(">") + 1]
        ppr = re.search(r"<w:pPr>.*?</w:pPr>", para, re.S)
        if not ppr:
            raise SystemExit("the abstract paragraph has no properties to preserve")
        runs = "".join(_run(text, bold=(kind == "b")) for kind, text in ABSTRACT)
        return document[: match.start()] + open_tag + ppr.group(0) + runs + "</w:p>" + document[match.end() :]
    raise SystemExit("the abstract paragraph was not found")


def drop_emphasis(document: str) -> tuple[str, int, int]:
    """Remove <w:i/> and <w:b/> from runs whose whole text is on the drop lists.

    Returns the modified document as well as the counts. The first version returned
    only the counts, so the caller reported four italics and two bold removed and
    wrote out a document in which nothing had changed -- a success message over an
    unchanged file, which is worse than an error.
    """
    italics = bold = 0

    def rewrite(match: re.Match) -> str:
        nonlocal italics, bold
        run = match.group(0)
        parts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", run, re.S)
        text = "".join(parts).strip()
        if text in DROP_ITALIC and re.search(r"<w:i/>", run):
            italics += 1
            return re.sub(r"<w:i/>", "", run, count=1)
        if text in DROP_BOLD and re.search(r"<w:b/>", run):
            bold += 1
            return re.sub(r"<w:b/>", "", run, count=1)
        return run

    document = re.sub(r"<w:r\b[^>]*>.*?</w:r>", rewrite, document, flags=re.S)
    return document, italics, bold


def build(proof: Path, out: Path) -> None:
    with zipfile.ZipFile(proof) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    document = parts["word/document.xml"].decode("utf-8")

    for description, old, new in REPLACEMENTS:
        count = document.count(_escape(old))
        if count != 1:
            raise SystemExit(f"{description}: matched {count} times, expected exactly 1")
        document = document.replace(_escape(old), _escape(new), 1)
        print(f"  applied: {description}")

    document = rebuild_abstract(document)
    print("  applied: abstract rebuilt with the four subheadings and the definitions")

    document, italics, bold = drop_emphasis(document)
    if (italics, bold) != (len(DROP_ITALIC), len(DROP_BOLD)):
        raise SystemExit(
            f"expected {len(DROP_ITALIC)} italic and {len(DROP_BOLD)} bold removals, "
            f"got {italics} and {bold}"
        )
    print(f"  applied: removed {italics} italic and {bold} bold emphasis runs")

    parts["word/document.xml"] = document.encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    print(f"  wrote {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--proof", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    build(args.proof, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
