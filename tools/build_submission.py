"""Build the Tomography submission files from ``paper/manuscript.md``.

Usage::

    python tools/build_submission.py

Writes ``paper/manuscript_tomography.docx`` and ``paper/manuscript_tomography.pdf``.

MDPI typesets accepted articles from their own template, so a submission does not need
one; what it needs is a clean, complete, readable document with the figures in place.
Pandoc is used through ``pypandoc_binary`` and the PDF through Typst, so the whole
toolchain installs with pip and needs neither a system LaTeX nor administrator rights —
which matters for a paper whose selling point is that a reader can reproduce it.

The build refuses to run if the manuscript's consistency tests fail, because a document
whose numbers have drifted from the results is exactly what must not reach a reviewer.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"
SOURCE = PAPER / "manuscript.md"
STEM = "manuscript_tomography"

#: Stripped before building: the file's internal note to its authors is not part of the
#: article, and a reviewer should not receive it.
HTML_COMMENT = re.compile(r"<!--.*?-->\s*", re.S)


def check_consistency() -> None:
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "pytest", "-q",
         str(REPO / "tests" / "test_manuscript_consistency.py"),
         str(REPO / "tests" / "test_analysis_integrity.py")],
        cwd=str(REPO), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        raise SystemExit(
            "the manuscript's consistency tests fail; the numbers in the text no longer "
            "follow from results/. Fix that before building a submission."
        )
    print("  consistency tests pass")


def prepare(source: Path, staged: Path) -> Path:
    text = HTML_COMMENT.sub("", source.read_text(encoding="utf-8"), count=1)
    # Thematic breaks separate sections while drafting; in a typeset document the
    # headings already do that, and Typst has no use for them.
    text = re.sub(r"^-{3,}\s*$", "", text, flags=re.M)
    staged.write_text(text, encoding="utf-8")
    return staged


def build_pdf(staged: Path, out_dir: Path, common: list[str]) -> Path | None:
    """PDF via Typst, compiled in process.

    Pandoc's ``--pdf-engine`` wants an executable, and the ``typst`` wheel ships a native
    extension rather than one. So the document is converted to Typst markup and compiled
    through the Python binding instead — same engine, no system install, and no LaTeX
    distribution to depend on.
    """
    import pypandoc

    try:
        import typst
    except ImportError:
        print("  ! Typst not available; PDF not built.  pip install typst")
        return None

    source = out_dir / f".{STEM}.typ"
    try:
        pypandoc.convert_file(
            str(staged), "typst", outputfile=str(source),
            # Standalone so pandoc emits its own Typst template, which defines the
            # helpers its writer references; a bare fragment does not compile.
            extra_args=[*common, "--standalone", "-V", "papersize=a4",
                        "-V", "fontsize=10pt"],
        )
        pdf = out_dir / f"{STEM}.pdf"
        typst.compile(str(source), output=str(pdf), root=str(PAPER))
    finally:
        source.unlink(missing_ok=True)
    print(f"  wrote {pdf.name}")
    return pdf


def build(staged: Path, out_dir: Path) -> list[Path]:
    import pypandoc

    written: list[Path] = []
    # No title/author metadata: the manuscript's own heading block carries the title,
    # the affiliation, the ORCID and the corresponding address, and passing metadata as
    # well renders the title block twice. The heading block is the one to keep, because
    # it is the one the consistency tests check.
    common = [
        "--resource-path", str(PAPER),
        "--from", "markdown+implicit_figures+pipe_tables+tex_math_dollars",
    ]

    docx = out_dir / f"{STEM}.docx"
    pypandoc.convert_file(str(staged), "docx", outputfile=str(docx), extra_args=common)
    written.append(docx)
    print(f"  wrote {docx.name}")

    pdf = build_pdf(staged, out_dir, common)
    if pdf is not None:
        written.append(pdf)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-checks", action="store_true", help="build even if the numbers have drifted")
    ap.add_argument("--out", type=Path, default=PAPER)
    args = ap.parse_args()

    if not SOURCE.exists():
        raise SystemExit(f"{SOURCE} not found")
    if not args.skip_checks:
        check_consistency()

    staged = PAPER / ".manuscript_staged.md"
    try:
        prepare(SOURCE, staged)
        written = build(staged, args.out)
    finally:
        staged.unlink(missing_ok=True)

    for path in written:
        print(f"  {path.name}: {path.stat().st_size / 1024:.0f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
