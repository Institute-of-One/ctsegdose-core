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
import json
import re
import shutil
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

#: Fields that can only be filled once a release exists. They are placeholders in the
#: source rather than prose, so that a submission build can *refuse* on finding them
#: instead of a reviewer discovering a TODO in the PDF.
PLACEHOLDER = re.compile(r"\{\{([A-Z_]+)\}\}")

#: What each placeholder is replaced with in the internal build, so a reader of the
#: internal PDF can see at a glance what is outstanding.
INTERNAL_MARK = "[[ pending: {name} ]]"

RELEASE_FILE = PAPER / "release_metadata.json"


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


def release_metadata() -> dict[str, str]:
    """Release fields, if the author has recorded them. Never invented."""
    if not RELEASE_FILE.exists():
        return {}
    # utf-8-sig, not utf-8: this file is hand-authored, and a Windows editor or a
    # PowerShell redirect writes a byte-order mark that plain utf-8 decoding chokes on.
    data = json.loads(RELEASE_FILE.read_text(encoding="utf-8-sig"))
    return {k: str(v).strip() for k, v in data.items() if str(v).strip()}


def prepare(source: Path, staged: Path, *, submission: bool) -> tuple[Path, list[str]]:
    """Stage the manuscript for building, resolving the release placeholders.

    Two builds come out of one source. The internal one marks what is still outstanding
    so it is visible while reviewing; the submission one refuses to produce a file at all
    while anything is outstanding, because the failure mode being designed against is a
    reviewer opening a PDF and finding a placeholder in it.
    """
    text = HTML_COMMENT.sub("", source.read_text(encoding="utf-8"), count=1)
    # Thematic breaks separate sections while drafting; in a typeset document the
    # headings already do that, and Typst has no use for them.
    text = re.sub(r"^-{3,}\s*$", "", text, flags=re.M)

    known = release_metadata()
    outstanding: list[str] = []

    def resolve(match: re.Match[str]) -> str:
        name = match.group(1)
        value = known.get(name)
        if value:
            return value
        outstanding.append(name)
        return INTERNAL_MARK.format(name=name)

    text = PLACEHOLDER.sub(resolve, text)

    if submission and outstanding:
        raise SystemExit(
            "refusing to build a submission PDF with unresolved release fields: "
            + ", ".join(sorted(set(outstanding)))
            + f".\nRecord them in {RELEASE_FILE.relative_to(REPO)} once the release "
            "exists, then re-run. Use --internal to build a working copy meanwhile."
        )
    staged.write_text(text, encoding="utf-8")
    return staged, sorted(set(outstanding))


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

    # The intermediate always lives in PAPER, whatever --out is: Typst resolves the
    # figure paths against its project root and refuses an input outside it.
    source = PAPER / f".{STEM}.typ"
    try:
        pypandoc.convert_file(
            str(staged), "typst", outputfile=str(source),
            # Standalone so pandoc emits its own Typst template, which defines the
            # helpers its writer references; a bare fragment does not compile.
            extra_args=[*common, "--standalone", "-V", "papersize=a4",
                        "-V", "fontsize=10pt"],
        )
        pdf = out_dir / f"{STEM}.pdf"
        # Compile beside the target and move into place. Writing directly fails with a
        # bare OSError when the previous PDF is open in a viewer -- a routine thing to
        # have happen while reviewing a draft -- and the message says nothing useful.
        staged_pdf = PAPER / f".{STEM}.pdf"
        typst.compile(str(source), output=str(staged_pdf), root=str(PAPER))
        try:
            # shutil.move, not Path.replace: --out may name another drive, and rename
            # cannot cross one.
            shutil.move(str(staged_pdf), str(pdf))
        except OSError as exc:
            staged_pdf.unlink(missing_ok=True)
            raise SystemExit(
                f"built the PDF but could not write it to {pdf.name}: it is open in "
                f"another program. Close it and re-run.\n  ({exc})"
            ) from exc
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
    ap.add_argument("--internal", action="store_true",
                    help="working copy: mark unresolved release fields instead of refusing")
    ap.add_argument("--out", type=Path, default=PAPER)
    args = ap.parse_args()

    global STEM
    if args.internal:
        STEM = f"{STEM}_internal"

    if not SOURCE.exists():
        raise SystemExit(f"{SOURCE} not found")
    if not args.skip_checks:
        check_consistency()

    staged = PAPER / ".manuscript_staged.md"
    try:
        _, outstanding = prepare(SOURCE, staged, submission=not args.internal)
        written = build(staged, args.out)
    finally:
        staged.unlink(missing_ok=True)

    for path in written:
        print(f"  {path.name}: {path.stat().st_size / 1024:.0f} kB")
    if outstanding:
        print("\n  outstanding release fields (see paper/PRE_SUBMISSION_CHECKLIST.md):")
        for name in outstanding:
            print(f"    - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
