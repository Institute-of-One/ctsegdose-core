"""Build the revised manuscript with its changes marked, for the editor and reviewers.

Usage::

    python tools/build_highlighted_revision.py --against submitted-tomography-r1

MDPI asks that revisions be highlighted "so editors and reviewers can see any changes
made". The comparison is against a git tag rather than against a file someone
remembered to keep, because what was submitted is a fact about the repository and not
a fact about anyone's desktop.

Marking is at paragraph granularity, not word. A word-level diff of a manuscript that
gained 2 400 words produces a document more marked than not, in which a reviewer
cannot see which passages are new; a paragraph either is the one that was submitted or
it is not, and that is the distinction a reviewer is actually looking for. Paragraphs
that are new or rewritten are highlighted; paragraphs carried over unchanged are not.

Writes ``paper/manuscript_tomography_revised_highlighted.docx``.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "paper" / "manuscript.md"
DEFAULT_TAG = "submitted-tomography-r1"
STEM = "manuscript_tomography_revised_highlighted"

#: Structural lines are never highlighted: a heading whose section number moved is not
#: a revision a reviewer needs to read, and marking it buries the ones that are.
_STRUCTURAL = re.compile(r"^\s*(#{1,6}\s|!\[\]\(|<!--|\|)")


def _submitted(tag: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{tag}:paper/manuscript.md"],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise SystemExit(
            f"cannot read the submitted manuscript at {tag}: {result.stderr.strip()}"
        )
    return result.stdout


def _normalise(paragraph: str) -> str:
    """Compare on flattened whitespace, so a reflow is not mistaken for a revision."""
    return " ".join(paragraph.split())


def mark(current: str, submitted: str) -> tuple[str, int, int]:
    was = {_normalise(p) for p in submitted.split("\n\n") if p.strip()}
    out: list[str] = []
    changed = total = 0

    for paragraph in current.split("\n\n"):
        if not paragraph.strip():
            out.append(paragraph)
            continue
        if _STRUCTURAL.match(paragraph):
            out.append(paragraph)
            continue
        total += 1
        if _normalise(paragraph) in was:
            out.append(paragraph)
            continue
        changed += 1
        # Pandoc renders a bracketed span with this class as highlighted text in Word.
        out.append(f"[{paragraph}]{{.mark}}")

    return "\n\n".join(out), changed, total


def build(tag: str, out_dir: Path) -> Path:
    import pypandoc

    marked, changed, total = mark(SOURCE.read_text(encoding="utf-8"), _submitted(tag))
    print(f"  {changed} of {total} prose paragraphs are new or rewritten")

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "marked.md"
        staged.write_text(marked, encoding="utf-8")
        # The figures are referenced relatively, so they have to sit beside the source.
        shutil.copytree(SOURCE.parent / "figures", Path(tmp) / "figures")

        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / f"{STEM}.docx"
        pypandoc.convert_file(
            str(staged),
            "docx",
            outputfile=str(output),
            extra_args=["--from", "markdown+bracketed_spans", f"--resource-path={tmp}"],
        )

    print(f"  wrote {output.name} ({output.stat().st_size // 1024} kB)")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--against", default=DEFAULT_TAG)
    parser.add_argument("--out", type=Path, default=REPO / "paper")
    args = parser.parse_args(argv)
    build(args.against, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
