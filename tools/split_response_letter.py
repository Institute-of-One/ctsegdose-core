"""Split the response letter into one file per reviewer, in SuSy's own format.

Usage::

    python tools/split_response_letter.py

SuSy gives each reviewer a separate "Author's Notes to Reviewer" box and supplies a
template — *Comments 1:* then *Response 1:* — so the reply has to be pasted per
reviewer rather than as one document. Writing two files by hand would make three
copies of the same argument, and this programme has never had three copies of anything
stay in agreement.

So the per-reviewer files are generated. The comments come from the transcribed
reports and the responses from the combined letter, both under version control, and
the two are matched by number. A comment with no response, or a response with no
comment, stops the build rather than producing a file with a gap in it.

Writes ``docs/review_r1/response_reviewer_1.md`` and ``_2.md``, and the same as
``.docx`` if pypandoc is available.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REVIEW_DIR = REPO / "docs" / "review_r1"
REPORTS = REVIEW_DIR / "reviewer_reports.md"
LETTER = REVIEW_DIR / "response_to_reviewers.md"

HEADER = """# Response to Reviewer {n} — manuscript tomography-4516935

Thank you for a careful and constructive report. Each comment is answered below in
order, with the location of the change in the revised manuscript. Revised passages are
highlighted in the marked file uploaded alongside this response.

"""

FOOTER = """
---

Shuji Yamamoto
Institute of One, LISIT Co., Ltd., Tokyo, Japan
"""


def _reviewer_section(text: str, n: int) -> str:
    """The block of `text` belonging to reviewer `n`, up to the next top-level heading."""
    marker = f"## Reviewer {n}"
    if marker not in text:
        raise SystemExit(f"{marker} not found")
    after = text.split(marker, 1)[1]
    return re.split(r"\n## (?!#)", after, maxsplit=1)[0]


def responses(n: int) -> list[tuple[str, str]]:
    """(heading, body) for each answered comment, in the order the letter gives them."""
    section = _reviewer_section(LETTER.read_text(encoding="utf-8"), n)
    parts = re.split(r"\n### ", section)[1:]
    out = []
    for part in parts:
        heading, _, body = part.partition("\n")
        out.append((heading.strip(), body.strip()))
    return out


def comments(n: int) -> list[str]:
    """Reviewer 1 numbers its remarks; reviewer 2 groups them under section headings.

    Both are returned as a flat list in reading order, because the reply answers them
    in that order and the two lists are matched by position.
    """
    section = _reviewer_section(REPORTS.read_text(encoding="utf-8"), n)
    quoted = [
        line[2:].rstrip() if line.startswith("> ") else ""
        for line in section.splitlines()
        if line.startswith(">")
    ]
    text = "\n".join(quoted)

    if n == 1:
        found = re.split(r"\n(?=\d\) )", text)
        return [" ".join(p.split()) for p in found if re.match(r"^\d\) ", p.strip())]

    # Reviewer 2 groups remarks under section labels in bold. Most are bullets, but the
    # remark on the abstract is a bare paragraph, and taking only the bullets dropped it
    # silently -- which the count check caught. Anything quoted that is neither blank nor
    # a section label is a comment.
    out: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block or re.fullmatch(r"\*\*[^*]+\*\*", block):
            continue
        for item in re.split(r"\n(?=- )", block):
            item = " ".join(item.split())
            if not item:
                continue
            out.append(item[2:] if item.startswith("- ") else item)
    return out


def render(n: int) -> str:
    answers = responses(n)
    asked = comments(n)
    if len(asked) != len(answers):
        raise SystemExit(
            f"reviewer {n}: {len(asked)} comments transcribed but {len(answers)} "
            f"answered. Every comment must have exactly one response."
        )

    lines = [HEADER.format(n=n)]
    for index, (comment, (heading, body)) in enumerate(zip(asked, answers, strict=True), 1):
        lines.append(f"**Comments {index}:** {comment}\n")
        lines.append(f"**Response {index}:** *{heading}*\n")
        lines.append(body + "\n")
    lines.append(FOOTER)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-docx", action="store_true")
    args = parser.parse_args(argv)

    for n in (1, 2):
        text = render(n)
        out = REVIEW_DIR / f"response_reviewer_{n}.md"
        out.write_text(text, encoding="utf-8")
        answered = text.count("**Response ")
        print(f"  reviewer {n}: {answered} comments answered -> {out.name}")

        if args.no_docx:
            continue
        try:
            import pypandoc
        except ImportError:  # pragma: no cover - convenience only
            print("    (pypandoc unavailable; markdown only)", file=sys.stderr)
            continue
        docx = out.with_suffix(".docx")
        pypandoc.convert_file(str(out), "docx", outputfile=str(docx))
        print(f"    wrote {docx.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
