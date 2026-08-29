"""Write the proof replies into the Word file's own comment threads.

MDPI's production editor reads the answers in the document, beside each comment,
not in a separate file. A companion document was supplied on the first round and
the reply threads were empty, which is the same as not answering.

Each reply is a real threaded reply: a new ``w:comment`` anchored to the parent's
range, linked to it through ``commentsExtended.xml`` by ``w15:paraIdParent``, and
registered in ``commentsIds.xml`` and ``commentsExtensible.xml`` so Word treats it
as part of the thread rather than as a second comment on the same words.

Usage::

    python tools/insert_comment_replies.py \\
        --proof docs/proof/manuscript.v9.docx \\
        --replies docs/proof/comment_replies.md \\
        --out docs/proof/manuscript.v9.replied.docx

The reply text is read from the markdown so the two cannot drift: what is checked
in the repository is what lands in the document.
"""

from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path

AUTHOR = "Shuji Yamamoto"
INITIALS = "SY"
#: Reply timestamp. Passed in rather than taken from the clock so that rebuilding
#: the file twice produces the same bytes.
DEFAULT_DATE = "2026-08-29T03:00:00Z"


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def parse_replies(path: Path) -> dict[int, list[str]]:
    """Comment id -> reply paragraphs, read from the markdown.

    A reply is the block-quoted text under a ``**[id] ...**`` heading. Markdown
    emphasis is stripped: a Word comment is plain text, and leaving the asterisks
    in would put them in front of the editor.
    """
    replies: dict[int, list[str]] = {}
    current: int | None = None
    buffer: list[str] = []

    def flush():
        if current is None:
            return
        text = "\n".join(buffer).strip()
        paragraphs = [
            re.sub(r"\s+", " ", p).strip()
            for p in re.split(r"\n\s*\n", text)
            if p.strip()
        ]
        if paragraphs:
            replies[current] = paragraphs

    for line in path.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"\*\*\[(\d+)\]", line)
        if heading:
            flush()
            current, buffer = int(heading.group(1)), []
            continue
        if line.startswith(">"):
            body = line[1:].strip()
            body = re.sub(r"\*\*(.+?)\*\*", r"\1", body)
            body = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", body)
            if re.match(r"^[-•]\s+", body):
                # One list item, one paragraph. Consecutive bullet lines carry no blank
                # line between them, so without this they run together into a single
                # line -- four manufacturer locations on one line, which is what the
                # first build produced.
                buffer.append("")
                buffer.append("— " + re.sub(r"^[-•]\s+", "", body))
                buffer.append("")
                continue
            buffer.append(body)
        elif current is not None and not line.strip():
            buffer.append("")
    flush()
    return replies


def _paragraph(para_id: str, paragraphs: list[str]) -> str:
    """One w:p per paragraph, the first carrying the annotation reference."""
    out = []
    for i, text in enumerate(paragraphs):
        pid = para_id if i == 0 else f"{(int(para_id, 16) + i + 1) & 0x7FFFFFFF:08X}"
        ref = (
            '<w:r><w:rPr><w:rStyle w:val="af0"/></w:rPr><w:annotationRef/></w:r>'
            if i == 0
            else ""
        )
        out.append(
            f'<w:p w14:paraId="{pid}" w14:textId="{pid}" w:rsidR="00A12D87" '
            f'w:rsidRDefault="00A12D87"><w:pPr><w:pStyle w:val="af1"/></w:pPr>'
            f"{ref}"
            f'<w:r><w:t xml:space="preserve">{_escape(text)}</w:t></w:r></w:p>'
        )
    return "".join(out)


def build(proof: Path, replies_md: Path, out: Path, date: str) -> tuple[int, list[int]]:
    replies = parse_replies(replies_md)

    with zipfile.ZipFile(proof) as z:
        parts = {n: z.read(n) for n in z.namelist()}

    comments = parts["word/comments.xml"].decode("utf-8")
    document = parts["word/document.xml"].decode("utf-8")
    extended = parts["word/commentsExtended.xml"].decode("utf-8")
    ids = parts["word/commentsIds.xml"].decode("utf-8")
    extensible = parts["word/commentsExtensible.xml"].decode("utf-8")
    people = parts["word/people.xml"].decode("utf-8")

    # parent comment id -> its first paragraph's paraId, which is what a reply points at
    parent_para: dict[int, str] = {}
    for m in re.finditer(
        r'<w:comment w:id="(\d+)"[^>]*>\s*<w:p w14:paraId="([0-9A-Fa-f]{8})"', comments
    ):
        parent_para[int(m.group(1))] = m.group(2)

    next_id = max(parent_para) + 1
    new_comments, new_ex, new_ids, new_cex = [], [], [], []
    answered, missing = [], []

    for parent_id in sorted(parent_para):
        paragraphs = replies.get(parent_id)
        if not paragraphs:
            missing.append(parent_id)
            continue
        para_id = f"{(0x5A000000 + parent_id * 977) & 0x7FFFFFFF:08X}"
        durable = f"{(0x3B000000 + parent_id * 613) & 0x7FFFFFFF:08X}"

        new_comments.append(
            f'<w:comment w:id="{next_id}" w:author="{AUTHOR}" w:date="{date}" '
            f'w:initials="{INITIALS}">{_paragraph(para_id, paragraphs)}</w:comment>'
        )
        new_ex.append(
            f'<w15:commentEx w15:paraId="{para_id}" '
            f'w15:paraIdParent="{parent_para[parent_id]}" w15:done="0"/>'
        )
        new_ids.append(
            f'<w16cid:commentId w16cid:paraId="{para_id}" '
            f'w16cid:durableId="{durable}"/>'
        )
        new_cex.append(
            f'<w16cex:commentExtensible w16cex:durableId="{durable}" '
            f'w16cex:dateUtc="{date}"/>'
        )

        # Anchor the reply on the parent's range. The start goes beside the parent's
        # start; the end and the reference go immediately after the parent's reference
        # run, so the reply covers exactly the words the parent covers.
        start = f'<w:commentRangeStart w:id="{parent_id}"/>'
        if start not in document:
            raise SystemExit(f"comment {parent_id} has no range start in the document")
        document = document.replace(
            start, start + f'<w:commentRangeStart w:id="{next_id}"/>', 1
        )
        ref_run = re.search(
            r"<w:r\b[^>]*>(?:(?!</w:r>).)*?"
            rf'<w:commentReference w:id="{parent_id}"/>\s*</w:r>',
            document,
            re.S,
        )
        if not ref_run:
            raise SystemExit(f"comment {parent_id} has no reference run")
        document = (
            document[: ref_run.end()]
            + f'<w:commentRangeEnd w:id="{next_id}"/>'
            f'<w:r><w:rPr><w:rStyle w:val="af0"/></w:rPr>'
            f'<w:commentReference w:id="{next_id}"/></w:r>'
            + document[ref_run.end() :]
        )
        answered.append(parent_id)
        next_id += 1

    comments = comments.replace("</w:comments>", "".join(new_comments) + "</w:comments>")
    extended = extended.replace(
        "</w15:commentsEx>", "".join(new_ex) + "</w15:commentsEx>"
    )
    ids = ids.replace("</w16cid:commentsIds>", "".join(new_ids) + "</w16cid:commentsIds>")
    extensible = extensible.replace(
        "</w16cex:commentsExtensible>", "".join(new_cex) + "</w16cex:commentsExtensible>"
    )
    if f'w15:author="{AUTHOR}"' not in people:
        people = people.replace(
            "</w15:people>",
            f'<w15:person w15:author="{AUTHOR}"><w15:presenceInfo '
            f'w15:providerId="None" w15:userId="{AUTHOR}"/></w15:person></w15:people>',
        )

    parts["word/comments.xml"] = comments.encode("utf-8")
    parts["word/document.xml"] = document.encode("utf-8")
    parts["word/commentsExtended.xml"] = extended.encode("utf-8")
    parts["word/commentsIds.xml"] = ids.encode("utf-8")
    parts["word/commentsExtensible.xml"] = extensible.encode("utf-8")
    parts["word/people.xml"] = people.encode("utf-8")

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    return len(answered), missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--proof", type=Path, required=True)
    ap.add_argument("--replies", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--date", default=DEFAULT_DATE)
    args = ap.parse_args(argv)

    answered, missing = build(args.proof, args.replies, args.out, args.date)
    print(f"  replied to {answered} comments")
    if missing:
        print(f"  NO REPLY WRITTEN for comment ids: {missing}")
        return 1
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
