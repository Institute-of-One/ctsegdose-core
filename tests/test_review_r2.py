"""The round-2 response documents, checked against the manuscript and the results.

A response letter is where numbers get retyped. The manuscript is guarded by
``test_manuscript_consistency``; the letter that tells an editor what the manuscript
says is guarded by nothing, and it is read more carefully than any single sentence of
the paper. These compare what the replies claim with what the results hold and with
what the manuscript actually contains.

Skips when the round-2 documents are absent, so a fresh clone stays green.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REVIEW = REPO / "docs" / "review_r2"
MANUSCRIPT = REPO / "paper" / "manuscript.md"

pytestmark = pytest.mark.skipif(
    not REVIEW.is_dir(), reason="round-2 review documents not in this checkout"
)


def _flat(path: Path) -> str:
    if not path.is_file():
        pytest.skip(f"{path.name} not in this checkout")
    return " ".join(path.read_text(encoding="utf-8").split())


@pytest.fixture(scope="module")
def reports():
    text = _flat(REVIEW / "reviewer_reports.md")
    return re.sub(r"> ?", "", text)


@pytest.fixture(scope="module")
def reply1():
    return _flat(REVIEW / "response_reviewer_1.md")


@pytest.fixture(scope="module")
def reply2():
    return _flat(REVIEW / "response_reviewer_2.md")


@pytest.fixture(scope="module")
def letter():
    return _flat(REVIEW / "cover_letter_revision.txt")


@pytest.fixture(scope="module")
def manuscript():
    return MANUSCRIPT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def sensitivity():
    path = REPO / "results" / "sensitivity_reconstructed_1.5mm.json"
    if not path.is_file():
        pytest.skip("run tools/sensitivity_reconstructed.py")
    return json.loads(path.read_text(encoding="utf-8"))


class TestTheReportsAreQuotedNotParaphrased:
    def test_reviewer_2s_comment_is_quoted_verbatim(self, reports, reply2):
        """SuSy shows the reviewer their own words beside the reply. A paraphrase reads
        as an author answering a question that was not asked."""
        quoted = re.search(r"\*Overall, the authors(.+?)\*", _flat(REVIEW / "response_reviewer_2.md"))
        assert quoted, "reviewer 2's comment is not quoted in the reply"
        probe = " ".join(("Overall, the authors" + quoted.group(1)).split())[:200]
        assert probe.rstrip(".*") in reports, (
            "reviewer 2's comment in the reply does not match the transcribed report"
        )

    def test_reviewer_1_is_not_sent_a_list_of_demands_they_did_not_make(self, reply1):
        """Reviewer 1 asked for nothing. The reply must say so rather than inventing
        responses to fill the template."""
        assert "No change is requested" in reply1
        assert "**Comments 1:**" not in reply1


class TestTheNumbersInTheRepliesAreTheComputedOnes:
    def test_the_sensitivity_figures_match_the_result_file(self, sensitivity, reply2, letter):
        e = sensitivity["exposure"]
        worst_weight = max(
            abs(r["relative_weight"]["delta"]) for r in sensitivity["by_organ"].values()
        )
        worst_index = max(
            abs(r["organ_weighted_ctdivol_mgy"]["delta"])
            for r in sensitivity["by_organ"].values()
        )
        for document, name in ((reply2, "reply to reviewer 2"), (letter, "cover letter")):
            assert f"{e['n_series_analysed']} series carrying an index" in document, name
            assert (
                f"{e['n_series_on_a_reconstructed_value']} rest on a reconstructed"
                in document
            ), name
            assert f"at most {worst_weight:.3f}" in document, name
            assert f"at most {worst_index:.2f} mGy" in document, name

    def test_the_ge_consequence_is_in_both_documents(self, sensitivity, reply2, letter):
        """The number a response letter is most tempted to leave out."""
        published = sensitivity["vendor_reach"]["published"]["GE"]
        assert sensitivity["vendor_reach"]["recorded_only"]["GE"] == 0
        for document, name in ((reply2, "reply to reviewer 2"), (letter, "cover letter")):
            assert f"from {published} series to none" in document, name

    def test_the_per_model_counts_match_the_cohort(self, reply2, letter):
        import csv

        path = REPO / "results" / "series_1.5mm.csv"
        if not path.is_file():
            pytest.skip("run tools/make_analysis.py")
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        counts = {
            v: len({r["model_name"] for r in rows if r["vendor"] == v})
            for v in ("GE", "Siemens", "Canon/Toshiba", "Philips")
        }
        phrase = (
            f"{counts['GE']} distinct GE models, {counts['Siemens']} Siemens, "
            f"{counts['Canon/Toshiba']} Canon/Toshiba and {counts['Philips']} Philips"
        )
        for document, name in ((reply2, "reply to reviewer 2"), (letter, "cover letter")):
            assert phrase in document, f"{name} does not carry {phrase!r}"

    def test_the_failed_mask_figures_agree_with_the_manuscript(self, reply2, manuscript):
        """The reply quotes a median pair the manuscript also quotes. They drifted once
        already, when the manuscript's pair was computed over a looser population."""
        pair = re.search(r"from (\d\.\d{3}) to (\d\.\d{3})", reply2)
        assert pair, "the reply does not give what removing the failed mask does"
        assert f"from {pair.group(1)} to\n{pair.group(2)}" in manuscript or (
            f"from {pair.group(1)} to {pair.group(2)}" in " ".join(manuscript.split())
        ), "the reply and the manuscript disagree on the median pair"


class TestTheDocumentsPointAtRealPlaces:
    def test_every_section_cited_exists(self, reply1, reply2, letter, manuscript):
        for document, name in (
            (reply1, "reply to reviewer 1"),
            (reply2, "reply to reviewer 2"),
            (letter, "cover letter"),
        ):
            for section in set(re.findall(r"Sections? (\d+\.\d+)", document)):
                assert f"### {section}." in manuscript, (
                    f"{name} cites Section {section}, which does not exist"
                )

    def test_every_file_named_in_the_replies_is_present(self, reply2):
        for path in set(re.findall(r"`([\w./-]+\.(?:py|json|md))`", reply2)):
            assert (REPO / path).is_file(), f"the reply names {path}, which is not here"

    def test_the_letter_is_a_revision_letter(self, letter):
        """The portal pre-fills the box with the original submission's cover letter."""
        assert "pleased to submit" not in letter
        assert "second revised version of manuscript tomography-4516935" in letter


class TestTheEditorsRequestIsAnswered:
    def test_all_three_documents_address_the_ai_usage_statement(
        self, reply1, reply2, letter, manuscript
    ):
        """It arrived outside the reviewer reports and is the item most easily marked as
        already done, because a statement of that name was already in the back matter."""
        methods = manuscript.split("## 2. Materials and Methods", 1)[1].split(
            "## 3. Results", 1
        )[0]
        assert "Generative Artificial Intelligence" in methods
        for document, name in (
            (reply1, "reply to reviewer 1"),
            (reply2, "reply to reviewer 2"),
            (letter, "cover letter"),
        ):
            assert "Methods" in document and "AI usage statement" in document or (
                "Generative Artificial Intelligence" in document
            ), f"{name} does not report the editor's request as addressed"
