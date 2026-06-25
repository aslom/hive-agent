"""Tests for scripts/rank.py — cap/status logic and prompt construction."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rank import (
    SECTION_RULES,
    assign_statuses,
    build_prompt,
    effective_cap,
)


# ── helpers ────────────────────────────────────────────────────────────────

def _entry(id_: str, score: int, section: str = "papers") -> dict:
    return {"id": id_, "score": score, "tags": [], "why": "test", "section": section}


def _scored(section: str, entries: list[dict]) -> dict[str, list[dict]]:
    return {section: entries}


# ── effective_cap ─────────────────────────────────────────────────────────

class TestEffectiveCap(unittest.TestCase):

    def test_news_static_cap(self):
        items = [_entry(f"n{i}", 8, "news") for i in range(10)]
        self.assertEqual(effective_cap("news", items), SECTION_RULES["news"]["cap"])

    def test_blogs_static_cap(self):
        items = [_entry(f"b{i}", 8, "blogs") for i in range(10)]
        self.assertEqual(effective_cap("blogs", items), SECTION_RULES["blogs"]["cap"])

    def test_papers_normal_cap(self):
        # Fewer than burst_trigger_count items at score >= burst_trigger_score
        items = [_entry(f"p{i}", 8) for i in range(5)]
        self.assertEqual(effective_cap("papers", items), SECTION_RULES["papers"]["cap"])

    def test_papers_burst_cap_engaged(self):
        rules = SECTION_RULES["papers"]
        # Fill with items at exactly the burst trigger score to hit the threshold
        count = rules["burst_trigger_count"]
        items = [_entry(f"p{i}", rules["burst_trigger_score"]) for i in range(count)]
        self.assertEqual(effective_cap("papers", items), rules["burst_cap"])

    def test_papers_burst_cap_not_engaged_one_short(self):
        rules = SECTION_RULES["papers"]
        count = rules["burst_trigger_count"] - 1
        items = [_entry(f"p{i}", rules["burst_trigger_score"]) for i in range(count)]
        self.assertEqual(effective_cap("papers", items), rules["cap"])


# ── assign_statuses ───────────────────────────────────────────────────────

class TestAssignStatusesPapers(unittest.TestCase):

    def test_top_scorers_get_featured(self):
        rules = SECTION_RULES["papers"]
        items = [_entry(f"p{i}", rules["featured_min"]) for i in range(rules["cap"])]
        decisions = assign_statuses(_scored("papers", items))
        statuses = [decisions[f"p{i}"]["status"] for i in range(rules["cap"])]
        self.assertTrue(all(s == "featured" for s in statuses))

    def test_cap_enforced(self):
        rules = SECTION_RULES["papers"]
        # One more than the cap, all at featured_min
        items = [_entry(f"p{i}", rules["featured_min"]) for i in range(rules["cap"] + 3)]
        decisions = assign_statuses(_scored("papers", items))
        featured = [k for k, v in decisions.items() if v["status"] == "featured"]
        self.assertEqual(len(featured), rules["cap"])

    def test_paper_above_min_beyond_cap_stays_candidate(self):
        rules = SECTION_RULES["papers"]
        # cap+1 items all at featured_min → the last one should stay 'candidate'
        items = [_entry(f"p{i}", rules["featured_min"]) for i in range(rules["cap"] + 1)]
        decisions = assign_statuses(_scored("papers", items))
        candidate = [k for k, v in decisions.items() if v["status"] == "candidate"]
        self.assertEqual(len(candidate), 1)

    def test_mid_band_paper_gets_appendix(self):
        rules = SECTION_RULES["papers"]
        score = rules["appendix_min"]  # below featured_min, at appendix_min
        items = [_entry("mid", score)]
        decisions = assign_statuses(_scored("papers", items))
        self.assertEqual(decisions["mid"]["status"], "appendix")

    def test_below_appendix_min_dropped(self):
        rules = SECTION_RULES["papers"]
        score = rules["appendix_min"] - 1
        items = [_entry("low", score)]
        decisions = assign_statuses(_scored("papers", items))
        self.assertEqual(decisions["low"]["status"], "dropped")

    def test_score_zero_dropped(self):
        items = [_entry("zero", 0)]
        decisions = assign_statuses(_scored("papers", items))
        self.assertEqual(decisions["zero"]["status"], "dropped")


class TestAssignStatusesNewsBlogs(unittest.TestCase):

    def _run(self, section: str, score: int, count: int = 1) -> list[str]:
        items = [_entry(f"{section}{i}", score, section) for i in range(count)]
        decisions = assign_statuses({section: items})
        return [decisions[f"{section}{i}"]["status"] for i in range(count)]

    def test_news_featured(self):
        score = SECTION_RULES["news"]["featured_min"]
        statuses = self._run("news", score, 1)
        self.assertEqual(statuses[0], "featured")

    def test_news_cap_enforced(self):
        cap = SECTION_RULES["news"]["cap"]
        statuses = self._run("news", SECTION_RULES["news"]["featured_min"], cap + 2)
        self.assertEqual(statuses.count("featured"), cap)

    def test_news_beyond_cap_gets_appendix_not_candidate(self):
        # News/blogs don't have a multi-day pool; overflow goes to appendix.
        rules = SECTION_RULES["news"]
        items = [_entry(f"n{i}", rules["featured_min"], "news") for i in range(rules["cap"] + 1)]
        decisions = assign_statuses({"news": items})
        overflow = [v["status"] for k, v in decisions.items() if v["status"] not in ("featured",)]
        self.assertTrue(all(s == "appendix" for s in overflow))

    def test_blogs_appendix_band(self):
        rules = SECTION_RULES["blogs"]
        score = rules["appendix_min"]
        statuses = self._run("blogs", score, 1)
        self.assertEqual(statuses[0], "appendix")

    def test_blogs_below_appendix_min_dropped(self):
        score = SECTION_RULES["blogs"]["appendix_min"] - 1
        statuses = self._run("blogs", score, 1)
        self.assertEqual(statuses[0], "dropped")


# ── build_prompt ──────────────────────────────────────────────────────────

class TestBuildPrompt(unittest.TestCase):

    def setUp(self):
        self.items = [
            {"id": "abc", "title": "An LLM Paper", "score": 9},
            {"id": "def", "title": "Another Paper", "score": 7},
        ]
        self.rubric = "## Rubric\nScore 0-10."

    def test_contains_section_header(self):
        prompt = build_prompt("papers", self.items, self.rubric)
        self.assertIn("# Section to rank: papers", prompt)

    def test_contains_item_count(self):
        prompt = build_prompt("papers", self.items, self.rubric)
        self.assertIn(f"Number of items: {len(self.items)}", prompt)

    def test_items_inlined_as_json(self):
        prompt = build_prompt("papers", self.items, self.rubric)
        # The items JSON should be parseable and round-trip correctly.
        start = prompt.index("```json\n") + len("```json\n")
        end = prompt.index("\n```", start)
        parsed = json.loads(prompt[start:end])
        self.assertEqual(len(parsed), len(self.items))
        self.assertEqual(parsed[0]["id"], "abc")

    def test_rubric_appended(self):
        prompt = build_prompt("papers", self.items, self.rubric)
        self.assertIn(self.rubric, prompt)

    def test_different_sections_produce_different_prompts(self):
        p1 = build_prompt("papers", self.items, self.rubric)
        p2 = build_prompt("news",   self.items, self.rubric)
        self.assertNotEqual(p1, p2)

    def test_empty_items_list(self):
        prompt = build_prompt("blogs", [], self.rubric)
        self.assertIn("Number of items: 0", prompt)


if __name__ == "__main__":
    unittest.main()
