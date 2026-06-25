"""Tests for scripts/prefilter.py — recency gate and near-duplicate collapse."""
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from prefilter import (
    RECENCY_DAYS,
    _passes_recency,
    _title_tokens,
    _jaccard,
    collapse_near_dups,
)


# ── helpers ────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _item(source: str, published_at: str | None = None, title: str = "Test Title",
          recency_days_override: int | None = None) -> dict:
    return {
        "source": source,
        "title": title,
        "published_at": published_at,
        "fetched_at": _now().isoformat(),
        "recency_days_override": recency_days_override,
    }


# ── recency gate ───────────────────────────────────────────────────────────

class TestPassesRecency(unittest.TestCase):

    def test_fresh_arxiv_passes(self):
        now = _now()
        pub = (now - timedelta(days=1)).isoformat()
        self.assertTrue(_passes_recency(_item("arxiv:cs.AI", pub), now))

    def test_stale_arxiv_fails(self):
        now = _now()
        pub = (now - timedelta(days=RECENCY_DAYS["arxiv"] + 1)).isoformat()
        self.assertFalse(_passes_recency(_item("arxiv:cs.AI", pub), now))

    def test_arxiv_at_boundary_passes(self):
        now = _now()
        pub = (now - timedelta(days=RECENCY_DAYS["arxiv"])).isoformat()
        self.assertTrue(_passes_recency(_item("arxiv:cs.AI", pub), now))

    def test_rss_has_longer_window_than_arxiv(self):
        self.assertGreater(RECENCY_DAYS["rss"], RECENCY_DAYS["arxiv"])

    def test_fresh_rss_passes(self):
        now = _now()
        pub = (now - timedelta(days=5)).isoformat()
        self.assertTrue(_passes_recency(_item("rss:blog", pub), now))

    def test_stale_rss_fails(self):
        now = _now()
        pub = (now - timedelta(days=RECENCY_DAYS["rss"] + 1)).isoformat()
        self.assertFalse(_passes_recency(_item("rss:blog", pub), now))

    def test_hn_stale_fails(self):
        now = _now()
        pub = (now - timedelta(days=RECENCY_DAYS["hn"] + 1)).isoformat()
        self.assertFalse(_passes_recency(_item("hn:front", pub), now))

    def test_recency_days_override_respected(self):
        now = _now()
        # 20 days old; normally stale for arxiv (window=7) but override=30 → passes
        pub = (now - timedelta(days=20)).isoformat()
        item = _item("arxiv:cs.AI", pub, recency_days_override=30)
        self.assertTrue(_passes_recency(item, now))

    def test_recency_days_override_can_tighten(self):
        now = _now()
        # 5 days old; normally fresh for rss (window=30) but override=3 → fails
        pub = (now - timedelta(days=5)).isoformat()
        item = _item("rss:blog", pub, recency_days_override=3)
        self.assertFalse(_passes_recency(item, now))

    def test_missing_published_at_falls_back_to_fetched_at(self):
        # fetched_at is set to now in _item(), so item is always fresh
        now = _now()
        item = _item("arxiv:cs.AI", published_at=None)
        self.assertTrue(_passes_recency(item, now))

    def test_unparseable_published_at_passes(self):
        now = _now()
        item = _item("arxiv:cs.AI", published_at="not-a-date")
        self.assertTrue(_passes_recency(item, now))

    def test_z_suffix_isoformat_parsed_correctly(self):
        now = _now()
        pub = (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.assertTrue(_passes_recency(_item("arxiv:cs.AI", pub), now))

    def test_naive_datetime_treated_as_utc(self):
        now = _now()
        pub = (now - timedelta(days=1)).replace(tzinfo=None).isoformat()
        self.assertTrue(_passes_recency(_item("arxiv:cs.AI", pub), now))


# ── title-tokenisation + Jaccard ──────────────────────────────────────────

class TestTitleTokensAndJaccard(unittest.TestCase):

    def test_title_tokens_lowercases_and_splits(self):
        tokens = _title_tokens("LLM Agents Are Useful")
        self.assertIn("llm", tokens)
        self.assertIn("agents", tokens)

    def test_title_tokens_strips_punctuation(self):
        tokens = _title_tokens("GPT-4: A Review!")
        self.assertNotIn("gpt-4:", tokens)

    def test_jaccard_identical(self):
        toks = _title_tokens("agent memory planning")
        self.assertAlmostEqual(_jaccard(toks, toks), 1.0)

    def test_jaccard_disjoint(self):
        a = _title_tokens("cats and dogs")
        b = _title_tokens("machine learning")
        self.assertAlmostEqual(_jaccard(a, b), 0.0)

    def test_jaccard_partial_overlap(self):
        a = {"a", "b", "c"}
        b = {"b", "c", "d"}
        # |intersection|=2, |union|=4 → 0.5
        self.assertAlmostEqual(_jaccard(a, b), 0.5)

    def test_jaccard_empty_sets(self):
        self.assertEqual(_jaccard(set(), {"a", "b"}), 0.0)
        self.assertEqual(_jaccard({"a"}, set()), 0.0)


# ── collapse_near_dups ────────────────────────────────────────────────────

class TestCollapseNearDups(unittest.TestCase):

    def _make(self, source: str, title: str) -> dict:
        return {"source": source, "title": title}

    def test_identical_titles_collapsed(self):
        items = [
            self._make("arxiv:cs.AI", "LLM Agents: A Survey"),
            self._make("rss:blog",   "LLM Agents: A Survey"),
        ]
        result = collapse_near_dups(items)
        self.assertEqual(len(result), 1)

    def test_distinct_titles_both_kept(self):
        items = [
            self._make("arxiv:cs.AI", "Multi-Agent Planning with LLMs"),
            self._make("rss:blog",   "Retrieval-Augmented Generation Overview"),
        ]
        result = collapse_near_dups(items)
        self.assertEqual(len(result), 2)

    def test_higher_priority_source_wins(self):
        # arxiv has higher SOURCE_PRIORITY than rss
        items = [
            self._make("rss:blog",   "LLM Agents: A Survey"),
            self._make("arxiv:cs.AI", "LLM Agents: A Survey"),
        ]
        result = collapse_near_dups(items)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source"], "arxiv:cs.AI")

    def test_near_duplicate_above_threshold_collapsed(self):
        # High overlap — should collapse under default threshold 0.85
        items = [
            self._make("arxiv:cs.AI", "LLM Agent Planning and Tool Use"),
            self._make("rss:blog",   "LLM Agent Planning and Tool Use in Practice"),
        ]
        result = collapse_near_dups(items, threshold=0.6)
        self.assertEqual(len(result), 1)

    def test_threshold_boundary_below_keeps_both(self):
        # Low threshold means almost anything is "a dup"; very strict 1.0 means
        # only exact token-set matches are collapsed.
        items = [
            self._make("arxiv:cs.AI", "Agent Planning"),
            self._make("rss:blog",   "LLM Agent Planning and Evaluation"),
        ]
        result = collapse_near_dups(items, threshold=1.0)
        self.assertEqual(len(result), 2)

    def test_empty_list(self):
        self.assertEqual(collapse_near_dups([]), [])

    def test_single_item_returned_unchanged(self):
        items = [self._make("arxiv:cs.AI", "Only One Paper")]
        self.assertEqual(collapse_near_dups(items), items)


if __name__ == "__main__":
    unittest.main()
