"""Smoke tests: index integrity, citation validity, two representative answers."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap  # noqa: F401,E402

from src.config import COMPANIES, PROCESSED  # noqa: E402
from src.qa import answer_question, load_retriever  # noqa: E402


class TestFinRAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = load_retriever()
        cls.chunks = [json.loads(l) for l in (PROCESSED / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    def test_corpus_shape(self):
        self.assertGreaterEqual(len(self.chunks), 5000)
        self.assertEqual(len({c["code"] for c in self.chunks}), 12)
        self.assertGreater(sum(1 for c in self.chunks if c["content_type"] == "table"), 3000)
        for c in self.chunks[:200]:
            self.assertTrue(c["company"])
            self.assertGreaterEqual(c["page_start"], 1)
            self.assertTrue(c["section"])

    def test_metadata_is_traceable(self):
        by_id = {c["chunk_id"]: c for c in self.chunks}
        result = answer_question("五粮液2024年归母净利润和经营活动现金流净额分别是多少？", self.retriever, top_k=8)
        self.assertEqual(result["answer_status"], "ok")
        self.assertGreaterEqual(len(result["citations"]), 2)
        for cite in result["citations"]:
            self.assertIn(cite["chunk_id"], by_id)
            self.assertEqual(cite["company"], by_id[cite["chunk_id"]]["company"])
            self.assertEqual(cite["page"], by_id[cite["chunk_id"]]["page_start"])

    def test_panorama_ranking(self):
        result = answer_question("12家公司2024年营业总收入排名前三的是谁？", self.retriever, top_k=12)
        self.assertEqual(result["answer_status"], "ok")
        self.assertIn("贵州茅台", result["headline"])
        self.assertEqual(len(result["table"]["rows"]), 12)
        self.assertEqual([r[1] for r in result["table"]["rows"][:3]], ["贵州茅台", "五粮液", "山西汾酒"])
        self.assertIn("亿元", result["table"]["rows"][0][2])

    def test_companies_registry(self):
        self.assertEqual(len(COMPANIES), 12)
        codes = {c.code for c in COMPANIES}
        self.assertEqual(codes, {c["code"] for c in self.chunks})


if __name__ == "__main__":
    unittest.main(verbosity=2)
