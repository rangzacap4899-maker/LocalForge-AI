import tempfile
import unittest
from pathlib import Path

from localforge_algorithms import VectorDB, cosine_similarity


class VectorDBTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = VectorDB(Path(self.temp.name) / "vectordb.sqlite")

    def tearDown(self):
        self.temp.cleanup()

    def test_counts_start_empty(self):
        self.assertEqual(self.db.counts(), (0, 0))

    def test_add_and_clear_rag(self):
        self.db.add_rag_chunk("doc.txt", "เนื้อหา", [1.0, 0.0])
        self.assertEqual(self.db.counts(), (1, 0))
        self.db.clear_rag()
        self.assertEqual(self.db.counts(), (0, 0))

    def test_replace_rag_source_does_not_duplicate_reindex(self):
        self.db.replace_rag_source("doc.md", [("old", [1.0, 0.0])])
        self.db.replace_rag_source("doc.md", [("new", [0.0, 1.0])])
        self.assertEqual(self.db.counts(), (1, 0))
        self.assertEqual(self.db.search_rag([0.0, 1.0])[0]["content"], "new")

    def test_add_and_clear_cache(self):
        self.db.add_to_cache("คำถาม", [0.0, 1.0], "คำตอบ")
        self.assertEqual(self.db.counts(), (0, 1))
        self.db.clear_cache()
        self.assertEqual(self.db.counts(), (0, 0))

    def test_search_rag_returns_top_k(self):
        self.db.add_rag_chunk("a.txt", "หนึ่ง", [1.0, 0.0])
        self.db.add_rag_chunk("b.txt", "สอง", [0.9, 0.1])
        results = self.db.search_rag([1.0, 0.0], top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "a.txt")


class CosineSimilarityTest(unittest.TestCase):
    def test_matching_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_empty_inputs(self):
        self.assertEqual(cosine_similarity([], [1.0]), 0.0)
        self.assertEqual(cosine_similarity([1.0], []), 0.0)


if __name__ == "__main__":
    unittest.main()
