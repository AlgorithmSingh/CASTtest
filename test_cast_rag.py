import unittest

from cast_rag import CastChunker, FixedChunker, build_synthetic_code_corpus, evaluate_retrieval, non_ws_len


class CastRagTests(unittest.TestCase):
    def setUp(self):
        self.files, self.cases = build_synthetic_code_corpus()

    def test_cast_chunks_respect_non_ws_budget(self):
        chunker = CastChunker(max_chunk_size=220)
        chunks = chunker.chunk(self.files)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(non_ws_len(ch.text) <= 220 for ch in chunks))

    def test_cast_recall_not_worse_than_fixed(self):
        fixed_chunks = FixedChunker(lines_per_chunk=8).chunk(self.files)
        cast_chunks = CastChunker(max_chunk_size=220).chunk(self.files)

        fixed = evaluate_retrieval(fixed_chunks, self.cases, k=3)
        cast = evaluate_retrieval(cast_chunks, self.cases, k=3)

        self.assertGreaterEqual(cast["recall_at_k"], fixed["recall_at_k"])
        self.assertGreaterEqual(cast["recall_at_k"], 2 / 3)

    def test_cache_query_hits_cache_repo_with_cast(self):
        cast_chunks = CastChunker(max_chunk_size=220).chunk(self.files)
        cast = evaluate_retrieval(cast_chunks, [self.cases[1]], k=3)
        self.assertEqual(1, cast["hits"])


if __name__ == "__main__":
    unittest.main()
