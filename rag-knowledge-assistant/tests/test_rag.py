import unittest
from app.rag import chunk_text, stable_id
class ChunkingTests(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(chunk_text("   "), [])
    def test_overlap_and_tail(self):
        words = [f"word{i}" for i in range(20)]
        chunks = chunk_text(" ".join(words), chunk_size=8, overlap=2)
        self.assertEqual(len(chunks), 4)
        self.assertIn("word6", chunks[0])
        self.assertIn("word6", chunks[1])
        self.assertIn("word19", chunks[-1])
    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_text("some words", chunk_size=5, overlap=5)
    def test_stable_ids(self):
        self.assertEqual(stable_id("a.pdf", 0, "hello"), stable_id("a.pdf", 0, "hello"))
        self.assertNotEqual(stable_id("a.pdf", 0, "hello"), stable_id("a.pdf", 1, "hello"))
if __name__ == "__main__":
    unittest.main()
