import unittest
from vector_index import VectorIndex
from hnsw import HNSWGraph

class TestVectorIndex(unittest.TestCase):
    def test_flat_cosine_search(self):
        idx = VectorIndex(dimension=3, metric="cosine")
        idx.add("doc1", [1.0, 0.0, 0.0])
        idx.add("doc2", [0.0, 1.0, 0.0])
        idx.add("doc3", [0.9, 0.1, 0.0])

        res = idx.search([1.0, 0.0, 0.0], top_k=2)
        self.assertEqual(res[0][0], "doc1")
        self.assertEqual(res[1][0], "doc3")
        self.assertAlmostEqual(res[0][1], 1.0, places=5)

    def test_ivf_recall(self):
        idx = VectorIndex(dimension=4, metric="cosine")
        for i in range(50):
            idx.add(f"v_{i}", [float(i%3), float(i%5), float(i%7), 1.0])
        idx.train_ivf(nlist=4)
        q = [2.0, 4.0, 6.0, 1.0]
        flat_res = idx.search(q, top_k=3)
        ivf_res = idx.search_ivf(q, top_k=3, nprobe=4)
        self.assertEqual(flat_res[0][0], ivf_res[0][0])

    def test_hnsw_search(self):
        graph = HNSWGraph(dimension=3, m=8)
        graph.add("n1", [1.0, 0.0, 0.0])
        graph.add("n2", [0.0, 1.0, 0.0])
        graph.add("n3", [0.8, 0.2, 0.0])
        res = graph.search([1.0, 0.0, 0.0], k=2)
        self.assertTrue(any(r[0] == "n1" for r in res))

if __name__ == '__main__':
    unittest.main()
