import math

class VectorIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.vectors = {}

    def add(self, id_str, vec):
        assert len(vec) == self.dimension, "Dimension mismatch"
        self.vectors[id_str] = vec

    def search(self, query_vec, top_k=5):
        scores = []
        for id_str, vec in self.vectors.items():
            dot = sum(a * b for a, b in zip(query_vec, vec))
            norm_q = math.sqrt(sum(a * a for a in query_vec))
            norm_v = math.sqrt(sum(b * b for b in vec))
            sim = dot / (norm_q * norm_v) if norm_q > 0 and norm_v > 0 else 0.0
            scores.append((id_str, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
