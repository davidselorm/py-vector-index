import math
import random
import json
from collections import defaultdict

class VectorIndex:
    """Multi-metric vector search engine supporting Flat and Inverted File (IVF) search."""
    def __init__(self, dimension, metric="cosine"):
        self.dimension = dimension
        self.metric = metric.lower()
        self.vectors = {}
        self.metadata = {}
        # IVF parameters
        self.nlist = 8
        self.centroids = []
        self.inverted_lists = defaultdict(list)
        self.is_trained = False

    def _distance(self, v1, v2):
        if self.metric == "cosine":
            dot = sum(a * b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a * a for a in v1))
            norm2 = math.sqrt(sum(b * b for b in v2))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot / (norm1 * norm2)
        elif self.metric == "euclidean":
            return -math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
        elif self.metric == "dot":
            return sum(a * b for a, b in zip(v1, v2))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def add(self, id_str, vec, meta=None):
        assert len(vec) == self.dimension, f"Expected dim {self.dimension}, got {len(vec)}"
        self.vectors[id_str] = [float(x) for x in vec]
        if meta:
            self.metadata[id_str] = meta

    def train_ivf(self, nlist=8, max_iters=10):
        """K-means centroid clustering for inverted file partitioning."""
        self.nlist = min(nlist, len(self.vectors))
        vec_list = list(self.vectors.values())
        if len(vec_list) < self.nlist:
            return

        # Random centroid initialization
        self.centroids = [list(c) for c in random.sample(vec_list, self.nlist)]

        for _ in range(max_iters):
            clusters = defaultdict(list)
            for vec in vec_list:
                best_c = max(range(self.nlist), key=lambda i: self._distance(vec, self.centroids[i]))
                clusters[best_c].append(vec)

            # Update centroids
            for c_idx in range(self.nlist):
                if clusters[c_idx]:
                    new_c = [0.0] * self.dimension
                    for v in clusters[c_idx]:
                        for d in range(self.dimension):
                            new_c[d] += v[d]
                    self.centroids[c_idx] = [x / len(clusters[c_idx]) for x in new_c]

        # Populate inverted lists
        self.inverted_lists.clear()
        for id_str, vec in self.vectors.items():
            best_c = max(range(self.nlist), key=lambda i: self._distance(vec, self.centroids[i]))
            self.inverted_lists[best_c].append(id_str)
        self.is_trained = True

    def search(self, query_vec, top_k=5, filter_fn=None):
        """Exact flat search across all indexed vectors."""
        scores = []
        for id_str, vec in self.vectors.items():
            if filter_fn and not filter_fn(self.metadata.get(id_str, {})):
                continue
            sim = self._distance(query_vec, vec)
            scores.append((id_str, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def search_ivf(self, query_vec, top_k=5, nprobe=2):
        """Approximate nearest neighbor search querying closest IVF clusters."""
        if not self.is_trained:
            return self.search(query_vec, top_k)

        # Find closest centroids
        c_scores = [(i, self._distance(query_vec, c)) for i, c in enumerate(self.centroids)]
        c_scores.sort(key=lambda x: x[1], reverse=True)
        probed_clusters = [c[0] for c in c_scores[:nprobe]]

        candidate_ids = set()
        for c_idx in probed_clusters:
            candidate_ids.update(self.inverted_lists[c_idx])

        scores = []
        for id_str in candidate_ids:
            scores.append((id_str, self._distance(query_vec, self.vectors[id_str])))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def save_index(self, filepath):
        data = {
            "dimension": self.dimension,
            "metric": self.metric,
            "vectors": self.vectors,
            "metadata": self.metadata
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    @classmethod
    def load_index(cls, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        idx = cls(data["dimension"], metric=data["metric"])
        idx.vectors = data["vectors"]
        idx.metadata = data["metadata"]
        return idx
