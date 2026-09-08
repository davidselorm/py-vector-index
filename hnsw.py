import math
import random
import heapq

class HNSWGraph:
    """Hierarchical Navigable Small World (HNSW) proximity graph for ANN search."""
    def __init__(self, dimension, m=16, ef_construction=64, ml=0.36):
        self.dimension = dimension
        self.m = m
        self.ef_construction = ef_construction
        self.ml = ml
        self.entry_point = None
        self.max_level = -1
        self.nodes = {}       # id -> vector
        self.layers = {}      # level -> {id -> set(neighbors)}

    def _cosine_dist(self, v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a * a for a in v1))
        n2 = math.sqrt(sum(b * b for b in v2))
        if n1 == 0 or n2 == 0:
            return 1.0
        return 1.0 - (dot / (n1 * n2))

    def _random_level(self):
        lvl = 0
        while random.random() < self.ml and lvl < 16:
            lvl += 1
        return lvl

    def add(self, node_id, vector):
        self.nodes[node_id] = vector
        node_level = self._random_level()

        for l in range(node_level + 1):
            if l not in self.layers:
                self.layers[l] = {}
            self.layers[l][node_id] = set()

        if self.entry_point is None:
            self.entry_point = node_id
            self.max_level = node_level
            return

        curr_obj = self.entry_point
        # Greedy routing through upper layers
        for l in range(self.max_level, node_level, -1):
            changed = True
            while changed:
                changed = False
                curr_dist = self._cosine_dist(vector, self.nodes[curr_obj])
                for neighbor in self.layers[l].get(curr_obj, []):
                    d = self._cosine_dist(vector, self.nodes[neighbor])
                    if d < curr_dist:
                        curr_dist = d
                        curr_obj = neighbor
                        changed = True

        # Connect at lower layers
        for l in range(min(node_level, self.max_level), -1, -1):
            neighbors = list(self.layers[l].get(curr_obj, []))
            neighbors.sort(key=lambda n: self._cosine_dist(vector, self.nodes[n]))
            selected = neighbors[:self.m]
            for n in selected:
                self.layers[l][node_id].add(n)
                self.layers[l][n].add(node_id)

        if node_level > self.max_level:
            self.max_level = node_level
            self.entry_point = node_id

    def search(self, query_vec, k=5):
        if self.entry_point is None:
            return []
        curr = self.entry_point
        curr_dist = self._cosine_dist(query_vec, self.nodes[curr])

        for l in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                for n in self.layers[l].get(curr, []):
                    d = self._cosine_dist(query_vec, self.nodes[n])
                    if d < curr_dist:
                        curr_dist = d
                        curr = n
                        changed = True

        # Beam search at bottom level
        candidates = [(curr_dist, curr)]
        visited = {curr}
        results = [(curr_dist, curr)]

        while candidates:
            d, c = heapq.heappop(candidates)
            if results and d > results[-1][0] and len(results) >= k:
                break
            for n in self.layers[0].get(c, []):
                if n not in visited:
                    visited.add(n)
                    nd = self._cosine_dist(query_vec, self.nodes[n])
                    heapq.heappush(candidates, (nd, n))
                    results.append((nd, n))
                    results.sort()
                    if len(results) > k * 2:
                        results.pop()

        return [(node_id, 1.0 - dist) for dist, node_id in results[:k]]
