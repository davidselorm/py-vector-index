class HNSWGraph:
    def __init__(self, m=16, ef=64):
        self.m = m
        self.ef = ef
        self.entry_point = None
    def connect_neighbors(self, node_id, neighbors):
        return neighbors[:self.m]
