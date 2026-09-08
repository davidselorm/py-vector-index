# py-vector-index

High-performance vector similarity search engine and Hierarchical Navigable Small World (HNSW) proximity graph in pure Python without third-party C++ dependencies.

## Features
- **Distance Metrics**: Cosine Similarity, Euclidean ($L_2$), and Dot Product.
- **Inverted File Index (IVF)**: K-means centroid clustering for sub-linear query times.
- **HNSW Graph**: Multi-layer skip-list graph for logarithmic nearest-neighbor routing.
- **Metadata Filtering**: Query-time predicate filtering on vector payloads.
- **Persistence**: Built-in JSON serialization and index restoration.

## Usage
```python
from vector_index import VectorIndex

idx = VectorIndex(dimension=128, metric="cosine")
idx.add("item_1", [0.12, 0.84, ...], meta={"category": "finance"})
results = idx.search(query_vec, top_k=5)
```
