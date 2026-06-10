An implementation of **HNSW (Hierarchical Navigable Small World)** algorithm for approximate nearest neighbor search.
This is pretty much "paper-compliant" based on the [original paper](https://arxiv.org/pdf/1603.09320),
it is simplified & documentated version and easy to understand & reason, while still being *reasonably* efficient and
robust, I guess

Checkout this repo: [blaze-db](https://github.com/ronakgh97/blaze-db), which is a vector database built on top of this
HNSW implementation.

**How to bench**?

Make sure to have cargo & this [dataset](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M)

```shell
cargo run --bench bench  -- ../../datasets/dim1536_size1M 4
                                         <path to dataset> <num of files to read>
```

```shell
# Alg. 3 (simple, paper default): max_n: 16, ef_const: 96, max_l: 18, metrics: cosine
Total vectors: 153848, dimension: 1536
Building index with 153848 vectors...
Index built in 220.0763757s and cached to disk.

Search with ef: 32 took, QPS: 3123.18
Search with ef: 64 took, QPS: 1914.70
Search with ef: 128 took, QPS: 949.07
Search with ef: 256 took, QPS: 639.72
Search with ef: 512 took, QPS: 387.17
Search with ef: 768 took, QPS: 275.18

Recall@12: 0.9830, Time: 32.465973
Recall@24: 0.9860, Time: 33.647808
Recall@48: 0.9898, Time: 34.24965
Recall@96: 0.9939, Time: 36.305805
Recall@192: 0.9958, Time: 37.395786
Recall@384: 0.9971, Time: 40.241714

# same config varying M (Alg. 3), sample_size: 32540
# Note: ef_const is NOT scales with M (ef_const = max(96, 2*M)) during this run
Simple selection (Alg. 3, paper default)
Build with M=8, time: 6.40s, Recall@32: 0.8687
Build with M=16, time: 11.25s, Recall@32: 0.9587
Build with M=32, time: 25.16s, Recall@32: 0.9847
Build with M=48, time: 45.78s, Recall@32: 0.9927
Build with M=64, time: 51.29s, Recall@32: 0.9940
Build with M=96, time: 54.42s, Recall@32: 0.9961

Heuristic selection (Alg. 4, opt-in via with_options)
Build with M=8, time: 46.45s, Recall@32: 0.9668
Build with M=16, time: 23.35s, Recall@32: 0.9857
Build with M=32, time: 20.12s, Recall@32: 0.9909
Build with M=48, time: 20.67s, Recall@32: 0.9907
Build with M=64, time: 20.12s, Recall@32: 0.9911
Build with M=96, time: 19.87s, Recall@32: 0.9908
```

**some observations**

- `ef_const = max(96, 2*M)` ensures candidate pool scales with Mmax0 (2*M at layer 0) for both algorithm, but in this
  run, we keep it fixed at 96 to isolate the effect of M
- Alg. 3 (simple_selection): build time grows with M; recall improves monotonically
- Alg. 4 (heuristic_selection): build time peaks at M=8-16, then plateaus; recall peaks at M=32
- At high M, simple selection gets more connections from fixed ef pool → higher recall, but heuristic selection
  starves → lower recall, build time drops due to fewer distance computations, selection overhead dominates, then
  finally it caps due to selection is ef_const-limited, not M-limited
- Use `HNSW::with_options(..., use_heuristic_selection=true)` for Alg. 4

![Bench plot](bench/plot.png)

Ref:

- https://en.wikipedia.org/wiki/Curse_of_dimensionality
- https://arxiv.org/pdf/1603.09320
- https://arxiv.org/abs/2512.06636
- https://arxiv.org/pdf/2406.03482
- https://arxiv.org/html/2412.01940v1
- https://www.pinecone.io/learn/series/faiss/hnsw/
- https://www.techrxiv.org/users/922842/articles/1311476-a-comparative-study-of-hnsw-implementations-for-scalable-approximate-nearest-neighbor-search

> Note: Some of the ref are of my TODO list, I have not read them yet, but I think they are relevant, so I put them here
> for future reference.