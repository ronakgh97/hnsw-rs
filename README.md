An implementation of **HNSW (Hierarchical Navigable Small World)** algorithm for approximate nearest neighbor search.
This is pretty much ["paper-compliant"](https://arxiv.org/pdf/1603.09320), its simplified, well documentated and easy to
understand & reason, while still being *reasonably* efficient and robust, I guess

Checkout this repo: [blaze-db](https://github.com/ronakgh97/blaze-db), [ARCHIVED/ABANDON] which is a vector database built on
top of this HNSW implementation.

**How to bench**?

Make sure to have cargo & this [dataset](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M),
and also change [build config](.cargo/config.toml) to your liking.

```shell
cargo bench --bench bencher  -- ../../datasets/dim1536_size1M 4
                                         <path to dataset> <num of files to read>
```

```shell
# Alg. 3 (simple, paper default): max_n: 16, ef_const: 96, max_l: 18, metrics: cosine
Total vectors: 153848, dimension: 1536
Index built in 136.3167937s with 1131 insert/s
Search with ef: 32 took, QPS: 3487.56
Search with ef: 64 took, QPS: 2166.77
Search with ef: 128 took, QPS: 1282.55
Search with ef: 256 took, QPS: 726.21
Search with ef: 512 took, QPS: 413.41
Search with ef: 768 took, QPS: 291.45

Recall@12: 0.9858, Time: 34.538246
Recall@24: 0.9889, Time: 34.598064
Recall@48: 0.9925, Time: 35.0367
Recall@96: 0.9943, Time: 36.10253
Recall@192: 0.9960, Time: 37.633686
Recall@384: 0.9973, Time: 41.085598

Simple selection
M: 8, Build Time: 5.3749s, Recall@32: 0.5101
M: 16, Build Time: 9.1476s, Recall@32: 0.9703
M: 32, Build Time: 20.5473s, Recall@32: 0.9854
M: 48, Build Time: 36.7546s, Recall@32: 0.9917
M: 64, Build Time: 63.0919s, Recall@32: 0.9969
M: 96, Build Time: 139.5258s, Recall@32: 0.9991

# same config varying M (Alg. 3), sample_size: 32540
# Note: ef_const IS scales with M (ef_const = max(96, 2*M)) during this run
Heuristic selection
M: 8, Build Time: 8.8142s, Recall@32: 0.9669
M: 16, Build Time: 15.2515s, Recall@32: 0.9888
M: 32, Build Time: 14.5271s, Recall@32: 0.9925
M: 48, Build Time: 14.9043s, Recall@32: 0.9938
M: 64, Build Time: 21.8074s, Recall@32: 0.9965
M: 96, Build Time: 38.0243s, Recall@32: 0.9987
```

**some observations**
on [commit](https://github.com/ronakgh97/hnsw-rs/commit/57aa9fd42877c267e22645d7821c1f671ed76896)

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