**Generic** implementation of **HNSW (Hierarchical Navigable Small World)** algorithm for approximate nearest neighbor
search. This is pretty much ["paper-compliant"](https://arxiv.org/pdf/1603.09320), its simplified, well documentated and
easy to understand & reason, while still being _reasonably_ efficient and robust, I guess. Its can be used to
`nearest-search` in _any SPACE_ with given distance valid metric, for examples: vector, string, locations, image, audio,
bioinformatics etc.

Checkout this repo: [blaze-db](https://github.com/ronakgh97/blaze-db), [ARCHIVED/ABANDON]
a vector database built on top of this HNSW implementation.

**How to bench**?

Make sure to have cargo & this [dataset](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M), and also
change [build config](.cargo/config.toml).

```shell
cargo bench --bench bencher  -- ../../datasets/dim1536_size1M 4 # takes around 10+ mins
                              <path to dataset> <num of files to read>
```

> NOTE; This lib uses [Portable SIMD](https://docs.rs/wide/1.6.0/wide/index.html)
> and [x86_64 intrinsics](https://doc.rust-lang.org/core/arch/x86_64/index.html)
> for vector operations, so _Peak performance_ can't be guaranteed on non-x86_64 targets, but it should still work &
> compile due to #[cfg].

Benched on I7 14650hx (rust 1.97.1, target: x86_64-pc-windows-msvc)

```shell
# Alg. 3 (simple, paper default): max_n: 16, ef_const: 96, max_l: 18, metrics: cosine
Total vectors: 153848, dimension: 1536
Index built in 146.9779942s with 1053 insert/s

Search with ef: 32 took, QPS: 3485.94
Search with ef: 64 took, QPS: 2127.37
Search with ef: 128 took, QPS: 1242.32
Search with ef: 256 took, QPS: 724.93
Search with ef: 512 took, QPS: 417.44
Search with ef: 768 took, QPS: 298.97

Recall@12: 0.9866, Time: 31.75659
Recall@24: 0.9877, Time: 32.119545
Recall@48: 0.9905, Time: 32.417294
Recall@96: 0.9939, Time: 33.47838
Recall@192: 0.9958, Time: 34.67813
Recall@384: 0.9971, Time: 37.336834

Simple selection
M: 8, Build Time: 4.9248s, Recall@32: 0.8809
M: 16, Build Time: 9.3076s, Recall@32: 0.9627
M: 32, Build Time: 20.9360s, Recall@32: 0.9866
M: 48, Build Time: 37.4348s, Recall@32: 0.9914
M: 64, Build Time: 66.7094s, Recall@32: 0.9962
M: 96, Build Time: 143.2905s, Recall@32: 0.9986

# same config varying M (Alg. 3), sample_size: 32540
# Note: ef_const IS scales with M (ef_const = max(96, 2*M)) during this run
Heuristic selection
M: 8, Build Time: 8.5594s, Recall@32: 0.9591
M: 16, Build Time: 14.9602s, Recall@32: 0.9876
M: 32, Build Time: 15.3445s, Recall@32: 0.9906
M: 48, Build Time: 15.7295s, Recall@32: 0.9911
M: 64, Build Time: 21.7596s, Recall@32: 0.9958
M: 96, Build Time: 38.0011s, Recall@32: 0.9991
```

> Last updated: 08/03/2026

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
- https://suif.stanford.edu/papers/lam-asplos91.pdf
- https://www.pinecone.io/learn/series/faiss/hnsw/
- https://www.techrxiv.org/users/922842/articles/1311476-a-comparative-study-of-hnsw-implementations-for-scalable-approximate-nearest-neighbor-search

> Note: Some of the ref are of my TODO list, I have not read them yet, but I think they are relevant, so I put them here
> for future reference.
