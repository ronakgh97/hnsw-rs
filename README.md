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
cargo bench --bench (32bit or 8bit)  -- ../../datasets/dim1536_size1M 4 # 32bit takes around 10+ mins!!!
                              <path to dataset> <num of files to read>
```

> NOTE; This lib uses [Portable SIMD](https://docs.rs/wide/latest/wide/index.html)
> and [x86_64 intrinsics](https://doc.rust-lang.org/core/arch/x86_64/index.html)
> for vector operations, so _peak performance_ can't be guaranteed on non-x86_64 targets, but it should still work &
> compile due to #[cfg].

Benched on I7 14650hx (rust 1.99.0, target: x86_64-pc-windows-msvc)

**_F32 Bench_**

```terminaloutput
# alg. 3 (simple, paper default): max_n: 16, ef_const: 96, max_l: 18, metrics: cosine

Total vectors: 153848, dimension: 1536
Index built in 141.92373s with 1091 insert/s
Size on disk: 988980965 bytes
Search with ef: 32, QPS: 3533.32
Search with ef: 64, QPS: 2136.20
Search with ef: 128, QPS: 1281.79
Search with ef: 256, QPS: 738.33
Search with ef: 512, QPS: 423.83
Search with ef: 768, QPS: 301.98

Recall@12: 0.9850, Time: 31.097332
Recall@24: 0.9870, Time: 32.529297
Recall@48: 0.9910, Time: 33.92178
Recall@96: 0.9941, Time: 34.369007
Recall@192: 0.9960, Time: 36.086975
Recall@384: 0.9972, Time: 38.018383

Simple selection
M: 8, Build Time: 4.6057s, Recall@32: 0.8907
M: 16, Build Time: 8.3876s, Recall@32: 0.9578
M: 32, Build Time: 21.0558s, Recall@32: 0.9852
M: 48, Build Time: 34.2004s, Recall@32: 0.9915
M: 64, Build Time: 66.5581s, Recall@32: 0.9952
M: 96, Build Time: 133.9449s, Recall@32: 0.9973

# same config varying M (Alg. 3), sample_size: 32540
# Note: ef_const IS scales with M (ef_const = max(96, 2*M)) during this run
Heuristic selection
M: 8, Build Time: 8.3946s, Recall@32: 0.9614
M: 16, Build Time: 15.3672s, Recall@32: 0.9866
M: 32, Build Time: 15.0714s, Recall@32: 0.9899
M: 48, Build Time: 14.9273s, Recall@32: 0.9912
M: 64, Build Time: 21.6749s, Recall@32: 0.9955
M: 96, Build Time: 39.1221s, Recall@32: 0.9987
```

![Bench plot](bench/plot_f32.png)

**_I8 Bench_**

```terminaloutput
# same config as 32bit run

Total vectors: 153848, dimension: 1536
Index built in 105.1351777s with 1465 insert/s
Size on disk: 280051021 bytes

Search with ef: 32 took, QPS: 5925.33
Search with ef: 64 took, QPS: 3773.94
Search with ef: 128 took, QPS: 2210.25
Search with ef: 256 took, QPS: 1245.92
Search with ef: 512 took, QPS: 694.52
Search with ef: 768 took, QPS: 491.33

Recall@12: 0.9854, Time: 12.860607
Recall@24: 0.9879, Time: 12.734484
Recall@48: 0.9919, Time: 13.520533
Recall@96: 0.9953, Time: 14.085893
Recall@192: 0.9964, Time: 15.149124
Recall@384: 0.9974, Time: 16.874908

# other bench are not needed here, since they depend on the algorithm properties, not data type. 
```

![Bench plot](bench/plot_i8.png)

> Last updated: 10/20/2026

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
