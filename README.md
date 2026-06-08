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
# config; max_n: 16, ef_const: 96, max_l: 18, metrics: cosine
Total vectors: 153848, dimension: 1536
Building index with 153848 vectors...
Index built in 220.0763757s and cached to disk.

Search with ef: 32 took, QPS: 3123.18
Search with ef: 64 took, QPS: 1914.70
Search with ef: 128 took, QPS: 849.07
Search with ef: 256 took, QPS: 639.72
Search with ef: 512 took, QPS: 387.17
Search with ef: 768 took, QPS: 275.18

Recall@12: 0.9830, Time: 32.465973
Recall@24: 0.9860, Time: 33.647808
Recall@48: 0.9898, Time: 34.24965
Recall@96: 0.9939, Time: 36.305805
Recall@192: 0.9958, Time: 37.395786
Recall@384: 0.9971, Time: 40.241714

# same config varyingM, sample_size: 32540
Build with M=8, time: 12.6110195s
Recall@32 with M=8: 0.9619
Build with M=16, time: 21.5927678s
Recall@32 with M=16: 0.9878
Build with M=32, time: 19.3520724s
Recall@32 with M=32: 0.9897
Build with M=48, time: 19.4276249s
Recall@32 with M=48: 0.9916
Build with M=64, time: 19.1534647s
Recall@32 with M=64: 0.9925
Build with M=96, time: 19.2955032s
Recall@32 with M=96: 0.9924
```

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