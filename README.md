An implementation of **HNSW (Hierarchical Navigable Small World)** algorithm for approximate nearest neighbor search.
This is not directly based on the [original paper](https://arxiv.org/pdf/1603.09320), it is simplified and easy to
understand, while still being *reasonably* efficient and robust, I guess

Checkout this repo: [blaze-db](https://github.com/ronakgh97/blaze-db), which is a vector database built on top of this
HNSW implementation.

**How to bench**?

Make sure to have cargo & this [dataset](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M)

```shell
cargo run --bench bench  -- ../../datasets/dim1536_size1M 4
                                         <path to dataset> <num of files to read>
```

```shell
# config; max_n: 32, ef_const: 256, max_l: 32, metrics: cosine
Total vectors: 153848 dimension: 1536
Building index with 153848 vectors...
Index built in 540.5682286s and cached to disk.

Search with ef: 32 took, QPS: 1706.59
Search with ef: 64 took, QPS: 1032.82
Search with ef: 128 took, QPS: 616.87
Search with ef: 256 took, QPS: 364.37
Search with ef: 512 took, QPS: 210.39
Search with ef: 768 took, QPS: 155.18

Recall@12: 0.9985, Time: 32.774128
Recall@24: 0.9989, Time: 33.136284
Recall@48: 0.9990, Time: 35.135895
Recall@96: 0.9993, Time: 36.384808
Recall@192: 0.9996, Time: 38.89311
Recall@384: 0.9997, Time: 42.66928
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