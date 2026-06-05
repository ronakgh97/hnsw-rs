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
# config; max_n: 32, ef_con: 256, m_l: 18, metrics: cosine
Total vectors: 153848, dimension: 1536

Search with ef: 32 took, QPS: 3635.61
Search with ef: 64 took, QPS: 2308.91
Search with ef: 128 took, QPS: 1425.42
Search with ef: 256 took, QPS: 811.65
Search with ef: 512 took, QPS: 463.76
Search with ef: 768 took, QPS: 317.07

Recall@12: 0.9799, Time: 35.553806
Recall@24: 0.9832, Time: 36.579773
Recall@48: 0.9848, Time: 37.701332
Recall@96: 0.9870, Time: 38.356697
Recall@192: 0.9886, Time: 39.76736
Recall@384: 0.9891, Time: 42.389786
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