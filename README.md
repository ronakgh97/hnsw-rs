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
# config; max_n: 16, ef_const: 96, max_l: 18, metrics: cosine
Total vectors: 153848, dimension: 1536
Building index with 153848 vectors...
Index built in 192.7884992s and cached to disk.

Search with ef: 32 took, QPS: 2858.56
Search with ef: 64 took, QPS: 1806.46
Search with ef: 128 took, QPS: 1082.76
Search with ef: 256 took, QPS: 636.78
Search with ef: 512 took, QPS: 367.97
Search with ef: 768 took, QPS: 264.41

Recall@12: 0.9784, Time: 33.53304
Recall@24: 0.9865, Time: 33.891148
Recall@48: 0.9905, Time: 34.027683
Recall@96: 0.9942, Time: 35.76142
Recall@192: 0.9957, Time: 36.853363
Recall@384: 0.9972, Time: 39.917393

# same config varyingM, sample_size: 65536
Build with M=8, time: 31.1000744s
Recall@32 with M=8: 0.9653
Build with M=16, time: 57.0895427s
Recall@32 with M=16: 0.9881
Build with M=32, time: 57.315268s
Recall@32 with M=32: 0.9921
Build with M=48, time: 53.3230103s
Recall@32 with M=48: 0.9936
Build with M=64, time: 56.4618391s
Recall@32 with M=64: 0.9934
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