An implementation of HNSW (Hierarchical Navigable Small World) algorithm for approximate nearest neighbor search.
The implementation is not directly based on the [original paper](https://arxiv.org/pdf/1603.09320), but is inspired
by it. This implementation is simple and easy to understand, while still being *reasonably* efficient and scalable in
future

Checkout this repo: [blaze-db](https://github.com/ronakgh97/blaze-db), which is a vector database built on top of this
HNSW implementation.

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