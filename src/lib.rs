//! # hnsw_rs
//!
//! An implementation of the HNSW (Hierarchical Navigable Small World) algorithm for efficient approximate nearest neighbor search.
//! This implementation is inspired by this [paper](https://arxiv.org/pdf/1603.09320)
//! but isn't fully based on that, I have done some my own simplifications and is `reasonably` efficient for most use cases, but not optimized for production use yet, and is still in early stages of development [GitHub](https://github.com/ronakgh97/hnsw-rs)
//!
//! ## Features
//!
//! - **Wincode Support**: Makes serialization/deserialization efficient and compact for disk-storage
//! - **Multiple Metrics**: Support for Cosine, Euclidean, and DotProduct similarity
//! - **SIMD Optimized**: Uses SIMD instructions for fast vector computations
//! - **Parallel Processing**: Uses Rayon for parallel operations where possible
//!
//! ## Modules
//!
//! - `prelude`: Re-exports commonly used types and functions
//! - `hnsw`: Kernal HNSW implementation
//! - `storage`: IO operations for saving/loading the HNSW index to/from disk
//! - `maths`: Similarity metric functions
//! - `utils`: Utility functions for testing and benchmarking
//!
//! ## References
//!
//! - <https://arxiv.org/pdf/1603.09320>
//! - <https://arxiv.org/abs/2512.06636>
//! - <https://arxiv.org/html/2412.01940v1>
//! - <https://www.techrxiv.org/users/922842/articles/1311476-a-comparative-study-of-hnsw-implementations-for-scalable-approximate-nearest-neighbor-search>
//!
mod hnsw;
mod maths;
mod quant;
mod storage;
mod utils;

pub mod prelude {
    pub use crate::hnsw::*;
    pub use crate::maths::*;
    pub use crate::storage::*;
    pub use crate::utils::*;
    pub use hex::*;
}

#[test]
fn basic_hnsw_test() {
    use crate::prelude::*;
    let mut hnsw = HNSW::default();
    let (vectors, seed) = gen_vec(32, 128, 42);

    for vector in vectors.iter() {
        let id = encode(gen_bytes(16));
        let meta = gen_bytes(64);
        let level = hnsw.get_random_level();
        hnsw.insert(id, vector, meta, level).unwrap();
    }
    assert_eq!(seed, 74);
    assert_eq!(hnsw.count(), 32);

    hnsw.auto_fill(32).unwrap();
    assert_eq!(hnsw.count(), 64);
    println!("Mem-size: {}", hnsw.size_in_bytes());
}
