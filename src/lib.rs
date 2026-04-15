//! # hnsw_rs
//!
//! An implementation of the HNSW (Hierarchical Navigable Small World) algorithm for efficient approximate nearest neighbor search.
//! This implementation is inspired by this [paper](https://arxiv.org/pdf/1603.09320)
//! but isn't fully based on that, I have done some my own simplifications and is reasonably efficient for most use cases, but not optimized for production use yet, and is still in early stages of development [GitHub](https://github.com/ronakgh97/hnsw-rs)
//!
//! ## Quick Start
//!
//! ```rust
//! use hnsw_rs::prelude::*;
//!
//! let mut hnsw = HNSW::default();
//! let vectors = vec![
//!     vec![1.0, 0.0, 0.0],
//!     vec![0.0, 1.0, 0.0],
//!     vec![0.0, 0.0, 1.0],
//! ];
//!
//! for (i, vector) in vectors.iter().enumerate() {
//!     let level = hnsw.get_random_level(); // <-- (-rand.ln() * m).floor(), where is 1/M
//!     hnsw.insert(i.to_string(), vector, vec![], level).unwrap();
//! }
//!
//! let results = hnsw.search(&[1.0, 0.0, 0.0], 2, None);
//! assert_eq!(results.len(), 2);
//! ```
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
//! - `hnsw`: Core HNSW implementation
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
    pub use crate::utils::{gen_fill, gen_vec, get_random_bytes};
}

#[test]
fn basic_insert_test() {
    use crate::prelude::*;
    let mut hnsw = HNSW::default();
    let vectors = gen_vec(32, 128, 42);

    for (i, vector) in vectors.0.iter().enumerate() {
        let level = hnsw.get_random_level();
        hnsw.insert(i.to_string(), vector, vec![], level).unwrap();
    }

    assert_eq!(hnsw.nodes.len(), 32);
}
