//! # hnsw_rs
//!
//! An implementation of the HNSW (Hierarchical Navigable Small World) algorithm for efficient approximate nearest neighbor search.
//! This implementation is inspired by this [paper](https://arxiv.org/pdf/1603.09320)
//! but isn't fully based on that, I have done some my own simplifications and is `reasonably` efficient for most use cases, but not optimized for production use yet, and is still in early stages of development [GitHub](https://github.com/ronakgh97/hnsw-rs)
//!
//! ```rust
//! use hnsw_rs::hnsw::*;
//! fn main() {
//!     let hnsw = HNSW::default();
//!
//!     // fill with rand 1024 vec, 3 dim for testing
//!     hnsw.fast_fill(1024, 3).unwrap();
//!
//!     // perform search for 10 nearest neighbors
//!     let query = vec![0.5, 0.5, 0.5];
//!     let results = hnsw.search(&query, 10, None);
//!     println!("Top 10 results: {:#?}", results);
//!     assert_eq!(results.len(), 10);
//! }
//! ```
//!
//! ## Features
//!
//! - **Wincode Support**: Makes serialization/deserialization efficient and compact for disk-storage
//! - **Multiple Metrics**: Support for Cosine, Euclidean, and DotProduct similarity
//! - **SIMD Optimized**: Uses Portable SIMD instructions for fast vector computations
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
pub mod hnsw;
pub mod maths;
pub mod quant;
pub mod storage;
pub mod utils;

pub mod prelude {
    pub use crate::hnsw::*;
    pub use crate::maths::*;
    pub use crate::storage::*;
    pub use crate::utils::*;
}

#[test]
fn lib_hnsw_test() {
    use crate::prelude::*;
    let mut hnsw = HNSW::default();
    assert_eq!(hnsw.size(), 0);
    hnsw.fast_fill(1024, 128).unwrap();
    println!("Inserted: 1024");
    hnsw.fast_fill(1024, 128).unwrap();
    println!("Inserted: 2048");
    hnsw.fast_fill(1024, 128).unwrap();
    println!("Inserted: 3072");
    hnsw.fast_fill(1024, 128).unwrap();
    println!("Inserted: 4096");
    assert_eq!(hnsw.size(), 4096);
    println!("Mem-size: {} bytes", hnsw.mem_size());
    hnsw.debug(None);
}
