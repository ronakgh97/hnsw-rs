//! # hnsw_rs
//!
//! A **generic** implementation of the HNSW (Hierarchical Navigable Small World) algorithm for efficient approximate nearest neighbor search.
//! This implementation is based by this [paper](https://arxiv.org/pdf/1603.09320)
//! but isn't fully based on that, I have done some of my own simplifications/modifications and is `reasonably` efficient for most use cases, but not optimized for production use yet [GitHub](https://github.com/ronakgh97/hnsw-rs)
//!
//! ## Benchmarks
//!
//! ![Bench Graph](https://raw.githubusercontent.com/ronakgh97/hnsw-rs/refs/heads/master/bench/plot.png)
//! Refer to [this](https://github.com/ronakgh97/hnsw-rs/blob/master/README.md) for more details on benchmarks
//!
//! ## Features
//!
//! - **Generic**: Pluggable backend via [`ItemBackend`](item::ItemBackend) trait — use [`FlatVectorStore`](item::FlatVectorStore) for vectors (which is specifically optimized using portable SIMD and unsafe code) or implement for any data type (string, locations, images, AST, bioinformatics, any arbitrary quantity etc.)
//! - **[Wincode Support](https://crates.io/crates/wincode)**: Makes serialization/deserialization efficient, in-place and compact for disk-storage
//! - **Parallel Processing**: Uses [Rayon](https://crates.io/crates/rayon) for parallel operations where possible
//!
//! ## References
//!
//! - <https://arxiv.org/pdf/1603.09320>
//! - <https://arxiv.org/pdf/2512.06636>
//! - <https://arxiv.org/pdf/2412.01940v1>
//! - <https://www.techrxiv.org/users/922842/articles/1311476-a-comparative-study-of-hnsw-implementations-for-scalable-approximate-nearest-neighbor-search>
//!

pub mod hnsw;
pub mod item;
pub mod quant;
pub mod storage;
pub mod utils;
pub mod vector;
pub use wincode::*;

pub mod prelude {
    pub use crate::hnsw::*;
    pub use crate::item::*;
    pub use crate::storage::*;
    pub use crate::utils::*;
    pub use crate::vector::*;
}
