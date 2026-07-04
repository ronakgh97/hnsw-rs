//! # tiny-hnsw
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
//! - **Parallel Processing**: Uses [rayon](https://crates.io/crates/rayon) for parallel operations where possible
//!
//! ## Examples
//!
//!```rust
//! use hnsw_rs::prelude::*;
//!
//! fn main() {
//!     // create a vector store with Euclidean distance
//!     let store = FlatVectorStore::init(128, Metrics::Euclidean, 100_000);
//!     // build the HNSW graph (M=16, ef=200, 12 layers)
//!     let mut hnsw = HNSW::new(store, 16, 200, 12, 1.0 / (16.0_f32).ln(), 100_000, true, true);
//!
//!     let (mut vectors, _) = gen_vec(100, 128, 42);
//!
//!     for v in vectors.iter_mut() {
//!         let mut uuid = [0u8; 32];
//!         fastrand::fill(&mut uuid);
//!         let level = hnsw.get_random_level();
//!         hnsw.insert(uuid, v, vec![], level).unwrap();
//!     }
//!
//!     let results = hnsw.search(&mut vectors[0], 5, None);
//!     println!("Top 5 neighbors: {:?}", results);
//! }
//!```
//!
//! For more options see [HNSW::with_options](hnsw::HNSW::with_options) and [FlatVectorStore](item::FlatVectorStore).
//!
//! ### Custom storage backends
//!
//! Implement [`ItemBackend`](item::ItemBackend) for any data type — strings, images, locations, bioinformatics etc.
//! See the trait docs for an example.
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
