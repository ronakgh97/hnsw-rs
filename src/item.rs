use crate::prelude::*;
use wincode::{SchemaRead, SchemaWrite};

/// Trait for storing item behavior and similarity/distance computation in [HNSW],
/// it handles & abstracts away all the internal graph logic,
/// BUT DOES NOT HANDLE DATA Manipulation, ONLY STORE THE [ItemBackend] along with "GRAPH" overhead.
/// Implementor need to use internal/external storage for retriving, pushing data and `distance` computation so HNSW can read/modify them,
/// and maybe use [wincode](https://crates.io/crates/wincode) derive macro [SchemaWrite] & [SchemaRead] for serialization/deserialization support if needed.
///
/// # Implementation example
///
///```rust
/// use hnsw_rs::prelude::*;
///
/// struct HammingBackend {
///     data: Vec<u8>,
///     bytes_per_item: usize,
/// }
///
/// impl ItemBackend for HammingBackend {
///     type Item = [u8];
///
///     fn validate_item(&self, item: &[u8]) -> bool {
///         item.len() == self.bytes_per_item
///     }
///
///     fn search_modify(&self, _item: &mut [u8]) {} // no pre-processing needed
///     fn insert_modify(&mut self, _item: &mut [u8]) {} // no pre-processing needed
///
///     fn get(&self, idx: NodeIndex) -> &[u8] {
///         let start = idx * self.bytes_per_item;
///         &self.data[start..start + self.bytes_per_item]
///     }
///
///     fn len(&self) -> usize { self.data.len() / self.bytes_per_item }
///     fn is_empty(&self) -> bool { self.data.is_empty() }
///     fn mem_size(&self) -> usize { self.data.len() }
///
///     fn similarity(&self, a: &[u8], b: &[u8]) -> f32 {
///         let dist: usize = a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones() as usize).sum();
///         1.0 / (1.0 + dist as f32)
///     }
///
///     fn distance(&self, a: &[u8], b: &[u8]) -> f32 {
///         a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones() as f32).sum()
///     }
/// }
///```
pub trait ItemBackend: Send + Sync {
    /// The type of items being compared.
    type Item: ?Sized + Send + Sync;

    /// Validate an item before insertion/searching into the store.
    fn validate_item(&self, item: &Self::Item) -> bool;

    /// Modify a query item before searching,
    /// This is useful for typical pre-normalizing or pre-transforming the item.
    fn search_modify(&self, item: &mut Self::Item);

    /// Modify an item before inserting into the store,
    /// Takes &mut of the item, so can be potentially modified it in place before storing in GRAPH.
    fn insert_modify(&mut self, item: &mut Self::Item);

    /// Retrieve an item by its index.
    /// Allows some level of optimization, if you know what type of item you are retrieving.
    fn get(&self, idx: NodeIndex) -> &Self::Item;

    /// Get the number of items in the store.
    fn len(&self) -> usize;

    /// Check if the store is empty.
    fn is_empty(&self) -> bool;

    /// Get the memory size of the store in bytes.
    fn mem_size(&self) -> usize;

    /// Compute similarity score between two items.
    /// Higher values indicate greater similarity (used for max-heap ranking).
    fn similarity(&self, a: &Self::Item, b: &Self::Item) -> f32;

    /// Compute distance between two items.
    /// Lower values indicate closer items (used for redundancy checks in heuristic selection).
    fn distance(&self, a: &Self::Item, b: &Self::Item) -> f32;
}

/// Convenience type alias for vector optimized HNSW.
pub type VectorHnsw = HNSW<FlatVectorStore>;

/// A flat vector store that holds 32-bit vectors in a contiguous memory layout.
/// This implements [ItemBackend] trait also derives [wincode::SchemaRead] & [wincode::SchemaWrite] and is specifically optimized for vector search ops.
#[derive(SchemaRead, SchemaWrite)]
pub struct FlatVectorStore {
    flat_vectors: Vec<f32>,
    dim: usize,
    metric: Metrics,
}

impl FlatVectorStore {
    /// Initialize a new FlatVectorStore with specified metric, dimension, and pre-alloc capacity.
    pub fn init(dim: usize, metric: Metrics, with_capacity: usize) -> Self {
        assert!(dim > 0, "Dimension must be greater than zero");
        Self {
            flat_vectors: Vec::with_capacity(with_capacity * dim),
            dim,
            metric,
        }
    }
}

impl Default for FlatVectorStore {
    /// Default initialization with Cosine metric, 1024 dimensions, and ~255k vector capacity.
    fn default() -> Self {
        let dim = 1024;
        let metric = Metrics::Cosine;
        let capacity = 255_000;
        Self {
            flat_vectors: Vec::with_capacity(dim * capacity),
            dim,
            metric,
        }
    }
}

impl ItemBackend for FlatVectorStore {
    /// The item type is a slice of f32, representing a vector in the flat storage.
    type Item = [f32];

    #[inline(always)]
    fn validate_item(&self, item: &Self::Item) -> bool {
        item.len() == self.dim
    }

    #[inline(always)]
    fn search_modify(&self, item: &mut Self::Item) {
        if matches!(self.metric, Metrics::Cosine) {
            normalize_l2(item);
        }
    }

    #[inline(always)]
    fn insert_modify(&mut self, item: &mut Self::Item) {
        if matches!(self.metric, Metrics::Cosine) {
            normalize_l2(item);
        }
        self.flat_vectors.extend_from_slice(item);
    }

    #[inline(always)]
    fn get(&self, idx: NodeIndex) -> &Self::Item {
        let start = idx * self.dim;
        unsafe { self.flat_vectors.get_unchecked(start..start + self.dim) }
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.flat_vectors.len() / self.dim
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.flat_vectors.is_empty()
    }

    #[inline(always)]
    fn mem_size(&self) -> usize {
        size_of::<f32>() * self.flat_vectors.len()
    }

    #[inline(always)]
    fn similarity(&self, a: &Self::Item, b: &Self::Item) -> f32 {
        unsafe {
            match self.metric {
                Metrics::Cosine | Metrics::RawDot => dot_product(a, b), // l2 norm, so both dot
                Metrics::Euclidean => euclidean_similarity(a, b),
            }
        }
    }

    #[inline(always)]
    fn distance(&self, item0: &Self::Item, item1: &Self::Item) -> f32 {
        match self.metric {
            Metrics::Cosine => 1.0 - unsafe { dot_product(item0, item1) },
            Metrics::Euclidean => unsafe { 1.0 / (euclidean_similarity(item0, item1) - 1.0) },
            Metrics::RawDot => -unsafe { dot_product(item0, item1) },
        }
    }
}
