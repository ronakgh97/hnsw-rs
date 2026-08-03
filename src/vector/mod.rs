use std::fmt::Display;
use wincode::{SchemaRead, SchemaWrite};

#[cfg(not(target_arch = "x86_64"))]
pub mod portable;
#[cfg(not(target_arch = "x86_64"))]
pub use portable::*;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;
#[cfg(target_arch = "x86_64")]
pub use x86_64::*;

/// Supported similarity metrics for vector search.
/// Internally uses SIMD-optimized implementations from this module.
#[derive(Debug, Clone, PartialEq, SchemaRead, SchemaWrite)]
pub enum Metrics {
    /// Cosine similarity (vectors are L2-normalized on insert/search).
    /// Range `[-1, 1]`, higher = more similar.
    Cosine,
    /// Euclidean distance converted to similarity: `1/(1+dist.sqrt())`.
    /// Higher = more similar.
    Euclidean,
    /// Raw dot product (no normalization).
    /// Higher = more similar.
    RawDot,
}

impl Metrics {
    #[inline]
    /// Calculate similarity between two vectors based on the selected metric
    pub fn calculate_sim(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metrics::Cosine => cosine_similarity(a, b),
            Metrics::Euclidean => euclidean_similarity(a, b),
            Metrics::RawDot => dot_product(a, b),
        }
    }

    #[inline]
    /// Calculate distance between two vectors based on the selected metric
    pub fn calculate_dist(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metrics::Cosine => 1.0 - cosine_similarity(a, b),
            Metrics::Euclidean => 1.0 / (euclidean_similarity(a, b) - 1.0),
            Metrics::RawDot => -dot_product(a, b),
        }
    }

    #[inline]
    /// Calculate similarity between two i8 vectors based on the selected metric
    pub fn calculate_sim_i8(&self, a: &[i8], b: &[i8]) -> f32 {
        match self {
            Metrics::Cosine => cosine_similarity_i8(a, b),
            Metrics::Euclidean => euclidean_similarity_i8(a, b),
            Metrics::RawDot => dot_product_i8(a, b),
        }
    }

    #[inline]
    /// Calculate distance between two i8 vectors based on the selected metric
    pub fn calculate_dist_i8(&self, a: &[i8], b: &[i8]) -> f32 {
        match self {
            Metrics::Cosine => 1.0 - cosine_similarity_i8(a, b),
            Metrics::Euclidean => 1.0 / (euclidean_similarity_i8(a, b) - 1.0),
            Metrics::RawDot => -dot_product_i8(a, b),
        }
    }
}

impl Display for Metrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Metrics::Cosine => "COSINE",
                Metrics::Euclidean => "EUCLIDEAN",
                Metrics::RawDot => "DOT",
            }
        )
    }
}

#[test]
fn test_perf() {
    use crate::quant::quant;
    use crate::utils::VectorGen;
    use std::hint::black_box;
    use std::time::Instant;

    let run = 4096;
    let dim = 3072;

    let vg = VectorGen::generate(6, dim, Some(42));

    let strt = Instant::now();
    for _ in 0..run {
        black_box(dot_product(vg.get(0), vg.get(1)));
    }
    let elapsed = strt.elapsed();

    let gops = 2.0 * (dim as f64) / elapsed.as_secs_f64() / 1e9 * run as f64;
    println!("Dot product: {:?}, GOPS: {}", elapsed, gops);

    let mut quant0 = vec![0i8; dim];
    quant(vg.get(0), &mut quant0);

    let mut quant1 = vec![0i8; dim];
    quant(vg.get(1), &mut quant1);

    let strt = Instant::now();
    for _ in 0..run {
        black_box(dot_product_i8(&quant0, &quant1));
    }
    let elapsed = strt.elapsed();

    let gops = 2.0 * (dim as f64) / elapsed.as_secs_f64() / 1e9 * run as f64;
    println!("Dot product (I8): {:?}, GOPS: {}", elapsed, gops);

    let strt = Instant::now();
    for _ in 0..run {
        black_box(cosine_similarity(vg.get(2), vg.get(3)));
    }
    let elapsed = strt.elapsed();

    let gops = 6.0 * (dim as f64) / elapsed.as_secs_f64() / 1e9 * run as f64;
    println!("Cosine similarity: {:?}, GOPS: {}", elapsed, gops);

    let mut quant2 = vec![0i8; dim];
    quant(vg.get(2), &mut quant2);

    let mut quant3 = vec![0i8; dim];
    quant(vg.get(3), &mut quant3);

    let strt = Instant::now();
    for _ in 0..run {
        black_box(cosine_similarity_i8(&quant2, &quant3));
    }
    let elapsed = strt.elapsed();

    let gops = 6.0 * (dim as f64) / elapsed.as_secs_f64() / 1e9 * run as f64;
    println!("Cosine similarity (I8): {:?}, GOPS: {}", elapsed, gops);

    let strt = Instant::now();
    for _ in 0..run {
        black_box(euclidean_similarity(vg.get(4), vg.get(5)));
    }
    let elapsed = strt.elapsed();

    let gops = 3.0 * (dim as f64) / elapsed.as_secs_f64() / 1e9 * run as f64;
    println!("Euclidean similarity: {:?}, GOPS: {}", elapsed, gops);

    let mut quant4 = vec![0i8; dim];
    quant(vg.get(4), &mut quant4);

    let mut quant5 = vec![0i8; dim];
    quant(vg.get(5), &mut quant5);

    let strt = Instant::now();
    for _ in 0..run {
        black_box(euclidean_similarity_i8(&quant4, &quant5));
    }
    let elapsed = strt.elapsed();

    let gops = 3.0 * (dim as f64) / elapsed.as_secs_f64() / 1e9 * run as f64;
    println!("Euclidean similarity (I8): {:?}, GOPS: {}", elapsed, gops);
}
