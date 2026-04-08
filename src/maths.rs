#[allow(unused_imports)]
use rayon::iter::ParallelIterator;
#[allow(unused_imports)]
use rayon::slice::ParallelSlice;
use wide::f32x8;
use wincode::{SchemaRead, SchemaWrite};

/// Supported similarity metrics for vector search
#[derive(Debug, Clone, PartialEq, SchemaRead, SchemaWrite)]
pub enum Metrics {
    Cosine,
    Euclidean,
    DotProduct,
}

impl Metrics {
    /// Calculate similarity between two vectors based on the selected metric
    #[inline]
    pub fn calculate(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metrics::Cosine => cosine_similarity(a, b),
            Metrics::Euclidean => euclidean_similarity(a, b),
            Metrics::DotProduct => dot_product(a, b),
        }
    }

    /// Get string representation for logging/debugging
    #[inline]
    pub fn string(&self) -> String {
        match self {
            Metrics::Cosine => "COSINE".to_string(),
            Metrics::Euclidean => "EUCLIDEAN".to_string(),
            Metrics::DotProduct => "DOT_PRODUCT".to_string(),
        }
    }
}

#[inline(always)]
/// SIMD-optimized cosine similarity
/// Returns value in `[-1, 1]`
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        // Yes, Im doing this *necessary* check, because I like symmetric, so stfu
        panic!("Vectors must not be empty");
    }
    if a.len() != b.len() {
        panic!(
            "Vector dimensions must match, a: {}, b: {}",
            a.len(),
            b.len()
        );
    }

    let chunks = a.len() / 8;
    let mut dot = f32x8::ZERO;
    let mut norm_a = f32x8::ZERO;
    let mut norm_b = f32x8::ZERO;

    // Process 8 elements at a time with SIMD
    for i in 0..chunks {
        let offset = i * 8;
        let va = f32x8::from(&a[offset..offset + 8]);
        let vb = f32x8::from(&b[offset..offset + 8]);
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    // Reduce SIMD vectors to scalars
    let arr_dot = dot.to_array();
    let arr_na = norm_a.to_array();
    let arr_nb = norm_b.to_array();

    let mut dot_sum: f32 = arr_dot.iter().sum();
    let mut na_sum: f32 = arr_na.iter().sum();
    let mut nb_sum: f32 = arr_nb.iter().sum();

    // Handle remaining elements (tail)
    let remainder_start = chunks * 8;
    for i in remainder_start..a.len() {
        dot_sum += a[i] * b[i];
        na_sum += a[i] * a[i];
        nb_sum += b[i] * b[i];
    }

    let denominator = (na_sum * nb_sum).sqrt();
    if denominator < f32::EPSILON {
        0.0
    } else {
        dot_sum / denominator
    }
}

#[inline(always)]
/// SIMD-optimized Euclidean similarity
/// Returns value in `(0, 1]`
pub fn euclidean_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        // Yes, Im doing this *necessary* check, because I like symmetric, so stfu
        panic!("Vectors must not be empty");
    }
    if a.len() != b.len() {
        panic!(
            "Vector dimensions must match, a: {}, b: {}",
            a.len(),
            b.len()
        );
    }

    let chunks = a.len() / 8;
    let mut sum_sq = f32x8::ZERO;

    for i in 0..chunks {
        let offset = i * 8;
        let va = f32x8::from(&a[offset..offset + 8]);
        let vb = f32x8::from(&b[offset..offset + 8]);
        let diff = va - vb;
        sum_sq += diff * diff;
    }

    let arr = sum_sq.to_array();
    let mut distance_sq: f32 = arr.iter().sum();

    // Handle remainder
    let remainder_start = chunks * 8;
    for i in remainder_start..a.len() {
        let diff = a[i] - b[i];
        distance_sq += diff * diff;
    }

    1.0 / (1.0 + distance_sq.sqrt())
}

#[inline(always)]
/// SIMD-optimized raw dot product
/// Returns value in `[-inf, inf]`
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        // Yes, Im doing this *necessary* check, because I like symmetric, so stfu
        panic!("Vectors must not be empty");
    }
    if a.len() != b.len() {
        panic!(
            "Vector dimensions must match, a: {}, b: {}",
            a.len(),
            b.len()
        );
    }
    let chunks = a.len() / 8;
    let mut sum = f32x8::ZERO;

    for i in 0..chunks {
        let offset = i * 8;
        let va = f32x8::from(&a[offset..offset + 8]);
        let vb = f32x8::from(&b[offset..offset + 8]);
        sum += va * vb;
    }

    let arr = sum.to_array();
    let mut total: f32 = arr.iter().sum();

    // Handle remainder
    let remainder_start = chunks * 8;
    for i in remainder_start..a.len() {
        total += a[i] * b[i];
    }

    total
}

/// Multiplies a matrix (flattened) with a vector, returning the resulting vector.
/// The matrix is expected to be in row-major order and the dimensions must match.
pub fn matrix_vec_mul(matrix: &[f32], vector: &[f32], dim: usize) -> Vec<f32> {
    if dim == 0 {
        panic!("Dimension must be greater than zero");
    }
    if vector.len() != dim || matrix.len() != dim * dim {
        panic!(
            "Dimension mismatch: matrix has {} rows, vector has {} elements, expected dimension {}",
            matrix.len() / dim,
            vector.len(),
            dim
        );
    }

    let mut vec = Vec::with_capacity(dim);
    for i in 0..dim {
        let row = &matrix[i * dim..(i + 1) * dim];
        vec.push(dot_product(row, vector));
    }

    // matrix.par_chunks_exact(dim)
    //     .map(|row| dot_product(row, vector))
    //     .collect();

    vec
} // TODO: We can parallelize these since each row can be computed independently, then it would be ultra-blazing fast 🔥🔥🔥
