use std::ptr::read_unaligned;
use wide::f32x8;
use wincode::{SchemaRead, SchemaWrite};

/// Helper function to reduce a SIMD vector to a scalar by summing its elements,
/// without any fancy horizontal add instructions, just a simple sum of the array representation of the SIMD vector.
#[inline(always)]
fn from_f32x8(v: f32x8) -> f32 {
    let a = v.to_array();
    a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7]
}

/// Supported similarity metrics for vector search
#[derive(Debug, Clone, PartialEq, SchemaRead, SchemaWrite)]
pub enum Metrics {
    Cosine,
    Euclidean,
    RawDot,
}

impl Metrics {
    /// Calculate similarity between two vectors based on the selected metric
    #[inline]
    pub fn calculate(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metrics::Cosine => cosine_similarity(a, b),
            Metrics::Euclidean => euclidean_similarity(a, b),
            Metrics::RawDot => dot_product(a, b),
        }
    }

    /// Get string representation for logging/debugging
    #[inline]
    pub fn string(&self) -> String {
        match self {
            Metrics::Cosine => "COSINE".to_string(),
            Metrics::Euclidean => "EUCLIDEAN".to_string(),
            Metrics::RawDot => "DOT_PRODUCT".to_string(),
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

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // Process 8 elements at a time with SIMD
    for i in 0..chunks {
        let offset = i * 8;
        unsafe {
            let va_ptr = a_ptr.add(offset);
            let vb_ptr = b_ptr.add(offset);

            let va = f32x8::from(read_unaligned(va_ptr as *const [f32; 8]));
            let vb = f32x8::from(read_unaligned(vb_ptr as *const [f32; 8]));
            dot = va.mul_add(vb, dot);
            norm_a = va.mul_add(va, norm_a);
            norm_b = vb.mul_add(vb, norm_b);
        }
    }

    // Reduce SIMD vectors to scalars
    let mut dot_sum = from_f32x8(dot);
    let mut na_sum = from_f32x8(norm_a);
    let mut nb_sum = from_f32x8(norm_b);

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

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        unsafe {
            let va_ptr = a_ptr.add(offset);
            let vb_ptr = b_ptr.add(offset);

            let va = f32x8::from(read_unaligned(va_ptr as *const [f32; 8]));
            let vb = f32x8::from(read_unaligned(vb_ptr as *const [f32; 8]));
            use std::ops::Sub;
            let diff = va.sub(vb);

            sum_sq = diff.mul_add(diff, sum_sq);
        }
    }

    // Reduce SIMD vector to scalar
    let mut dist = from_f32x8(sum_sq);

    // Handle remainder
    let remainder_start = chunks * 8;
    for i in remainder_start..a.len() {
        let diff = a[i] - b[i];
        dist += diff * diff;
    }

    1.0 / (1.0 + dist.sqrt())
}

#[inline(always)]
/// SIMD-optimized raw Dot product
/// Returns value in `[-inf, inf]`
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        // Yes, I'm doing this *unnecessary* check, because I like symmetric, so stfu
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

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        unsafe {
            let offset = i * 8;

            let va_ptr = a_ptr.add(offset);
            let ba_ptr = b_ptr.add(offset);

            let va = f32x8::from(read_unaligned(va_ptr as *const [f32; 8]));
            let vb = f32x8::from(read_unaligned(ba_ptr as *const [f32; 8]));
            sum = va.mul_add(vb, sum);
        }
    }

    // Reduce SIMD vector to scalar
    let mut total_sum = from_f32x8(sum);

    // Handle remainder
    let remainder_start = chunks * 8;
    for i in remainder_start..a.len() {
        total_sum += a[i] * b[i];
    }

    total_sum
}

#[inline(always)]
/// Multiply two matrices using 4 simd registers at a time, fallbacks if less, returning the resulting matrix
/// The matrices are expected to be in row-major order and the dimensions must match
pub fn matmul(
    matrix_a: &[f32],
    matrix_b: &[f32],
    rows_a: usize,
    cols_a: usize,
    rows_b: usize,
    cols_b: usize,
) -> Vec<f32> {
    let mut result = vec![0.0f32; rows_a * cols_b];
    let mut b_t = vec![0.0f32; rows_b * cols_b];
    matmul_into(
        matrix_a,
        matrix_b,
        rows_a,
        cols_a,
        rows_b,
        cols_b,
        &mut b_t,
        &mut result,
    );
    result
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
/// In-place matmul, the result is stored in the `result` slice, which must be pre-allocated and zeroed to the correct size (rows_a * cols_b).
/// The matrix is expected to be in row-major order and the dimensions must match, otherwise it will panic.
pub fn matmul_into(
    matrix_a: &[f32],
    matrix_b: &[f32],
    rows_a: usize,
    cols_a: usize,
    rows_b: usize,
    cols_b: usize,
    b_t: &mut [f32],
    result: &mut [f32],
) {
    if cols_a != rows_b {
        panic!(
            "Inner dimensions must match for multiplication, got cols_a: {}, rows_b: {}",
            cols_a, rows_b
        );
    }

    let size_a = rows_a * cols_a;
    let size_b = rows_b * cols_b;
    let size_res = rows_a * cols_b;
    if matrix_a.len() != size_a
        || matrix_b.len() != size_b
        || result.len() != size_res
        || b_t.len() != size_b
    {
        panic!(
            "Size mismatch: matrix_a {}, matrix_b {}, result {}, b_t {}, expected a {}, b {}, result {}, b_t {}",
            matrix_a.len(),
            matrix_b.len(),
            result.len(),
            b_t.len(),
            size_a,
            size_b,
            size_res,
            size_b,
        );
    }

    if result.len() != size_res {
        panic!(
            "Result buffer size mismatch: expected {}, got {}",
            size_res,
            result.len()
        );
    }

    transpose_mat_into(rows_b, cols_b, matrix_b, b_t);
    result.fill(0.0f32);

    // Number of rows of A to process in one block
    let block_a = 256;
    // Number of columns of B to process in one block
    let block_b = 128;
    // Number of elements in the inner dimension to process in one block,
    // This should be tuned and large enough to amortize the overhead but small enough to fit in cache
    let block_c = 64;

    let b_ptr = b_t.as_ptr();

    for i in (0..rows_a).step_by(block_a) {
        for j in (0..cols_b).step_by(block_b) {
            for k in (0..cols_a).step_by(block_c) {
                // Clamp to matrix dimensions
                let i_max = (i + block_a).min(rows_a);
                let j_max = (j + block_b).min(cols_b);
                let k_max = (k + block_c).min(cols_a);

                //TODO: Try to fetch the next block of A and B into L1 cache, this is a bit tricky because we have to calculate the correct offsets

                // Process the block of A rows against the block of B columns
                for row in i..i_max {
                    // Get the current row of A, we will use it across the block of B columns
                    let a_row = &matrix_a[row * cols_a..(row + 1) * cols_a];
                    let a_ptr = a_row.as_ptr();

                    // Check how many full 4-column blocks we can process it
                    // (j_max - j) is the number of columns in this block,
                    // we want to round it down to the nearest multiple of 4
                    let col_limit = j_max - ((j_max - j) % 4);

                    // Step forward by 4 columns at a time
                    for col in (j..col_limit).step_by(4) {
                        let offset = 8;
                        let mut sum0 = f32x8::ZERO;
                        let mut sum1 = f32x8::ZERO;
                        let mut sum2 = f32x8::ZERO;
                        let mut sum3 = f32x8::ZERO;

                        let b0_strt = col * cols_a;
                        let b1_strt = (col + 1) * cols_a;
                        let b2_strt = (col + 2) * cols_a;
                        let b3_strt = (col + 3) * cols_a;

                        let mut t = k;
                        while t + offset <= k_max {
                            unsafe {
                                // Pull 8 elements from the current row of A
                                let va_ptr = a_ptr.add(t);
                                let va = f32x8::from(read_unaligned(va_ptr as *const [f32; 8]));

                                let b0_ptr = b_ptr.add(b0_strt + t);
                                let b1_ptr = b_ptr.add(b1_strt + t);
                                let b2_ptr = b_ptr.add(b2_strt + t);
                                let b3_ptr = b_ptr.add(b3_strt + t);

                                // Pull 8 elements from the columns of B (which are rows in b_t)
                                let b0 = f32x8::from(read_unaligned(b0_ptr as *const [f32; 8]));
                                let b1 = f32x8::from(read_unaligned(b1_ptr as *const [f32; 8]));
                                let b2 = f32x8::from(read_unaligned(b2_ptr as *const [f32; 8]));
                                let b3 = f32x8::from(read_unaligned(b3_ptr as *const [f32; 8]));

                                sum0 = va.mul_add(b0, sum0);
                                sum1 = va.mul_add(b1, sum1);
                                sum2 = va.mul_add(b2, sum2);
                                sum3 = va.mul_add(b3, sum3);
                            }
                            t += offset;
                        }

                        let base = row * cols_b + col;
                        result[base] += from_f32x8(sum0);
                        result[base + 1] += from_f32x8(sum1);
                        result[base + 2] += from_f32x8(sum2);
                        result[base + 3] += from_f32x8(sum3);

                        // Handle leftovers
                        for t in t..k_max {
                            //TODO: Double result access, bad if result does not fit in cache
                            let a_val = a_row[t];
                            result[base] += a_val * b_t[b0_strt + t];
                            result[base + 1] += a_val * b_t[b1_strt + t];
                            result[base + 2] += a_val * b_t[b2_strt + t];
                            result[base + 3] += a_val * b_t[b3_strt + t];
                        }
                    }

                    // Handle remaining columns that don't fit into a 4-column block
                    for col in col_limit..j_max {
                        let wide = 8;
                        let mut sum = f32x8::ZERO;
                        let b_base = col * cols_a;

                        let mut t = k;

                        while t + wide <= k_max {
                            unsafe {
                                let va_ptr = a_ptr.add(t);
                                let vb_ptr = b_ptr.add(b_base + t);

                                let va = f32x8::from(read_unaligned(va_ptr as *const [f32; 8]));
                                let vb = f32x8::from(read_unaligned(vb_ptr as *const [f32; 8]));

                                sum = va.mul_add(vb, sum);
                            }
                            t += wide;
                        }

                        let mut sum_f32 = from_f32x8(sum);

                        // Handle tail for this column
                        for t in t..k_max {
                            sum_f32 += a_row[t] * b_t[b_base + t];
                        }

                        result[row * cols_b + col] += sum_f32;
                    }
                }
            }
        }
    }
}

#[inline(always)]
/// In-place transpose of a matrix, the input and output slices must be the same size and the matrix is expected to be in row-major order.
/// The output will also be in row-major order but with rows and columns swapped.
fn transpose_mat_into(rows: usize, cols: usize, matrix: &[f32], output: &mut [f32]) {
    let len = rows * cols;
    if len != matrix.len() || len != output.len() {
        panic!(
            "Size mismatch: rows={} cols={} input={} output={}",
            rows,
            cols,
            matrix.len(),
            output.len()
        );
    }
    for col in 0..cols {
        let start = col * rows;
        let in_ptr = matrix.as_ptr();
        let out_ptr = output.as_mut_ptr();
        for row in 0..rows {
            unsafe {
                *out_ptr.add(start + row) = *in_ptr.add(row * cols + col);
            }
        }
    }
}

#[test]
fn test_maths() {
    use crate::utils::gen_vec;
    use std::time::Instant;
    let run = 2148;
    let dim = 1536;

    let (vec, _seed) = gen_vec(2, dim, 42);

    let strt = Instant::now();
    for _ in 0..run {
        dot_product(&vec[0], &vec[1]);
    }
    let elapsed = strt.elapsed();

    let gflops = 2.0 * (dim as f64) / elapsed.as_secs_f64() / 1e9 * run as f64;
    println!("Dot product: {:?}, GLOPS: {}", elapsed, gflops);

    let strt = Instant::now();
    for _ in 0..run {
        cosine_similarity(&vec[0], &vec[1]);
    }
    let elapsed = strt.elapsed();

    let gflops = 6.0 * (dim as f64) / elapsed.as_secs_f64() / 1e9 * run as f64;
    println!("Cosine similarity: {:?}, GLOPS: {}", elapsed, gflops);

    let strt = Instant::now();
    for _ in 0..run {
        euclidean_similarity(&vec[0], &vec[1]);
    }
    let elapsed = strt.elapsed();

    let gflops = 3.0 * (dim as f64) / elapsed.as_secs_f64() / 1e9 * run as f64;
    println!("Euclidean similarity: {:?}, GLOPS: {}", elapsed, gflops);
}
