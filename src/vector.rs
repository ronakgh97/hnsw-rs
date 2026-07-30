use std::ptr::{read_unaligned, write_unaligned};
use wide::{f32x8, i8x32, i32x8};
use wincode::{SchemaRead, SchemaWrite};

#[inline(always)]
fn unpack(v: i8x32) -> (i32x8, i32x8, i32x8, i32x8) {
    let a = v.to_array();

    #[rustfmt::skip]
    let a0 = i32x8::from([
        a[0] as i32, a[1] as i32, a[2] as i32, a[3] as i32,
        a[4] as i32, a[5] as i32, a[6] as i32, a[7] as i32,
    ]);

    #[rustfmt::skip]
    let a1 = i32x8::from([
        a[8] as i32, a[9] as i32, a[10] as i32, a[11] as i32,
        a[12] as i32, a[13] as i32, a[14] as i32, a[15] as i32,
    ]);

    #[rustfmt::skip]
    let a2 = i32x8::from([
        a[16] as i32, a[17] as i32, a[18] as i32, a[19] as i32,
        a[20] as i32, a[21] as i32, a[22] as i32, a[23] as i32,
    ]);

    #[rustfmt::skip]
    let a3 = i32x8::from([
        a[24] as i32, a[25] as i32, a[26] as i32, a[27] as i32,
        a[28] as i32, a[29] as i32, a[30] as i32, a[31] as i32,
    ]);

    (a0, a1, a2, a3)
}

/// Supported similarity metrics for vector search.
/// Internally uses SIMD-optimized implementations from this module.
#[derive(Debug, Clone, PartialEq, SchemaRead, SchemaWrite)]
pub enum Metrics {
    /// Cosine similarity (vectors are L2-normalized on insert/search). Range `[-1, 1]`, higher = more similar.
    Cosine,
    /// Euclidean distance converted to similarity: `1/(1+dist)`. Higher = more similar.
    Euclidean,
    /// Raw dot product (no normalization). Higher = more similar.
    RawDot,
}

impl Metrics {
    /// Calculate similarity between two vectors based on the selected metric.
    /// Both slices must have the same length or get UB :)
    #[inline]
    pub fn calculate_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metrics::Cosine => unsafe { cosine_similarity(a, b) },
            Metrics::Euclidean => unsafe { euclidean_similarity(a, b) },
            Metrics::RawDot => unsafe { dot_product(a, b) },
        }
    }

    // pub fn calculate_i8(&self, a: &[i8], b: &[i8]) -> i8 {}

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

/// SIMD-optimized cosine similarity, use this for raw, un-normalized 32 vectors.
/// Safety: No bound/length checks
#[inline(always)]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    const LANE: usize = 4 * 8;

    // process 32 elements per iteration
    // (4 * 8-lane vectors)
    let chunks = a.len() / LANE;

    let mut dot0 = f32x8::ZERO;
    let mut dot1 = f32x8::ZERO;

    let mut na0 = f32x8::ZERO;
    let mut na1 = f32x8::ZERO;

    let mut nb0 = f32x8::ZERO;
    let mut nb1 = f32x8::ZERO;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * LANE;
        unsafe {
            let va0_ptr = a_ptr.add(offset);
            let va1_ptr = a_ptr.add(offset + 8);
            let va2_ptr = a_ptr.add(offset + 16);
            let va3_ptr = a_ptr.add(offset + 24);

            let vb0_ptr = b_ptr.add(offset);
            let vb1_ptr = b_ptr.add(offset + 8);
            let vb2_ptr = b_ptr.add(offset + 16);
            let vb3_ptr = b_ptr.add(offset + 24);

            let va0 = f32x8::from(read_unaligned(va0_ptr as *const [f32; 8]));
            let va1 = f32x8::from(read_unaligned(va1_ptr as *const [f32; 8]));
            let va2 = f32x8::from(read_unaligned(va2_ptr as *const [f32; 8]));
            let va3 = f32x8::from(read_unaligned(va3_ptr as *const [f32; 8]));

            let vb0 = f32x8::from(read_unaligned(vb0_ptr as *const [f32; 8]));
            let vb1 = f32x8::from(read_unaligned(vb1_ptr as *const [f32; 8]));
            let vb2 = f32x8::from(read_unaligned(vb2_ptr as *const [f32; 8]));
            let vb3 = f32x8::from(read_unaligned(vb3_ptr as *const [f32; 8]));

            dot0 = va0.mul_add(vb0, dot0);
            dot1 = va1.mul_add(vb1, dot1);
            dot0 = va2.mul_add(vb2, dot0);
            dot1 = va3.mul_add(vb3, dot1);

            na0 = va0.mul_add(va0, na0);
            na1 = va1.mul_add(va1, na1);
            na0 = va2.mul_add(va2, na0);
            na1 = va3.mul_add(va3, na1);

            nb0 = vb0.mul_add(vb0, nb0);
            nb1 = vb1.mul_add(vb1, nb1);
            nb0 = vb2.mul_add(vb2, nb0);
            nb1 = vb3.mul_add(vb3, nb1);
        }
    }

    // reduce scalars
    let mut dot_sum = f32x8::reduce_add(dot0 + dot1);
    let mut na_sum = f32x8::reduce_add(na0 + na1);
    let mut nb_sum = f32x8::reduce_add(nb0 + nb1);

    // handle remaining elements
    let remainder_start = chunks * LANE;
    for i in remainder_start..a.len() {
        dot_sum = a[i].mul_add(b[i], dot_sum);
        na_sum = a[i].mul_add(a[i], na_sum);
        nb_sum = b[i].mul_add(b[i], nb_sum);
    }

    let denominator = (na_sum * nb_sum).sqrt();
    if denominator < f32::EPSILON {
        0.0
    } else {
        dot_sum / denominator
    }
}

/// SIMD-optimized cosine similarity for i8 vectors: `dot / sqrt(norm_a * norm_b)`
/// Safety: No bound/length checks
#[inline(always)]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn cosine_similarity_i8(a: &[i8], b: &[i8]) -> f32 {
    const LANES: usize = 32;

    let chunks = a.len() / LANES;

    let mut dot_acc = i32x8::ZERO;
    let mut na_acc = i32x8::ZERO;
    let mut nb_acc = i32x8::ZERO;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * LANES;

        unsafe {
            let va = i8x32::from(read_unaligned(a_ptr.add(offset) as *const [i8; 32]));
            let vb = i8x32::from(read_unaligned(b_ptr.add(offset) as *const [i8; 32]));

            let (a0, a1, a2, a3) = unpack(va);
            let (b0, b1, b2, b3) = unpack(vb);

            dot_acc += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
            na_acc += a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
            nb_acc += b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3;
        }
    }

    let mut dot = dot_acc.reduce_add();
    let mut norm_a = na_acc.reduce_add();
    let mut norm_b = nb_acc.reduce_add();

    let remainder_start = chunks * LANES;
    for i in remainder_start..a.len() {
        let ai = a[i] as i32;
        let bi = b[i] as i32;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let dot = dot as f32;
    let denominator = (norm_a as f32 * norm_b as f32).sqrt();

    if denominator < f32::EPSILON {
        0.0
    } else {
        dot / denominator
    }
}

/// SIMD-optimized L1-normalize a f32 vector slice in-place
#[inline(always)]
pub fn normalize_l1(v: &mut [f32]) {
    const LANES: usize = 4 * 8;
    // TODO: this is scalar, how lame ;(
    let norm: f32 = v.iter().map(|x| x.abs()).sum();

    unsafe {
        if norm > f32::EPSILON {
            let chunks = v.len() / LANES;
            let inv = 1.0 / norm;
            let inv_v = f32x8::splat(inv);
            let ptr = v.as_mut_ptr();

            for i in 0..chunks {
                let offset = i * LANES;
                let p = ptr.add(offset);

                let v0 = f32x8::from(read_unaligned(p as *const [f32; 8]));
                let v1 = f32x8::from(read_unaligned(p.add(8) as *const [f32; 8]));
                let v2 = f32x8::from(read_unaligned(p.add(16) as *const [f32; 8]));
                let v3 = f32x8::from(read_unaligned(p.add(24) as *const [f32; 8]));

                write_unaligned(p as *mut [f32; 8], (v0 * inv_v).into());
                write_unaligned(p.add(8) as *mut [f32; 8], (v1 * inv_v).into());
                write_unaligned(p.add(16) as *mut [f32; 8], (v2 * inv_v).into());
                write_unaligned(p.add(24) as *mut [f32; 8], (v3 * inv_v).into());
            }

            let remainder = chunks * LANES;
            for i in remainder..v.len() {
                *v.get_unchecked_mut(i) *= inv;
            }
        }
    }
}

/// L2-normalize a f32 vector slice in-place
#[inline(always)]
pub fn normalize_l2(v: &mut [f32]) {
    const LANES: usize = 4 * 8;

    unsafe {
        let dot_prod_norm = dot_product(v, v);
        if dot_prod_norm > f32::EPSILON {
            let chunks = v.len() / LANES;
            let inv = 1.0 / dot_prod_norm.sqrt();
            let inv_v = f32x8::splat(inv);
            let ptr = v.as_mut_ptr();

            for i in 0..chunks {
                let offset = i * LANES;
                let p = ptr.add(offset);

                let v0 = f32x8::from(read_unaligned(p as *const [f32; 8]));
                let v1 = f32x8::from(read_unaligned(p.add(8) as *const [f32; 8]));
                let v2 = f32x8::from(read_unaligned(p.add(16) as *const [f32; 8]));
                let v3 = f32x8::from(read_unaligned(p.add(24) as *const [f32; 8]));

                write_unaligned(p as *mut [f32; 8], (v0 * inv_v).into());
                write_unaligned(p.add(8) as *mut [f32; 8], (v1 * inv_v).into());
                write_unaligned(p.add(16) as *mut [f32; 8], (v2 * inv_v).into());
                write_unaligned(p.add(24) as *mut [f32; 8], (v3 * inv_v).into());
            }

            let remainder = chunks * LANES;
            for i in remainder..v.len() {
                *v.get_unchecked_mut(i) *= inv;
            }
        }
    }
}

/// SIMD-optimized raw Dot product for f32 vectors
/// Safety: No bound/length checks
#[inline(always)]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 4 * 8;

    let chunks = a.len() / LANES;

    let mut sum0 = f32x8::ZERO;
    let mut sum1 = f32x8::ZERO;
    let mut sum2 = f32x8::ZERO;
    let mut sum3 = f32x8::ZERO;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * LANES;
        unsafe {
            let va0_ptr = a_ptr.add(offset);
            let va1_ptr = a_ptr.add(offset + 8);
            let va2_ptr = a_ptr.add(offset + 16);
            let va3_ptr = a_ptr.add(offset + 24);

            let vb0_ptr = b_ptr.add(offset);
            let vb1_ptr = b_ptr.add(offset + 8);
            let vb2_ptr = b_ptr.add(offset + 16);
            let vb3_ptr = b_ptr.add(offset + 24);

            let va0 = f32x8::from(read_unaligned(va0_ptr as *const [f32; 8]));
            let va1 = f32x8::from(read_unaligned(va1_ptr as *const [f32; 8]));
            let va2 = f32x8::from(read_unaligned(va2_ptr as *const [f32; 8]));
            let va3 = f32x8::from(read_unaligned(va3_ptr as *const [f32; 8]));

            let vb0 = f32x8::from(read_unaligned(vb0_ptr as *const [f32; 8]));
            let vb1 = f32x8::from(read_unaligned(vb1_ptr as *const [f32; 8]));
            let vb2 = f32x8::from(read_unaligned(vb2_ptr as *const [f32; 8]));
            let vb3 = f32x8::from(read_unaligned(vb3_ptr as *const [f32; 8]));

            sum0 = va0.mul_add(vb0, sum0);
            sum1 = va1.mul_add(vb1, sum1);
            sum2 = va2.mul_add(vb2, sum2);
            sum3 = va3.mul_add(vb3, sum3);
        }
    }

    // reduce scalar
    let mut total_sum = f32x8::reduce_add((sum0 + sum1) + (sum2 + sum3));

    // handle remainder
    let remainder_start = chunks * LANES;
    for i in remainder_start..a.len() {
        total_sum = a[i].mul_add(b[i], total_sum);
    }

    total_sum
}

/// SIMD-optimized raw Dot product for i8 vectors
/// Safety: No bound/length checks
#[inline(always)]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn dot_product_i8(a: &[i8], b: &[i8]) -> f32 {
    const LANES: usize = 32;

    let chunks = a.len() / LANES;
    let mut acc = i32x8::ZERO;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * LANES;

        unsafe {
            let va = i8x32::from(read_unaligned(a_ptr.add(offset) as *const [i8; 32]));
            let vb = i8x32::from(read_unaligned(b_ptr.add(offset) as *const [i8; 32]));

            let (a0, a1, a2, a3) = unpack(va);
            let (b0, b1, b2, b3) = unpack(vb);

            acc += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
        }
    }

    let mut total_sum = acc.reduce_add();

    let remainder_start = chunks * LANES;
    for i in remainder_start..a.len() {
        total_sum += (a[i] as i32) * (b[i] as i32);
    }

    total_sum as f32
}

/// SIMD-optimized Euclidean similarity f32 vectors
/// Safety: No bound/length checks
#[inline(always)]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn euclidean_similarity(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 4 * 8;
    let chunks = a.len() / LANES;

    let mut sumsq0 = f32x8::ZERO;
    let mut sumsq1 = f32x8::ZERO;
    let mut sumsq2 = f32x8::ZERO;
    let mut sumsq3 = f32x8::ZERO;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * LANES;
        unsafe {
            let va0_ptr = a_ptr.add(offset);
            let va1_ptr = a_ptr.add(offset + 8);
            let va2_ptr = a_ptr.add(offset + 16);
            let va3_ptr = a_ptr.add(offset + 24);

            let vb0_ptr = b_ptr.add(offset);
            let vb1_ptr = b_ptr.add(offset + 8);
            let vb2_ptr = b_ptr.add(offset + 16);
            let vb3_ptr = b_ptr.add(offset + 24);

            let va0 = f32x8::from(read_unaligned(va0_ptr as *const [f32; 8]));
            let va1 = f32x8::from(read_unaligned(va1_ptr as *const [f32; 8]));
            let va2 = f32x8::from(read_unaligned(va2_ptr as *const [f32; 8]));
            let va3 = f32x8::from(read_unaligned(va3_ptr as *const [f32; 8]));

            let vb0 = f32x8::from(read_unaligned(vb0_ptr as *const [f32; 8]));
            let vb1 = f32x8::from(read_unaligned(vb1_ptr as *const [f32; 8]));
            let vb2 = f32x8::from(read_unaligned(vb2_ptr as *const [f32; 8]));
            let vb3 = f32x8::from(read_unaligned(vb3_ptr as *const [f32; 8]));

            let d0 = va0 - vb0;
            let d1 = va1 - vb1;
            let d2 = va2 - vb2;
            let d3 = va3 - vb3;

            sumsq0 = d0.mul_add(d0, sumsq0);
            sumsq1 = d1.mul_add(d1, sumsq1);
            sumsq2 = d2.mul_add(d2, sumsq2);
            sumsq3 = d3.mul_add(d3, sumsq3);
        }
    }

    // reduce to scalar
    let mut dist = f32x8::reduce_add((sumsq0 + sumsq1) + (sumsq2 + sumsq3));

    // handle remainder
    let remainder_start = chunks * LANES;
    for i in remainder_start..a.len() {
        let diff = a[i] - b[i];
        dist = diff.mul_add(diff, dist);
    }

    1.0 / (1.0 + dist.sqrt())
}

/// SIMD-optimized Euclidean similarity i8 vectors: `1.0 / (1.0 + sqrt(sum_of_squared_diffs))`
/// Safety: No bound/length checks
#[inline(always)]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn euclidean_similarity_i8(a: &[i8], b: &[i8]) -> f32 {
    const LANES: usize = 32;
    let chunks = a.len() / LANES;

    let mut acc = i32x8::ZERO;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * LANES;

        unsafe {
            let va = i8x32::from(read_unaligned(a_ptr.add(offset) as *const [i8; 32]));
            let vb = i8x32::from(read_unaligned(b_ptr.add(offset) as *const [i8; 32]));

            let (a0, a1, a2, a3) = unpack(va);
            let (b0, b1, b2, b3) = unpack(vb);

            let d0 = a0 - b0;
            let d1 = a1 - b1;
            let d2 = a2 - b2;
            let d3 = a3 - b3;

            acc += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        }
    }

    let mut dist = acc.reduce_add() as f32;

    let remainder_start = chunks * LANES;
    for i in remainder_start..a.len() {
        let d = a[i] as i32 - b[i] as i32;
        dist += (d * d) as f32;
    }

    1.0 / (1.0 + dist.sqrt())
}

/// Multiply two f32 matrices using 4 SIMD avx2 registers at a time, fallbacks if less, returning the resulting matrix.
/// The matrices are expected to be in row-major order and the dimensions must match
#[inline(always)]
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

// TODO: write portable SIMD for ARM & x86_64, ditch `wide` crate
/// In-place SIMD matmul, the result is stored in the `result` slice, which must be pre-allocated and zeroed to the correct size (rows_a * cols_b).
/// The matrix is expected to be in row-major order and the dimensions must match, otherwise it will panic.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn matmul_into(
    matrix_a: &[f32],
    matrix_b: &[f32],
    rows_a: usize,
    cols_a: usize,
    rows_b: usize,
    cols_b: usize,
    matrix_b_t: &mut [f32],
    result: &mut [f32],
) {
    assert_eq!(
        cols_a, rows_b,
        "Inner dimensions must match for multiplication, got cols_a: {}, rows_b: {}",
        cols_a, rows_b
    );

    assert_eq!(matrix_a.len(), rows_a * cols_a);
    assert_eq!(matrix_b.len(), rows_b * cols_b);
    assert_eq!(matrix_b_t.len(), rows_b * cols_b);
    assert_eq!(result.len(), rows_a * cols_b);

    unsafe { transpose_mat_into(rows_b, cols_b, matrix_b, matrix_b_t) };
    result.fill(0.0f32);

    const BLOCK_A: usize = 256;
    const BLOCK_B: usize = 128;
    const BLOCK_C: usize = 96;

    const LANE_SIZE: usize = 8;
    let b_ptr = matrix_b_t.as_ptr();

    for i in (0..rows_a).step_by(BLOCK_A) {
        let i_max = (i + BLOCK_A).min(rows_a);

        for j in (0..cols_b).step_by(BLOCK_B) {
            let j_max = (j + BLOCK_B).min(cols_b);

            for k in (0..cols_a).step_by(BLOCK_C) {
                let k_max = (k + BLOCK_C).min(cols_a);

                // process the block of A rows against the block of B columns
                for row in i..i_max {
                    let row_base = row * cols_b;
                    // get the current row of A use it across the block of B columns
                    let a_row = unsafe { matrix_a.get_unchecked(row * cols_a..(row + 1) * cols_a) };
                    let a_ptr = a_row.as_ptr();
                    let col_limit = j_max - ((j_max - j) % 4);

                    // step forward by 4 columns at a time (4x8=32 elements)
                    for col in (j..col_limit).step_by(4) {
                        let base = row_base + col;
                        let b0_strt = col * cols_a;
                        let b1_strt = (col + 1) * cols_a;
                        let b2_strt = (col + 2) * cols_a;
                        let b3_strt = (col + 3) * cols_a;

                        let mut sum0 = 0.0f32;
                        let mut sum1 = 0.0f32;
                        let mut sum2 = 0.0f32;
                        let mut sum3 = 0.0f32;

                        let mut vsum0 = f32x8::ZERO;
                        let mut vsum1 = f32x8::ZERO;
                        let mut vsum2 = f32x8::ZERO;
                        let mut vsum3 = f32x8::ZERO;

                        let mut t = k;
                        while t + LANE_SIZE <= k_max {
                            unsafe {
                                // pull 8 elements from the current row of A
                                let va =
                                    f32x8::from(read_unaligned(a_ptr.add(t) as *const [f32; 8]));

                                // pull 32 elements from the columns of B (which are rows in b_t)
                                let b0 = f32x8::from(read_unaligned(
                                    b_ptr.add(b0_strt + t) as *const [f32; 8]
                                ));
                                let b1 = f32x8::from(read_unaligned(
                                    b_ptr.add(b1_strt + t) as *const [f32; 8]
                                ));
                                let b2 = f32x8::from(read_unaligned(
                                    b_ptr.add(b2_strt + t) as *const [f32; 8]
                                ));
                                let b3 = f32x8::from(read_unaligned(
                                    b_ptr.add(b3_strt + t) as *const [f32; 8]
                                ));

                                vsum0 = va.mul_add(b0, vsum0);
                                vsum1 = va.mul_add(b1, vsum1);
                                vsum2 = va.mul_add(b2, vsum2);
                                vsum3 = va.mul_add(b3, vsum3);
                            }
                            t += LANE_SIZE;
                        }

                        // horizontal reduce once
                        sum0 += f32x8::reduce_add(vsum0);
                        sum1 += f32x8::reduce_add(vsum1);
                        sum2 += f32x8::reduce_add(vsum2);
                        sum3 += f32x8::reduce_add(vsum3);

                        // handle leftovers
                        while t < k_max {
                            unsafe {
                                let a_val = *a_ptr.add(t);

                                sum0 += a_val * *b_ptr.add(b0_strt + t);
                                sum1 += a_val * *b_ptr.add(b1_strt + t);
                                sum2 += a_val * *b_ptr.add(b2_strt + t);
                                sum3 += a_val * *b_ptr.add(b3_strt + t);
                            }

                            t += 1;
                        }

                        // single writeback
                        result[base] += sum0;
                        result[base + 1] += sum1;
                        result[base + 2] += sum2;
                        result[base + 3] += sum3;
                    }

                    // handle remaining columns that don't fit into a 4-column block (32 elements)
                    for col in col_limit..j_max {
                        let base = row_base + col;
                        let b_base = col * cols_a;

                        let mut sum = 0.0f32;
                        let mut vsum = f32x8::ZERO;

                        let mut t = k;

                        while t + LANE_SIZE <= k_max {
                            unsafe {
                                let va =
                                    f32x8::from(read_unaligned(a_ptr.add(t) as *const [f32; 8]));
                                let vb = f32x8::from(read_unaligned(
                                    b_ptr.add(b_base + t) as *const [f32; 8]
                                ));

                                vsum = va.mul_add(vb, vsum);
                            }
                            t += LANE_SIZE;
                        }

                        sum += f32x8::reduce_add(vsum);

                        while t < k_max {
                            unsafe {
                                sum += *a_ptr.add(t) * *b_ptr.add(b_base + t);
                            }

                            t += 1;
                        }

                        result[base] += sum;
                    }
                }
            }
        }
    }
}

/// In-place transpose of a matrix, the input and output slices must be the same size and the input matrix is expected to be in row-major order.
/// The output will also be in row-major order but with rows and columns swapped.
#[inline(always)]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn transpose_mat_into(rows: usize, cols: usize, input: &[f32], output: &mut [f32]) {
    // let len = rows * cols;
    // if len != matrix.len() || len != output.len() {
    //     panic!(
    //         "Size mismatch: rows={} cols={} input={} output={}",
    //         rows,
    //         cols,
    //         matrix.len(),
    //         output.len()
    //     );
    // }

    const TILE: usize = 64;

    for ii in (0..rows).step_by(TILE) {
        for jj in (0..cols).step_by(TILE) {
            let i_max = (ii + TILE).min(rows);
            let j_max = (jj + TILE).min(cols);

            for i in ii..i_max {
                for j in jj..j_max {
                    unsafe {
                        *output.get_unchecked_mut(j * rows + i) =
                            *input.get_unchecked(i * cols + j);
                    }
                }
            }
        }
    }
}

#[test]
fn test_maths() {
    use crate::utils::gen_vec;
    use std::hint::black_box;
    use std::time::Instant;
    let run = 4096;
    let dim = 2512;

    let (vec, _seed) = gen_vec(6, dim, 42);

    let strt = Instant::now();
    for _ in 0..run {
        unsafe {
            black_box(dot_product(&vec[0], &vec[1]));
        }
    }
    let elapsed = strt.elapsed();

    let gflops = 2.0 * (dim as f64) / elapsed.as_secs_f64() / 1e9 * run as f64;
    println!("Dot product: {:?}, GLOPS: {}", elapsed, gflops);

    let strt = Instant::now();
    for _ in 0..run {
        unsafe {
            black_box(cosine_similarity(&vec[2], &vec[3]));
        }
    }
    let elapsed = strt.elapsed();

    let gflops = 6.0 * (dim as f64) / elapsed.as_secs_f64() / 1e9 * run as f64;
    println!("Cosine similarity: {:?}, GLOPS: {}", elapsed, gflops);

    let strt = Instant::now();
    for _ in 0..run {
        unsafe {
            black_box(euclidean_similarity(&vec[4], &vec[5]));
        }
    }
    let elapsed = strt.elapsed();

    let gflops = 3.0 * (dim as f64) / elapsed.as_secs_f64() / 1e9 * run as f64;
    println!("Euclidean similarity: {:?}, GLOPS: {}", elapsed, gflops);
}
