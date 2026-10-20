use std::arch::x86_64::*;

#[macro_export]
/// Horizontal sum of i32x8 `__m256i` vector, returning a single `i32` value, could overflow
macro_rules! hsum_i32x8 {
    ($v:expr) => {{
        let lo = _mm256_extracti128_si256::<0>($v); // [a0, a1, a2, a3]
        let hi = _mm256_extracti128_si256::<1>($v); // [a4, a5, a6, a7]
        let sum = _mm_add_epi32(lo, hi); // [a0+a4, a1+a5, a2+a6, a3+a7]
        let sum = _mm_hadd_epi32(sum, sum); // [a0+a4+a1+a5, a2+a6+a3+a7, a0+a4+a1+a5, a2+a6+a3+a7]
        let sum = _mm_hadd_epi32(sum, sum); // [a0+a4+a1+a5+a2+a6+a3+a7, ..]
        _mm_cvtsi128_si32(sum) // return the first element
    }};
}

#[macro_export]
/// Multiply two i8x32 `__m256i` vectors (preserve sign) and return it as packed i32x8 `__m256i` vector.
///
/// All elements must be in `[-127, 127]`. A value of `-128` produces incorrect results:
/// `sign_epi8(-128, -128)` yields `-128`, and `abs(-128)` wraps to `-128` (not `+128`),
/// causing `maddubs_epi16` to compute `128 × (-128) = -16384` instead of `+16384`.
macro_rules! i8x32_dot_i32x8 {
    ($a:expr, $b:expr) => {
        {
            let sign_b = _mm256_sign_epi8($b, $a); // determine sign
            let abs_a = _mm256_abs_epi8($a); // get abs
            let lo16 = _mm256_maddubs_epi16(abs_a, sign_b); // mul both, produce signed i16
            _mm256_madd_epi16(lo16, _mm256_set1_epi16(1)) // mul again with ones, produce full signed i32
        }
    };
}

#[macro_export]
/// Horizontal sum of f32x8 `__m256d` vector, returning a single `f32` value, could overflow
macro_rules! hsum_f32x8 {
    ($v:expr) => {{
        // first pass; [a,b,c,d | e,f,g,h]
        let hi = _mm256_extractf128_ps($v, 1); // [e,f,g,h]
        let lo = _mm256_castps256_ps128($v); // [a,b,c,d]
        let hsum = _mm_add_ps(lo, hi); // [a+b, c+d, e+f, g+h]

        // second pass; add the two halves together
        let sum = _mm_hadd_ps(hsum, hsum); // [a+e+b+f c+g+d+h a+e+b+f c+g+d+h]
        let sum = _mm_hadd_ps(sum, sum); // [a+e+b+f+c+g+d+h,_,_,_]
        _mm_cvtss_f32(sum) // just extract one
    }};
}

#[inline(always)]
/// SIMD-optimized cosine similarity for f32 vectors, using `AVX2` intrinsics for `x86_64` arch
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector lengths do not match");

    const LANE: usize = 32;
    let simd_range = 0..(a.len() / LANE);
    let reminder_range = (simd_range.end * LANE)..a.len();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let (mut dot, mut na, mut nb) = unsafe {
        let mut dot0 = _mm256_setzero_ps();
        let mut dot1 = _mm256_setzero_ps();

        let mut na0 = _mm256_setzero_ps();
        let mut na1 = _mm256_setzero_ps();

        let mut nb0 = _mm256_setzero_ps();
        let mut nb1 = _mm256_setzero_ps();

        for i in simd_range {
            let offset = i * LANE;

            let va0 = _mm256_loadu_ps(a_ptr.add(offset));
            let va1 = _mm256_loadu_ps(a_ptr.add(offset + 8));
            let va2 = _mm256_loadu_ps(a_ptr.add(offset + 16));
            let va3 = _mm256_loadu_ps(a_ptr.add(offset + 24));

            let vb0 = _mm256_loadu_ps(b_ptr.add(offset));
            let vb1 = _mm256_loadu_ps(b_ptr.add(offset + 8));
            let vb2 = _mm256_loadu_ps(b_ptr.add(offset + 16));
            let vb3 = _mm256_loadu_ps(b_ptr.add(offset + 24));

            dot0 = _mm256_fmadd_ps(va0, vb0, dot0);
            dot1 = _mm256_fmadd_ps(va1, vb1, dot1);
            dot0 = _mm256_fmadd_ps(va2, vb2, dot0);
            dot1 = _mm256_fmadd_ps(va3, vb3, dot1);

            na0 = _mm256_fmadd_ps(va0, va0, na0);
            na1 = _mm256_fmadd_ps(va1, va1, na1);
            na0 = _mm256_fmadd_ps(va2, va2, na0);
            na1 = _mm256_fmadd_ps(va3, va3, na1);

            nb0 = _mm256_fmadd_ps(vb0, vb0, nb0);
            nb1 = _mm256_fmadd_ps(vb1, vb1, nb1);
            nb0 = _mm256_fmadd_ps(vb2, vb2, nb0);
            nb1 = _mm256_fmadd_ps(vb3, vb3, nb1);
        }

        let dot = hsum_f32x8!(dot0) + hsum_f32x8!(dot1);
        let na = hsum_f32x8!(na0) + hsum_f32x8!(na1);
        let nb = hsum_f32x8!(nb0) + hsum_f32x8!(nb1);

        (dot, na, nb)
    };

    for i in reminder_range {
        dot = a[i].mul_add(b[i], dot);
        na = a[i].mul_add(a[i], na);
        nb = b[i].mul_add(b[i], nb);
    }

    let demon = (na * nb).sqrt();
    if demon < 1e-12 { 0.0 } else { dot / demon }
}

#[inline(always)]
/// SIMD-optimized cosine similarity for i8 vectors, using `AVX2` intrinsics for `x86_64` arch.
///
/// All elements must be in `[-127, 127]`. A value of `-128` produces incorrect results
/// because `maddubs_epi16` treats its second argument as signed i8, and `abs(-128)` wraps
/// to `-128` instead of `+128`.
pub fn cosine_similarity_i8(a: &[i8], b: &[i8]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector lengths do not match");

    const LANE: usize = 64;
    let simd_range = 0..(a.len() / LANE);
    let reminder_range = (simd_range.end * LANE)..a.len();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let (mut dot, mut na, mut nb) = unsafe {
        let mut dot_0 = _mm256_setzero_si256();
        let mut dot_1 = _mm256_setzero_si256();

        let mut na_0 = _mm256_setzero_si256();
        let mut na_1 = _mm256_setzero_si256();

        let mut nb_0 = _mm256_setzero_si256();
        let mut nb_1 = _mm256_setzero_si256();

        for i in simd_range {
            let off = i * LANE;
            // load first chunk
            let va0 = _mm256_loadu_si256(a_ptr.add(off) as *const __m256i);
            let vb0 = _mm256_loadu_si256(b_ptr.add(off) as *const __m256i);

            let va_vb = i8x32_dot_i32x8!(va0, vb0);

            dot_0 = _mm256_add_epi32(dot_0, va_vb);
            let abs_va0 = _mm256_abs_epi8(va0);
            let abs_vb0 = _mm256_abs_epi8(vb0);
            na_0 = _mm256_add_epi32(
                na_0,
                _mm256_madd_epi16(_mm256_maddubs_epi16(abs_va0, abs_va0), _mm256_set1_epi16(1)),
            );
            nb_0 = _mm256_add_epi32(
                nb_0,
                _mm256_madd_epi16(_mm256_maddubs_epi16(abs_vb0, abs_vb0), _mm256_set1_epi16(1)),
            );

            // load second chunk
            let va1 = _mm256_loadu_si256(a_ptr.add(off + 32) as *const __m256i);
            let vb1 = _mm256_loadu_si256(b_ptr.add(off + 32) as *const __m256i);

            let va_vb = i8x32_dot_i32x8!(va1, vb1);

            dot_1 = _mm256_add_epi32(dot_1, va_vb);
            let abs_va1 = _mm256_abs_epi8(va1);
            let abs_vb1 = _mm256_abs_epi8(vb1);
            na_1 = _mm256_add_epi32(
                na_1,
                _mm256_madd_epi16(_mm256_maddubs_epi16(abs_va1, abs_va1), _mm256_set1_epi16(1)),
            );
            nb_1 = _mm256_add_epi32(
                nb_1,
                _mm256_madd_epi16(_mm256_maddubs_epi16(abs_vb1, abs_vb1), _mm256_set1_epi16(1)),
            );
        }

        // hsum dot
        let dot = hsum_i32x8!(dot_0) + hsum_i32x8!(dot_1);
        // hsum na
        let na = hsum_i32x8!(na_0) + hsum_i32x8!(na_1);
        // hsum nb
        let nb = hsum_i32x8!(nb_0) + hsum_i32x8!(nb_1);

        (dot as f32, na as f32, nb as f32)
    };

    for i in reminder_range {
        let ai = a[i] as f32;
        let bi = b[i] as f32;
        dot = ai.mul_add(bi, dot);
        na = ai.mul_add(ai, na);
        nb = bi.mul_add(bi, nb);
    }

    let denom = (na * nb).sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

#[inline(always)]
/// SIMD-optimized raw Dot product for f32 vectors, using `AVX2` intrinsics for `x86_64` arch
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector lengths do not match");

    const LANE: usize = 32;
    let simd_range = 0..(a.len() / LANE);
    let reminder_range = (simd_range.end * LANE)..a.len();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut total = unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        for i in simd_range {
            let off = i * LANE;
            let va0 = _mm256_loadu_ps(a_ptr.add(off));
            let va1 = _mm256_loadu_ps(a_ptr.add(off + 8));
            let va2 = _mm256_loadu_ps(a_ptr.add(off + 16));
            let va3 = _mm256_loadu_ps(a_ptr.add(off + 24));

            let vb0 = _mm256_loadu_ps(b_ptr.add(off));
            let vb1 = _mm256_loadu_ps(b_ptr.add(off + 8));
            let vb2 = _mm256_loadu_ps(b_ptr.add(off + 16));
            let vb3 = _mm256_loadu_ps(b_ptr.add(off + 24));

            acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
            acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
            acc2 = _mm256_fmadd_ps(va2, vb2, acc2);
            acc3 = _mm256_fmadd_ps(va3, vb3, acc3);
        }
        hsum_f32x8!(acc0) + hsum_f32x8!(acc1) + hsum_f32x8!(acc2) + hsum_f32x8!(acc3)
    };

    for i in reminder_range {
        total = a[i].mul_add(b[i], total);
    }

    total
}

#[inline(always)]
/// SIMD-optimized raw Dot product for i8 vectors, using `AVX2` intrinsics for `x86_64` arch.
///
/// All elements must be in `[-127, 127]`. A value of `-128` produces incorrect results
/// because `sign_epi8(-128, -128)` yields `-128`, and `abs(-128)` wraps to `-128`,
/// causing `maddubs_epi16` to compute `128 × (-128) = -16384` instead of `+16384`.
pub fn dot_product_i8(a: &[i8], b: &[i8]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector lengths do not match");

    const LANE: usize = 128;
    let simd_range = 0..(a.len() / LANE);
    let reminder_range = (simd_range.end * LANE)..a.len();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut total = unsafe {
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();

        for i in simd_range {
            let off = i * LANE;
            let va0 = _mm256_loadu_si256(a_ptr.add(off) as *const __m256i);
            let va1 = _mm256_loadu_si256(a_ptr.add(off + 32) as *const __m256i);
            let va2 = _mm256_loadu_si256(a_ptr.add(off + 64) as *const __m256i);
            let va3 = _mm256_loadu_si256(a_ptr.add(off + 96) as *const __m256i);

            let vb0 = _mm256_loadu_si256(b_ptr.add(off) as *const __m256i);
            let vb1 = _mm256_loadu_si256(b_ptr.add(off + 32) as *const __m256i);
            let vb2 = _mm256_loadu_si256(b_ptr.add(off + 64) as *const __m256i);
            let vb3 = _mm256_loadu_si256(b_ptr.add(off + 96) as *const __m256i);

            acc0 = _mm256_add_epi32(acc0, i8x32_dot_i32x8!(va0, vb0));
            acc1 = _mm256_add_epi32(acc1, i8x32_dot_i32x8!(va1, vb1));
            acc2 = _mm256_add_epi32(acc2, i8x32_dot_i32x8!(va2, vb2));
            acc3 = _mm256_add_epi32(acc3, i8x32_dot_i32x8!(va3, vb3));
        }
        (hsum_i32x8!(acc0) + hsum_i32x8!(acc1) + hsum_i32x8!(acc2) + hsum_i32x8!(acc3)) as f32
    };

    for i in reminder_range {
        let ai = a[i] as f32;
        let bi = b[i] as f32;
        total = ai.mul_add(bi, total);
    }
    total
}

#[inline(always)]
/// SIMD-optimized Euclidean similarity for f32 vectors, using `AVX2` intrinsics for `x86_64` arch
pub fn euclidean_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector lengths do not match");

    const LANE: usize = 32;
    let simd_range = 0..(a.len() / LANE);
    let reminder_range = (simd_range.end * LANE)..a.len();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut dist = unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        for i in simd_range {
            let off = i * LANE;

            let va0 = _mm256_loadu_ps(a_ptr.add(off));
            let va1 = _mm256_loadu_ps(a_ptr.add(off + 8));
            let va2 = _mm256_loadu_ps(a_ptr.add(off + 16));
            let va3 = _mm256_loadu_ps(a_ptr.add(off + 24));

            let vb0 = _mm256_loadu_ps(b_ptr.add(off));
            let vb1 = _mm256_loadu_ps(b_ptr.add(off + 8));
            let vb2 = _mm256_loadu_ps(b_ptr.add(off + 16));
            let vb3 = _mm256_loadu_ps(b_ptr.add(off + 24));

            let diff0 = _mm256_sub_ps(va0, vb0);
            let diff1 = _mm256_sub_ps(va1, vb1);
            let diff2 = _mm256_sub_ps(va2, vb2);
            let diff3 = _mm256_sub_ps(va3, vb3);

            acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);
            acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
            acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);
            acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);
        }
        hsum_f32x8!(acc0) + hsum_f32x8!(acc1) + hsum_f32x8!(acc2) + hsum_f32x8!(acc3)
    };

    for i in reminder_range {
        let d = a[i].algebraic_sub(b[i]);
        dist = d.mul_add(d, dist);
    }

    1.0 / (1.0 + dist.sqrt())
}

#[inline(always)]
/// SIMD-optimized Euclidean similarity for i8 vectors, using `AVX2` intrinsics for `x86_64` arch.
///
/// All elements must be in `[-127, 127]`. A value of `-128` produces incorrect results
/// because `maddubs_epi16` treats its second argument as signed i8, and `abs(-128)` wraps
/// to `-128` instead of `+128`.
pub fn euclidean_similarity_i8(a: &[i8], b: &[i8]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector lengths do not match");

    const LANE: usize = 128;
    let simd_range = 0..(a.len() / LANE);
    let reminder_range = (simd_range.end * LANE)..a.len();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut dist = unsafe {
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();

        for i in simd_range {
            let off = i * LANE;

            let va0 = _mm256_loadu_si256(a_ptr.add(off) as *const __m256i);
            let va1 = _mm256_loadu_si256(a_ptr.add(off + 32) as *const __m256i);
            let va2 = _mm256_loadu_si256(a_ptr.add(off + 64) as *const __m256i);
            let va3 = _mm256_loadu_si256(a_ptr.add(off + 96) as *const __m256i);

            let vb0 = _mm256_loadu_si256(b_ptr.add(off) as *const __m256i);
            let vb1 = _mm256_loadu_si256(b_ptr.add(off + 32) as *const __m256i);
            let vb2 = _mm256_loadu_si256(b_ptr.add(off + 64) as *const __m256i);
            let vb3 = _mm256_loadu_si256(b_ptr.add(off + 96) as *const __m256i);

            let diff0 = _mm256_sub_epi8(va0, vb0);
            let diff1 = _mm256_sub_epi8(va1, vb1);
            let diff2 = _mm256_sub_epi8(va2, vb2);
            let diff3 = _mm256_sub_epi8(va3, vb3);

            let abs_diff0 = _mm256_abs_epi8(diff0);
            let abs_diff1 = _mm256_abs_epi8(diff1);
            let abs_diff2 = _mm256_abs_epi8(diff2);
            let abs_diff3 = _mm256_abs_epi8(diff3);

            let sq0 = _mm256_maddubs_epi16(abs_diff0, abs_diff0);
            let sq1 = _mm256_maddubs_epi16(abs_diff1, abs_diff1);
            let sq2 = _mm256_maddubs_epi16(abs_diff2, abs_diff2);
            let sq3 = _mm256_maddubs_epi16(abs_diff3, abs_diff3);

            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(sq0, _mm256_set1_epi16(1)));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(sq1, _mm256_set1_epi16(1)));
            acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(sq2, _mm256_set1_epi16(1)));
            acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(sq3, _mm256_set1_epi16(1)));
        }
        (hsum_i32x8!(acc0) + hsum_i32x8!(acc1) + hsum_i32x8!(acc2) + hsum_i32x8!(acc3)) as f32
    };

    for i in reminder_range {
        let d = (a[i] as f32).algebraic_sub(b[i] as f32);
        dist = d.mul_add(d, dist);
    }

    1.0 / (1.0 + dist.sqrt())
}

#[inline(always)]
/// SIMD-optimized L1 normalization for f32 vectors, using `AVX2` intrinsics for `x86_64` arch,
/// `v` is modified in place
pub fn normalize_l1(v: &mut [f32]) {
    const LANES: usize = 32;
    let norm: f32 = v.iter().map(|x| x.abs()).sum(); // TODO; bet on llvm

    if norm > 1e-12 {
        let inv = 1.0 / norm;
        let inv_v = unsafe { _mm256_set1_ps(inv) };
        let ptr = v.as_mut_ptr();
        let chunks = v.len() / LANES;

        unsafe {
            for i in 0..chunks {
                let offset = i * LANES;
                let p = ptr.add(offset);

                let v0 = _mm256_loadu_ps(p);
                let v1 = _mm256_loadu_ps(p.add(8));
                let v2 = _mm256_loadu_ps(p.add(16));
                let v3 = _mm256_loadu_ps(p.add(24));

                _mm256_storeu_ps(p, _mm256_mul_ps(v0, inv_v));
                _mm256_storeu_ps(p.add(8), _mm256_mul_ps(v1, inv_v));
                _mm256_storeu_ps(p.add(16), _mm256_mul_ps(v2, inv_v));
                _mm256_storeu_ps(p.add(24), _mm256_mul_ps(v3, inv_v));
            }

            let remainder = chunks * LANES;
            for i in remainder..v.len() {
                *v.get_unchecked_mut(i) *= inv;
            }
        }
    }
}

#[inline(always)]
/// SIMD-optimized L2 normalization for f32 vectors, using `AVX2` intrinsics for `x86_64` arch,
/// `v` is modified in place
pub fn normalize_l2(v: &mut [f32]) {
    const LANES: usize = 32;

    let dot_prod_norm = dot_product(v, v);
    if dot_prod_norm > 1e-12 {
        let inv = 1.0 / dot_prod_norm.sqrt();
        let inv_v = unsafe { _mm256_set1_ps(inv) };
        let ptr = v.as_mut_ptr();
        let chunks = v.len() / LANES;

        unsafe {
            for i in 0..chunks {
                let offset = i * LANES;
                let p = ptr.add(offset);

                let v0 = _mm256_loadu_ps(p);
                let v1 = _mm256_loadu_ps(p.add(8));
                let v2 = _mm256_loadu_ps(p.add(16));
                let v3 = _mm256_loadu_ps(p.add(24));

                _mm256_storeu_ps(p, _mm256_mul_ps(v0, inv_v));
                _mm256_storeu_ps(p.add(8), _mm256_mul_ps(v1, inv_v));
                _mm256_storeu_ps(p.add(16), _mm256_mul_ps(v2, inv_v));
                _mm256_storeu_ps(p.add(24), _mm256_mul_ps(v3, inv_v));
            }

            let remainder = chunks * LANES;
            for i in remainder..v.len() {
                *v.get_unchecked_mut(i) *= inv;
            }
        }
    }
}

#[inline]
/// SIMD-optimized matrix multiplication for f32 matrices, using `AVX2` intrinsics for `x86_64` arch,
/// returns a new matrix, `matrix_a` must be of size `rows_a x cols_a`, `matrix_b` must be of size `rows_b x cols_b`, and `cols_a` must equal `rows_b`
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
    unsafe {
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
    }
    result
}

#[inline]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::missing_safety_doc)]
/// SIMD-optimized in-place matrix (row-major) multiplication for f32 matrices, using `AVX2` intrinsics for `x86_64` arch,
/// `result` must be preallocated to correct size and matrics size must be valid for multiplication or else UB
pub unsafe fn matmul_into(
    matrix_a: &[f32],
    matrix_b: &[f32],
    rows_a: usize,
    cols_a: usize,
    rows_b: usize,
    cols_b: usize,
    matrix_b_t: &mut [f32],
    result: &mut [f32],
) {
    // assert_eq!(cols_a, rows_b);
    // assert_eq!(matrix_a.len(), rows_a * cols_a);
    // assert_eq!(matrix_b.len(), rows_b * cols_b);
    // assert_eq!(matrix_b_t.len(), rows_b * cols_b);
    // assert_eq!(result.len(), rows_a * cols_b);

    unsafe { transpose_mat_into(rows_b, cols_b, matrix_b, matrix_b_t) };
    result.fill(0.0f32); // fast memcpy

    const BLOCK_A: usize = 256;
    const BLOCK_B: usize = 128;
    const BLOCK_C: usize = 96;
    const LANE: usize = 8;

    let b_ptr = matrix_b_t.as_ptr();

    unsafe {
        for i in (0..rows_a).step_by(BLOCK_A) {
            let i_max = (i + BLOCK_A).min(rows_a);

            for j in (0..cols_b).step_by(BLOCK_B) {
                let j_max = (j + BLOCK_B).min(cols_b);

                for k in (0..cols_a).step_by(BLOCK_C) {
                    let k_max = (k + BLOCK_C).min(cols_a);

                    for row in i..i_max {
                        let row_base = row * cols_b;
                        let a_row = matrix_a.get_unchecked(row * cols_a..(row + 1) * cols_a);
                        let a_ptr = a_row.as_ptr();
                        let col_limit = j_max - ((j_max - j) % 4);

                        for col in (j..col_limit).step_by(4) {
                            let base = row_base + col;
                            let b0_strt = col * cols_a;
                            let b1_strt = (col + 1) * cols_a;
                            let b2_strt = (col + 2) * cols_a;
                            let b3_strt = (col + 3) * cols_a;

                            let mut vsum0 = _mm256_setzero_ps();
                            let mut vsum1 = _mm256_setzero_ps();
                            let mut vsum2 = _mm256_setzero_ps();
                            let mut vsum3 = _mm256_setzero_ps();

                            let mut t = k;
                            while t + LANE <= k_max {
                                let va = _mm256_loadu_ps(a_ptr.add(t));
                                let b0 = _mm256_loadu_ps(b_ptr.add(b0_strt + t));
                                let b1 = _mm256_loadu_ps(b_ptr.add(b1_strt + t));
                                let b2 = _mm256_loadu_ps(b_ptr.add(b2_strt + t));
                                let b3 = _mm256_loadu_ps(b_ptr.add(b3_strt + t));

                                vsum0 = _mm256_fmadd_ps(va, b0, vsum0);
                                vsum1 = _mm256_fmadd_ps(va, b1, vsum1);
                                vsum2 = _mm256_fmadd_ps(va, b2, vsum2);
                                vsum3 = _mm256_fmadd_ps(va, b3, vsum3);
                                t += LANE;
                            }

                            let mut sum0 = hsum_f32x8!(vsum0);
                            let mut sum1 = hsum_f32x8!(vsum1);
                            let mut sum2 = hsum_f32x8!(vsum2);
                            let mut sum3 = hsum_f32x8!(vsum3);

                            while t < k_max {
                                let a_val = *a_ptr.add(t);
                                sum0 += a_val * *b_ptr.add(b0_strt + t);
                                sum1 += a_val * *b_ptr.add(b1_strt + t);
                                sum2 += a_val * *b_ptr.add(b2_strt + t);
                                sum3 += a_val * *b_ptr.add(b3_strt + t);
                                t += 1;
                            }

                            *result.get_unchecked_mut(base) += sum0;
                            *result.get_unchecked_mut(base + 1) += sum1;
                            *result.get_unchecked_mut(base + 2) += sum2;
                            *result.get_unchecked_mut(base + 3) += sum3;
                        }

                        for col in col_limit..j_max {
                            let base = row_base + col;
                            let b_base = col * cols_a;

                            let mut vsum = _mm256_setzero_ps();
                            let mut t = k;

                            while t + LANE <= k_max {
                                let va = _mm256_loadu_ps(a_ptr.add(t));
                                let vb = _mm256_loadu_ps(b_ptr.add(b_base + t));
                                vsum = _mm256_fmadd_ps(va, vb, vsum);
                                t += LANE;
                            }

                            let mut sum = hsum_f32x8!(vsum);
                            while t < k_max {
                                sum += *a_ptr.add(t) * *b_ptr.add(b_base + t);
                                t += 1;
                            }

                            *result.get_unchecked_mut(base) += sum;
                        }
                    }
                }
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::missing_safety_doc)]
/// SIMD-optimized transpose a row-major matrix using `AVX2` intrinsics for `x86_64` arch,
/// `output` must be preallocated to correct size or else UB,
pub unsafe fn transpose_mat_into(rows: usize, cols: usize, input: &[f32], output: &mut [f32]) {
    const TILE: usize = 128;
    const LANE: usize = 8;

    // process in size of TILE x TILE
    for ii in (0..rows).step_by(TILE) {
        for jj in (0..cols).step_by(TILE) {
            let i_max = (ii + TILE).min(rows);
            let j_max = (jj + TILE).min(cols);

            unsafe {
                for i in (ii..i_max).step_by(LANE) {
                    for j in (jj..j_max).step_by(LANE) {
                        let i_end = (i + LANE).min(i_max);
                        let j_end = (j + LANE).min(j_max);

                        if i_end - i == LANE && j_end - j == LANE {
                            // 8x8 micro transpose
                            let src = input.as_ptr().add(i * cols + j);
                            let dst = output.as_mut_ptr().add(j * rows + i);
                            transpose_8x8_avx2(src, cols, dst, rows);
                        } else {
                            // remaining
                            for x in i..i_end {
                                for y in j..j_end {
                                    *output.get_unchecked_mut(y * rows + x) =
                                        *input.get_unchecked(x * cols + y);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[inline(always)]
/// 8×8 in-register matrix transpose micro-kernel using `AVX2`.
unsafe fn transpose_8x8_avx2(src: *const f32, src_stride: usize, dst: *mut f32, dst_stride: usize) {
    unsafe {
        // load 8 source rows
        let r0 = _mm256_loadu_ps(src);
        let r1 = _mm256_loadu_ps(src.add(src_stride));
        let r2 = _mm256_loadu_ps(src.add(src_stride * 2));
        let r3 = _mm256_loadu_ps(src.add(src_stride * 3));
        let r4 = _mm256_loadu_ps(src.add(src_stride * 4));
        let r5 = _mm256_loadu_ps(src.add(src_stride * 5));
        let r6 = _mm256_loadu_ps(src.add(src_stride * 6));
        let r7 = _mm256_loadu_ps(src.add(src_stride * 7));

        // interleave adjacent pairs
        let t0 = _mm256_unpacklo_ps(r0, r1); // take first 4 elements of r0 and r1
        let t1 = _mm256_unpackhi_ps(r0, r1); // take last 4 elements of r0 and r1
        let t2 = _mm256_unpacklo_ps(r2, r3); // repeat...
        let t3 = _mm256_unpackhi_ps(r2, r3);
        let t4 = _mm256_unpacklo_ps(r4, r5);
        let t5 = _mm256_unpackhi_ps(r4, r5);
        let t6 = _mm256_unpacklo_ps(r6, r7);
        let t7 = _mm256_unpackhi_ps(r6, r7);

        // shuffle 64-bit pairs inside 128-bit lanes using control byte
        let u0 = _mm256_shuffle_ps(t0, t2, 0x44);
        let u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
        let u2 = _mm256_shuffle_ps(t1, t3, 0x44);
        let u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
        let u4 = _mm256_shuffle_ps(t4, t6, 0x44);
        let u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
        let u6 = _mm256_shuffle_ps(t5, t7, 0x44);
        let u7 = _mm256_shuffle_ps(t5, t7, 0xEE);

        // 'cross' 128-bit lane permute
        let v0 = _mm256_permute2f128_ps(u0, u4, 0x20);
        let v1 = _mm256_permute2f128_ps(u1, u5, 0x20);
        let v2 = _mm256_permute2f128_ps(u2, u6, 0x20);
        let v3 = _mm256_permute2f128_ps(u3, u7, 0x20);
        let v4 = _mm256_permute2f128_ps(u0, u4, 0x31);
        let v5 = _mm256_permute2f128_ps(u1, u5, 0x31);
        let v6 = _mm256_permute2f128_ps(u2, u6, 0x31);
        let v7 = _mm256_permute2f128_ps(u3, u7, 0x31);

        // store the 8 transposed rows
        _mm256_storeu_ps(dst, v0);
        _mm256_storeu_ps(dst.add(dst_stride), v1);
        _mm256_storeu_ps(dst.add(dst_stride * 2), v2);
        _mm256_storeu_ps(dst.add(dst_stride * 3), v3);
        _mm256_storeu_ps(dst.add(dst_stride * 4), v4);
        _mm256_storeu_ps(dst.add(dst_stride * 5), v5);
        _mm256_storeu_ps(dst.add(dst_stride * 6), v6);
        _mm256_storeu_ps(dst.add(dst_stride * 7), v7);
    }
}
