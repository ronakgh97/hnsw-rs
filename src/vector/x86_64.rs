use std::arch::x86_64::*;

/// Horizontal sum of i32x8 `__m256i` vector, returning a single i32 value, **can overflow!!!**
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

/// Multiply two i8x32 `__m256i` vectors (preserve sign) and return it as packed i32x8 `__m256i` vector
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

/// Horizontal sum of f32x8 `__m256d` vector, returning a single f32 value, **can overflow!**
macro_rules! hsum_f32x8 {
    ($v:expr) => {
        {
            // first pass; [a  b  c  d | e  f  g  h]
            let hi = _mm256_permute2f128_ps($v, $v, 1); // [e  f  g  h | a  b  c  d]
            let sum = _mm256_add_ps($v, hi); // [a+e  b+f  c+g  d+h | e+a  f+b  g+c  h+d]
            // [x0 x1 x2 x3] -> [x2 x3 x0 x1]
            let hi = _mm256_permute_ps(sum, 0b01001110); // [c+g  d+h  a+e  b+f | g+c  h+d  e+a  f+b]

            // second pass to add the two halves together
            let sum = _mm256_add_ps(sum, hi); // [a+e+b+f  c+g+d+h  e+a+f+b  g+c+h+d]
            let hi = _mm256_permute_ps(sum, 0b10110001); // [c+g+d+h  a+e+b+f  g+c+h+d  e+a+f+b]
            let sum = _mm256_add_ps(sum, hi); // [a+e+b+f+c+g+d+h  a+e+b+f+c+g+d+h  e+a+f+b+g+c+h+d  e+a+f+b+g+c+h+d]
            _mm256_cvtss_f32(sum) // extract first

        }
    };
}

#[inline(always)]
/// SIMD-optimized cosine similarity for f32 vectors, using `AVX2` intrinsics for `x86_64` arch
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "a: {}!= b: {}", a.len(), b.len());

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
            let va3 = _mm256_loadu_ps(a_ptr.add(offset + 32));

            let vb0 = _mm256_loadu_ps(b_ptr.add(offset));
            let vb1 = _mm256_loadu_ps(b_ptr.add(offset + 8));
            let vb2 = _mm256_loadu_ps(b_ptr.add(offset + 16));
            let vb3 = _mm256_loadu_ps(b_ptr.add(offset + 32));

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
/// SIMD-optimized cosine similarity for i8 vectors, using `AVX2` intrinsics for `x86_64` arch
pub fn cosine_similarity_i8(a: &[i8], b: &[i8]) -> f32 {
    assert_eq!(a.len(), b.len(), "a: {}!= b: {}", a.len(), b.len());

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
            na_0 = _mm256_add_epi32(
                na_0,
                _mm256_madd_epi16(_mm256_maddubs_epi16(va0, va0), _mm256_set1_epi16(1)),
            );
            nb_0 = _mm256_add_epi32(
                nb_0,
                _mm256_madd_epi16(_mm256_maddubs_epi16(vb0, vb0), _mm256_set1_epi16(1)),
            );

            // load second chunk
            let va1 = _mm256_loadu_si256(a_ptr.add(off + 32) as *const __m256i);
            let vb1 = _mm256_loadu_si256(b_ptr.add(off + 32) as *const __m256i);

            let va_vb = i8x32_dot_i32x8!(va1, vb1);

            dot_1 = _mm256_add_epi32(dot_1, va_vb);
            na_1 = _mm256_add_epi32(
                na_1,
                _mm256_madd_epi16(_mm256_maddubs_epi16(va1, va1), _mm256_set1_epi16(1)),
            );
            nb_1 = _mm256_add_epi32(
                nb_1,
                _mm256_madd_epi16(_mm256_maddubs_epi16(vb1, vb1), _mm256_set1_epi16(1)),
            );
        }

        // hsum dot
        let dot = hsum_i32x8!(dot_0) + hsum_i32x8!(dot_1);
        // hsum na
        let na = hsum_i32x8!(na_0) + hsum_i32x8!(na_1);
        // hsum nb
        let nb = hsum_i32x8!(nb_0) + hsum_i32x8!(nb_1);

        (dot, na, nb)
    };

    for i in reminder_range {
        let ai = a[i] as i32;
        let bi = b[i] as i32;
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }

    let denom = (na as f32 * nb as f32).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        dot as f32 / denom
    }
}

#[inline(always)]
/// SIMD-optimized raw Dot product for f32 vectors, using `AVX2` intrinsics for `x86_64` arch
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "a: {}!= b: {}", a.len(), b.len());

    const LANE: usize = 64;
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
        let ai = a[i];
        let bi = b[i];
        total = ai.mul_add(bi, total);
    }

    total
}

#[inline(always)]
/// SIMD-optimized raw Dot product for i8 vectors, using `AVX2` intrinsics for `x86_64` arch
pub fn dot_product_i8(a: &[i8], b: &[i8]) -> f32 {
    assert_eq!(a.len(), b.len(), "a: {}!= b: {}", a.len(), b.len());

    const LANE: usize = 64;
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
        hsum_i32x8!(acc0) + hsum_i32x8!(acc1) + hsum_i32x8!(acc2) + hsum_i32x8!(acc3)
    };

    for i in reminder_range {
        total += a[i] as i32 * b[i] as i32;
    }
    total as f32
}

pub fn euclidean_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "a: {}!= b: {}", a.len(), b.len());

    const LANE: usize = 64;
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
            let vb1 = _mm256_loadu_ps(b_ptr.add(off + 32));
            let vb2 = _mm256_loadu_ps(b_ptr.add(off + 64));
            let vb3 = _mm256_loadu_ps(b_ptr.add(off + 96));

            let diff0 = _mm256_sub_ps(va0, vb0);
            let diff1 = _mm256_sub_ps(va1, vb1);
            let diff2 = _mm256_sub_ps(va2, vb2);
            let diff3 = _mm256_sub_ps(va3, vb3);

            let sq0 = _mm256_mul_ps(diff0, diff1);
            let sq1 = _mm256_mul_ps(diff2, diff3);
            let sq2 = _mm256_mul_ps(diff0, diff2);
            let sq3 = _mm256_mul_ps(diff0, diff3);

            acc0 = _mm256_add_ps(acc0, sq0);
            acc1 = _mm256_add_ps(acc1, sq1);
            acc2 = _mm256_add_ps(acc2, sq2);
            acc3 = _mm256_add_ps(acc3, sq3);
        }
        hsum_f32x8!(acc0) + hsum_f32x8!(acc1) + hsum_f32x8!(acc2) + hsum_f32x8!(acc3)
    };

    for i in reminder_range {
        let d = a[i] - b[i];
        dist = d.mul_add(d, dist);
    }

    1.0 / (1.0 + dist).sqrt()
}

#[inline(always)]
/// SIMD-optimized Euclidean similarity for i8 vectors, using `AVX2` intrinsics for `x86_64` arch
pub fn euclidean_similarity_i8(a: &[i8], b: &[i8]) -> f32 {
    assert_eq!(a.len(), b.len(), "a: {}!= b: {}", a.len(), b.len());

    const LANE: usize = 64;
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

            let sq0 = _mm256_maddubs_epi16(diff0, diff0);
            let sq1 = _mm256_maddubs_epi16(diff1, diff1);
            let sq2 = _mm256_maddubs_epi16(diff2, diff2);
            let sq3 = _mm256_maddubs_epi16(diff3, diff3);

            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(sq0, _mm256_set1_epi16(1)));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(sq1, _mm256_set1_epi16(1)));
            acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(sq2, _mm256_set1_epi16(1)));
            acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(sq3, _mm256_set1_epi16(1)));
        }
        hsum_i32x8!(acc0) + hsum_i32x8!(acc1) + hsum_i32x8!(acc2) + hsum_i32x8!(acc3)
    };

    for i in reminder_range {
        let d = a[i] as i32 - b[i] as i32;
        dist += d * d;
    }

    1.0 / (1.0 + (dist as f32).sqrt())
}
