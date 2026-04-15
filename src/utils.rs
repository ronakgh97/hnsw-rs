use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

/// Generates a random vector of given dimension with values in range `[-1.0, 1.0]`,
/// Still bad for similarity test due to high [dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality), but useful testing
/// Returns a tuple of (generated vectors, final seed).
#[inline]
pub fn gen_vec(num: usize, dim: usize, base_seed: u64) -> (Vec<Vec<f32>>, u64) {
    let mut result = vec![vec![0.0f32; dim]; num];

    result.par_iter_mut().enumerate().for_each(|(i, v)| {
        let mut rng = SmallRng::seed_from_u64(base_seed.wrapping_add(i as u64));
        for x in v {
            *x = rng.random_range(-1.0..1.0);
        }
    });

    let final_seed = base_seed.wrapping_add(num as u64);

    (result, final_seed)
}

/// Generates a random byte vector of given size, useful for testing with binary data or metadata.
/// Each call produces different random bytes. (no seed)
#[inline]
pub fn get_random_bytes(size: u32) -> Vec<u8> {
    let mut rng = rand::rng();
    (0..size).map(|_| rng.random::<u8>()).collect()
}

/// Fills an existing buffer with random values in range `[-1.0, 1.0]`
#[inline]
pub fn gen_fill(buf: &mut [f64]) {
    buf.par_iter_mut().for_each(|x| {
        *x = fastrand::f64() * 2.0 - 1.0;
    });
}

#[test]
fn test_seed_generation() {
    let num_vectors = 2048;
    let dimensions = 128;
    let base_seed = 42;

    let (gen_1, _seed) = gen_vec(num_vectors, dimensions, base_seed);
    let (gen_2, _seed) = gen_vec(num_vectors, dimensions, base_seed);

    assert_eq!(gen_2.len(), num_vectors);
    assert_eq!(gen_1.len(), num_vectors);

    for (v1, v2) in gen_1.iter().zip(gen_2.iter()) {
        assert_eq!(v1.len(), dimensions);
        assert_eq!(v2.len(), dimensions);
        assert_eq!(v1, v2); // Both methods should produce the same vectors with the same seed
    }
}
