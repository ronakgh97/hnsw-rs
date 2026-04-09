use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

/// Generates a random vector of given dimension with values in range `[-1.0, 1.0]`, still bad for similarity test due to high [dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality), but useful for benchmarking and testing
/// `base_seed` is used to ensure reproducibility across runs. Each vector will have a different seed derived from `base_seed` to ensure different random values.
/// > `parallel` can be significant for large `num` and `dimensions`, but for small sizes, the sequential version may be faster for better gains.
#[inline]
pub fn generate_random_vectors(
    num: usize,
    dimensions: usize,
    base_seed: u64,
    parallel: bool,
) -> Vec<Vec<f32>> {
    if parallel {
        let mut result = vec![vec![0.0f32; dimensions]; num];

        result.par_iter_mut().enumerate().for_each(|(i, v)| {
            let mut rng = SmallRng::seed_from_u64(base_seed.wrapping_add(i as u64));
            for x in v {
                *x = rng.random_range(-1.0..1.0);
            }
        });

        return result;
    }

    let mut result = Vec::with_capacity(num);
    for i in 0..num {
        let mut rng = SmallRng::seed_from_u64(base_seed.wrapping_add(i as u64));
        let mut v = vec![0.0f32; dimensions];
        for i in &mut v {
            *i = rng.random_range(-1.0..1.0);
        }
        result.push(v);
    }
    result
}

/// Generates a random byte vector of given size, useful for testing with binary data or metadata.
/// Each call produces different random bytes. (no seed)
#[inline]
pub fn get_random_bytes(size: u32) -> Vec<u8> {
    let mut rng = rand::rng();
    (0..size).map(|_| rng.random::<u8>()).collect()
}

#[test]
fn test_seed_generation() {
    let num_vectors = 1000;
    let dimensions = 128;
    let base_seed = 42;

    let vectors_sequential = generate_random_vectors(num_vectors, dimensions, base_seed, false);
    let vectors_parallel = generate_random_vectors(num_vectors, dimensions, base_seed, true);

    assert_eq!(vectors_parallel.len(), num_vectors);
    assert_eq!(vectors_sequential.len(), num_vectors);

    for (v1, v2) in vectors_sequential.iter().zip(vectors_parallel.iter()) {
        assert_eq!(v1.len(), dimensions);
        assert_eq!(v2.len(), dimensions);
        assert_eq!(v1, v2); // Both methods should produce the same vectors with the same seed
    }
}

#[test]
fn test_parallel() {
    let num_vectors = 1024 * 1024;
    let dimensions = 128;
    let base_seed = 49;

    let time_seq = std::time::Instant::now();
    let vectors_sequential = generate_random_vectors(num_vectors, dimensions, base_seed, false);
    let elapsed_seq = time_seq.elapsed();

    let time_parallel = std::time::Instant::now();
    let vectors_parallel = generate_random_vectors(num_vectors, dimensions, base_seed, true);
    let elapsed_parallel = time_parallel.elapsed();

    assert_eq!(vectors_parallel.len(), num_vectors);
    assert_eq!(vectors_sequential.len(), num_vectors);
    assert_eq!(vectors_sequential, vectors_parallel);

    assert!(elapsed_seq > elapsed_parallel);
    println!(
        "Generated {} vectors of dimension {} in {:?} (sequential) vs {:?} (parallel)",
        num_vectors, dimensions, elapsed_seq, elapsed_parallel
    );
}
