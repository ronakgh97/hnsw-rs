/// Generates a random vector of given dimension with values in range `(-1.0, 1.0)`,
/// still bad for similarity test due to high [dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality).
///
/// Returns a tuple of (newly generated vectors, final seed).
#[inline]
pub fn gen_vec(num: usize, dim: usize, base_seed: usize) -> (Vec<Vec<f32>>, usize) {
    let result: Vec<Vec<f32>> = (0..num)
        .map(|i| {
            let mut rng = fastrand::Rng::with_seed((base_seed + i) as u64);
            (0..dim).map(|_| rng.f32_inclusive() * 2.0 - 1.0).collect()
        })
        .collect();

    (result, base_seed + num)
}

// UNSAFE!!!
// #[inline(always)]
// /// Utility function to convert a slice of any type into a byte slice
// pub fn to_bytes<T>(data: &[T]) -> &[u8] {
//     unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, size_of_val(data)) }
// }
//
// #[inline(always)]
// /// Utility function to convert a byte slice back into a reference of any type
// pub fn from_bytes<T: Copy>(data: &[u8]) -> T {
//     unsafe { *(data.as_ptr() as *const T) }
// }

#[test]
fn test_seed_generation() {
    let num_vectors = 2048;
    let dimensions = 128;
    let base_seed = 111;

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
