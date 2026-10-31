use rayon::prelude::*;

// /// Generates a random vector of given dimension with values in range `(-1.0, 1.0)`,
// /// still bad for similarity test due to high [dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality).
// ///
// /// Returns a tuple of (newly generated vectors, final seed).
// #[inline]
// pub fn gen_vec(num: usize, dim: usize, base_seed: usize) -> (Vec<Vec<f32>>, usize) {
//     let result: Vec<Vec<f32>> = (0..num)
//         .map(|i| {
//             let mut rng = fastrand::Rng::with_seed((base_seed + i) as u64);
//             (0..dim).map(|_| rng.f32_inclusive() * 2.0 - 1.0).collect()
//         })
//         .collect();
//
//     (result, base_seed + num)
// }

/// Random sample vector generator
pub struct VectorGen {
    vector_dim: usize,
    flat_vectors: Vec<f32>,
}

impl VectorGen {
    /// Generate & store f32 vectors in flat layout
    /// [`Curse of dimensionality`](https://en.wikipedia.org/wiki/Curse_of_dimensionality) is still a problem, but this is a good start for testing
    pub fn generate(num: usize, vector_dim: usize, seed: Option<u64>) -> Self {
        let seed = if let Some(seed) = seed {
            seed
        } else {
            fastrand::u64(0..=u64::MAX)
        };

        let mut flat_vectors = vec![0.0f32; num * vector_dim];
        flat_vectors.par_iter_mut().enumerate().for_each(|(i, x)| {
            let mut rng = fastrand::Rng::with_seed(seed + i as u64);
            *x = rng.f32_inclusive() * 2.0 - 1.0;
        });

        Self {
            flat_vectors,
            vector_dim,
        }
    }

    #[inline(always)]
    /// Returns a vector at given index as an immutable slice
    pub fn get(&self, index: usize) -> &[f32] {
        let start = index * self.vector_dim;
        assert!(
            start + self.vector_dim <= self.flat_vectors.len(),
            "VectorGen: index {} out of bounds",
            index
        );
        unsafe {
            self.flat_vectors
                .get_unchecked(start..start + self.vector_dim)
        }
    }

    #[inline(always)]
    /// Returns a vector at given index as a mutable slice
    pub fn get_mut(&mut self, index: usize) -> &mut [f32] {
        let start = index * self.vector_dim;
        assert!(
            start + self.vector_dim <= self.flat_vectors.len(),
            "VectorGen: index {} out of bounds",
            index
        );
        unsafe {
            self.flat_vectors
                .get_unchecked_mut(start..start + self.vector_dim)
        }
    }
}

#[test]
fn test_seed_generation() {
    let num_vectors = 2048;
    let dimensions = 1024;
    let base_seed = 97;

    let gen_1 = VectorGen::generate(num_vectors, dimensions, Some(base_seed));
    let gen_2 = VectorGen::generate(num_vectors, dimensions, Some(base_seed));

    for i in 0..num_vectors {
        assert_eq!(gen_1.get(i), gen_2.get(i));
    }
}
