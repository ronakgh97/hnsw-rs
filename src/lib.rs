mod hnsw;
mod maths;
mod utils;

pub mod prelude {
    pub use crate::hnsw::*;
    pub use crate::maths::*;
    pub use crate::utils::{generate_random_vectors, get_random_bytes};
}

#[test]
fn hnsw_test() {
    use crate::prelude::*;
    let mut hnsw = HNSW::default();

    let random_vec = generate_random_vectors(128, 1024, 97, false);

    for (i, vectors) in random_vec.iter().enumerate() {
        let level_asg = hnsw.get_random_level();
        let garbage = get_random_bytes(16);
        hnsw.insert(i.to_string(), vectors, garbage, level_asg)
            .unwrap();
    }
    assert_eq!(hnsw.nodes.len(), random_vec.len());
}
