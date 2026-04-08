use hnsw_rs::maths::{cosine_similarity, dot_product, euclidean_similarity};
use hnsw_rs::prelude::*;

#[test]
fn test_cosine_similarity_identical_vectors() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert!((sim - 1.0).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_orthogonal_vectors() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert!(sim.abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_opposite_vectors() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![-1.0, 0.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert!((sim - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_3d_vectors() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let sim = cosine_similarity(&a, &b);
    let expected = (1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0)
        / ((1.0_f32 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0).sqrt()
            * (4.0_f32 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0).sqrt());
    assert!((sim - expected).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_with_zeros() {
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![1.0, 2.0, 3.0];
    let sim = cosine_similarity(&a, &b);
    assert!(sim.abs() < 1e-6);
}

#[test]
fn test_euclidean_similarity_identical_vectors() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0, 3.0];
    let sim = euclidean_similarity(&a, &b);
    assert!((sim - 1.0).abs() < 1e-6);
}

#[test]
fn test_euclidean_similarity_different_vectors() {
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![3.0, 4.0, 0.0];
    let sim = euclidean_similarity(&a, &b);
    assert!((sim - (1.0 / 6.0)).abs() < 1e-6); // 1 / (1 + 5) = 1/6 ≈ 0.1667
}

#[test]
fn test_euclidean_similarity_far_vectors() {
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![10.0, 10.0, 10.0];
    let sim = euclidean_similarity(&a, &b);
    // 1 / (1 + sqrt(300)) ≈ 1 / 18.32 ≈ 0.0546
    assert!(sim > 0.05 && sim < 0.06);
}

#[test]
fn test_dot_product_basic() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let dot = dot_product(&a, &b);
    assert!((dot - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
}

#[test]
fn test_dot_product_orthogonal() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let dot = dot_product(&a, &b);
    assert!(dot.abs() < 1e-6);
}

#[test]
fn test_dot_product_negative() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![-1.0, -2.0, -3.0];
    let dot = dot_product(&a, &b);
    assert!((dot - (-14.0)).abs() < 1e-6); // -1 -4 -9 = -14
}

#[test]
fn test_metrics_enum_cosine() {
    let metric = Metrics::Cosine;
    let a = vec![1.0, 0.0];
    let b = vec![1.0, 0.0];
    let sim = metric.calculate(&a, &b);
    assert!((sim - 1.0).abs() < 1e-6);
}

#[test]
fn test_metrics_enum_euclidean() {
    let metric = Metrics::Euclidean;
    let a = vec![0.0, 0.0];
    let b = vec![3.0, 4.0];
    let sim = metric.calculate(&a, &b);
    assert!((sim - (1.0 / 6.0)).abs() < 1e-6);
}

#[test]
fn test_metrics_enum_dot_product() {
    let metric = Metrics::DotProduct;
    let a = vec![2.0, 3.0];
    let b = vec![4.0, 5.0];
    let sim = metric.calculate(&a, &b);
    assert!((sim - 23.0).abs() < 1e-6);
}

#[test]
fn test_metrics_string() {
    assert_eq!(Metrics::Cosine.string(), "COSINE");
    assert_eq!(Metrics::Euclidean.string(), "EUCLIDEAN");
    assert_eq!(Metrics::DotProduct.string(), "DOT_PRODUCT");
}

#[test]
#[should_panic(expected = "Vector dimensions must match")]
fn test_cosine_similarity_mismatched_dimensions() {
    let a = vec![1.0, 2.0];
    let b = vec![1.0, 2.0, 3.0];
    cosine_similarity(&a, &b);
}

#[test]
#[should_panic(expected = "Vector dimensions must match")]
fn test_euclidean_similarity_mismatched_dimensions() {
    let a = vec![1.0, 2.0];
    let b = vec![1.0, 2.0, 3.0];
    euclidean_similarity(&a, &b);
}

#[test]
#[should_panic(expected = "Vector dimensions must match")]
fn test_dot_product_mismatched_dimensions() {
    let a = vec![1.0, 2.0];
    let b = vec![1.0, 2.0, 3.0];
    dot_product(&a, &b);
}

#[test]
fn test_cosine_similarity_large_vectors() {
    let dim = 1024;
    let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
    let sim = cosine_similarity(&a, &b);
    assert!((sim - 1.0).abs() < 1e-4);
}

#[test]
fn test_cosine_similarity_aligned_vectors() {
    let a = vec![2.0, 2.0];
    let b = vec![3.0, 3.0];
    let sim = cosine_similarity(&a, &b);
    assert!((sim - 1.0).abs() < 1e-6);
}
