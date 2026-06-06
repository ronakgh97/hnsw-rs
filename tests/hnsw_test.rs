use hnsw_rs::prelude::*;

/// Test Fixture: Initializes and populates an HNSW graph with random data.
/// Returns both the generated node IDs and their corresponding vectors.
fn setup_populated_hnsw(
    hnsw: &mut HNSW,
    num: usize,
    dim: usize,
    seed: usize,
) -> (Vec<[u8; 32]>, Vec<Vec<f32>>) {
    let mut vectors = gen_vec(num, dim, seed).0;
    let mut ids = Vec::with_capacity(num);

    for vector in vectors.iter_mut() {
        let id = gen_uuid();
        let level = hnsw.get_random_level();
        hnsw.insert(id, vector, gen_bytes(64), level).unwrap();
        ids.push(id);
    }
    (ids, vectors)
}

#[inline(always)]
pub fn gen_uuid() -> [u8; 32] {
    let mut id = [0u8; 32];
    fastrand::fill(&mut id);
    id
}

#[inline(always)]
pub fn gen_bytes(size: usize) -> Vec<u8> {
    let mut tmp = vec![0u8; size];
    fastrand::fill(&mut tmp);
    tmp
}

// --- TESTS ---

#[test]
fn test_hnsw_empty_search() {
    let hnsw = HNSW::default();
    assert!(hnsw.search(&mut [1.0, 0.0, 0.0], 3, None).is_empty());
}

#[test]
fn test_hnsw_basic_insertion_and_search() {
    let mut hnsw = HNSW::default();
    let mut vectors = [
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];

    for v in &mut vectors {
        hnsw.insert(gen_uuid(), v, vec![], 0).unwrap();
    }
    assert_eq!(hnsw.size(), 3);

    let results = hnsw.search(&mut [1.0, 0.0, 0.0], 1, None);
    assert_eq!(results.len(), 1);
}

#[test]
fn test_hnsw_search_returns_correct_k() {
    let mut hnsw = HNSW::default();
    let (_, mut vectors) = setup_populated_hnsw(&mut hnsw, 20, 32, 42);

    assert_eq!(hnsw.search(&mut vectors[0], 5, None).len(), 5);
}

#[test]
fn test_hnsw_search_with_different_ef() {
    let mut hnsw = HNSW::default();
    let (_, mut vectors) = setup_populated_hnsw(&mut hnsw, 50, 64, 99);

    assert_eq!(hnsw.search(&mut vectors[10], 5, Some(10)).len(), 5);
    assert_eq!(hnsw.search(&mut vectors[10], 5, Some(50)).len(), 5);
}

#[test]
fn test_hnsw_search_with_metadata() {
    let mut hnsw = HNSW::default();
    hnsw.insert(gen_uuid(), &mut [1.0, 0.0], b"meta0".to_vec(), 0)
        .unwrap();
    hnsw.insert(gen_uuid(), &mut [0.0, 1.0], b"meta1".to_vec(), 0)
        .unwrap();

    let results = hnsw.search_metadata(&mut [1.0, 0.0], 1, None);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].2, b"meta0");
}

#[test]
fn test_hnsw_search_with_different_metrics() {
    let test_cases = [
        (Metrics::Cosine, vec![1.0, 0.0, 0.0]),
        (Metrics::Euclidean, vec![0.0, 0.0, 0.0]),
        (Metrics::RawDot, vec![1.0, 0.0, 0.0]),
    ];

    for (metric, mut query) in test_cases {
        let mut hnsw = HNSW::new(16, 64, 4, 1.0, Some(metric), 1000, true);
        let mut vectors = [
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 0.0],
        ];

        for v in &mut vectors {
            hnsw.insert(gen_uuid(), v, vec![], hnsw.get_random_level())
                .unwrap();
        }
        assert_eq!(hnsw.search(&mut query, 2, None).len(), 2);
    }
}

#[test]
fn test_hnsw_brute_force_search() {
    let mut hnsw = HNSW::default();
    let (_, mut vectors) = setup_populated_hnsw(&mut hnsw, 30, 32, 123);

    let hnsw_ids: Vec<_> = hnsw
        .search(&mut vectors[5], 5, None)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let bf_ids: Vec<_> = hnsw
        .brute_search(&mut vectors[5], 5)
        .into_iter()
        .map(|(id, _)| id)
        .collect();

    assert!(hnsw_ids.iter().any(|id| bf_ids.contains(id)));
}

#[test]
fn test_hnsw_delete_node() {
    let mut hnsw = HNSW::default();
    let (ids, _) = setup_populated_hnsw(&mut hnsw, 10, 32, 1);

    hnsw.delete_node(&ids[0]).unwrap();
    assert_eq!(hnsw.active_count(), 9);
    assert_eq!(hnsw.tombstone_count(), 1);
    assert!(hnsw.delete_node(&gen_uuid()).is_err());
}

#[test]
fn test_hnsw_get_node() {
    let mut hnsw = HNSW::default();
    let id = gen_uuid();
    hnsw.insert(id, &mut [1.0, 2.0, 3.0], b"metadata".to_vec(), 0)
        .unwrap();

    assert_eq!(hnsw.get_node(&id).unwrap().metadata, b"metadata");
    assert!(hnsw.get_node(&gen_uuid()).is_none());
}

#[test]
fn test_hnsw_duplicate_insert() {
    let mut hnsw = HNSW::default();
    let id = [9u8; 32];
    hnsw.insert(id, &mut [1.0], vec![], 0).unwrap();
    assert!(hnsw.insert(id, &mut [2.0], vec![], 0).is_err());
}

#[test]
fn test_hnsw_active_count() {
    let mut hnsw = HNSW::default();
    let (ids, _) = setup_populated_hnsw(&mut hnsw, 5, 16, 5);

    assert_eq!(hnsw.active_count(), 5);
    hnsw.delete_node(&ids[0]).unwrap();
    hnsw.delete_node(&ids[1]).unwrap();
    assert_eq!(hnsw.active_count(), 3);
}

#[test]
fn test_hnsw_multiple_searches() {
    let mut hnsw = HNSW::default();
    let (_, mut vectors) = setup_populated_hnsw(&mut hnsw, 30, 48, 7);

    for i in 0..10 {
        let idx = (i * 6969) % 30;
        assert_eq!(hnsw.search(&mut vectors[idx], 5, None).len(), 5);
    }
}

#[test]
fn test_hnsw_with_large_vectors() {
    let mut hnsw = HNSW::default();
    let (_, mut vectors) = setup_populated_hnsw(&mut hnsw, 50, 512, 8);

    assert_eq!(hnsw.search(&mut vectors[25], 10, None).len(), 10);
}
