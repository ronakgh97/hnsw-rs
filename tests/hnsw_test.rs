use hnsw_rs::prelude::*;

#[inline(always)]
fn gen_helper() -> (String, Vec<u8>) {
    (encode(gen_bytes(32)), gen_bytes(64))
}

// THESE TEST ARE AI-GENERATED

#[test]
fn test_hnsw_basic_insert() {
    let mut hnsw = HNSW::default();
    let vectors = [
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];

    for vector in vectors.iter() {
        let (idx, meta) = gen_helper();
        let level = hnsw.get_random_level();
        hnsw.insert(idx, vector, meta, level).unwrap();
    }

    assert_eq!(hnsw.count(), 3);
}

#[test]
fn test_hnsw_search_empty() {
    let hnsw: HNSW = HNSW::default();
    let results = hnsw.search(&[1.0, 0.0, 0.0], 3, None);
    assert!(results.is_empty());
}

#[test]
fn test_hnsw_search_single_node() {
    let mut hnsw = HNSW::default();
    hnsw.insert("X".to_string(), &[1.0, 0.0, 0.0], vec![], 0)
        .unwrap();

    let results = hnsw.search(&[1.0, 0.0, 0.0], 1, None);
    assert_eq!(results.len(), 1);
}

#[test]
fn test_hnsw_search_returns_correct_k() {
    let mut hnsw = HNSW::default();
    let num_vectors = 20;
    let dimensions = 32;

    let vectors = gen_vec(num_vectors, dimensions, 42);

    for vector in vectors.0.iter() {
        let (idx, meta) = gen_helper();
        let level = hnsw.get_random_level();
        hnsw.insert(idx, vector, meta, level).unwrap();
    }

    let query = vectors.0[0].clone();
    let results = hnsw.search(&query, 5, None);

    assert_eq!(results.len(), 5);
}

#[test]
fn test_hnsw_search_with_different_ef() {
    let mut hnsw = HNSW::default();
    let vectors = gen_vec(50, 64, 99);

    for vector in vectors.0.iter() {
        let level = hnsw.get_random_level();
        let (idx, meta) = gen_helper();
        hnsw.insert(idx, vector, meta, level).unwrap();
    }

    let query = vectors.0[10].clone();

    let results_ef10 = hnsw.search(&query, 5, Some(10));
    let results_ef50 = hnsw.search(&query, 5, Some(50));

    assert_eq!(results_ef10.len(), 5);
    assert_eq!(results_ef50.len(), 5);
}

#[test]
fn test_hnsw_search_with_metadata() {
    let mut hnsw = HNSW::default();

    hnsw.insert("X".to_string(), &[1.0, 0.0], b"meta0".to_vec(), 0)
        .unwrap();
    hnsw.insert("Y".to_string(), &[0.0, 1.0], b"meta1".to_vec(), 0)
        .unwrap();

    let results = hnsw.search_with_metadata(&[1.0, 0.0], 1, None);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].2, b"meta0");
}

#[test]
fn test_hnsw_search_with_cosine_metric() {
    let mut hnsw = HNSW::new(16, 64, 4, 1.0, Some(Metrics::Cosine), 1000);

    let vectors = [
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 1.0, 0.0],
    ];

    for vector in vectors.iter() {
        let (idx, meta) = gen_helper();
        let level = hnsw.get_random_level();
        hnsw.insert(idx, vector, meta, level).unwrap();
    }

    let results = hnsw.search(&[1.0, 0.0, 0.0], 2, None);
    assert_eq!(results.len(), 2);
}

#[test]
fn test_hnsw_search_with_euclidean_metric() {
    let mut hnsw = HNSW::new(16, 64, 4, 1.0, Some(Metrics::Euclidean), 1000);

    let vectors = [vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];

    for vector in vectors.iter() {
        let (idx, meta) = gen_helper();
        let level = hnsw.get_random_level();
        hnsw.insert(idx, vector, meta, level).unwrap();
    }

    let results = hnsw.search(&[0.0, 0.0], 2, None);
    assert_eq!(results.len(), 2);
}

#[test]
fn test_hnsw_search_with_dot_product_metric() {
    let mut hnsw = HNSW::new(16, 64, 4, 1.0, Some(Metrics::RawDot), 1000);

    let vectors = [
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.5, 0.5, 0.0],
    ];

    for vector in vectors.iter() {
        let (idx, meta) = gen_helper();
        let level = hnsw.get_random_level();
        hnsw.insert(idx, vector, meta, level).unwrap();
    }

    let results = hnsw.search(&[1.0, 0.0, 0.0], 2, None);
    assert_eq!(results.len(), 2);
}

#[test]
fn test_hnsw_brute_force_search() {
    let mut hnsw = HNSW::default();
    let vectors = gen_vec(30, 32, 123);

    for vector in vectors.0.iter() {
        let (idx, meta) = gen_helper();
        let level = hnsw.get_random_level();
        hnsw.insert(idx, vector, meta, level).unwrap();
    }

    let query = vectors.0[5].clone();

    let hnsw_results = hnsw.search(&query, 5, None);
    let brute_force_results = hnsw.brute_search(&query, 5);

    let hnsw_ids: Vec<_> = hnsw_results.iter().map(|(id, _)| id).collect();
    let bf_ids: Vec<_> = brute_force_results.iter().map(|(id, _)| id).collect();

    assert!(hnsw_ids.iter().any(|id| bf_ids.contains(id)));
}

#[test]
fn test_hnsw_delete_node() {
    let mut hnsw = HNSW::default();
    let vectors = gen_vec(10, 32, 1);

    for (i, vector) in vectors.0.iter().enumerate() {
        let level = hnsw.get_random_level();
        hnsw.insert(i.to_string(), vector, vec![], level).unwrap();
    }

    hnsw.delete_node("6").unwrap();

    assert_eq!(hnsw.active_count(), 9);
    assert_eq!(hnsw.tombstone_count(), 1);
}

#[test]
fn test_hnsw_delete_nonexistent() {
    let mut hnsw = HNSW::default();
    hnsw.insert("0".to_string(), &[1.0], vec![], 0).unwrap();

    let result = hnsw.delete_node("99999999");
    assert!(result.is_err());
}

#[test]
fn test_hnsw_get_node() {
    let mut hnsw = HNSW::default();
    let idx = hnsw
        .insert("X".to_string(), &[1.0, 2.0, 3.0], b"metadata".to_vec(), 0)
        .unwrap();

    let node = hnsw.get_node(&idx);
    assert!(node.is_some());
    assert_eq!(node.unwrap().metadata, b"metadata");
}

#[test]
fn test_hnsw_get_node_nonexistent() {
    let hnsw = HNSW::default();
    let node = hnsw.get_node("999");
    assert!(node.is_none());
}

#[test]
fn test_hnsw_duplicate_insert() {
    let mut hnsw = HNSW::default();
    hnsw.insert("0".to_string(), &[1.0], vec![], 0).unwrap();

    let result = hnsw.insert("0".to_string(), &[2.0], vec![], 0);
    assert!(result.is_err());
}

#[test]
fn test_hnsw_active_count() {
    let mut hnsw = HNSW::default();
    let vectors = gen_vec(5, 16, 5);

    for (i, vector) in vectors.0.iter().enumerate() {
        let level = hnsw.get_random_level();
        hnsw.insert(i.to_string(), vector, vec![], level).unwrap();
    }

    assert_eq!(hnsw.active_count(), 5);

    hnsw.delete_node("1").unwrap();
    hnsw.delete_node("2").unwrap();

    assert_eq!(hnsw.active_count(), 3);
}

// #[test]
// fn test_hnsw_search_preserves_entry_point() {
//     let mut hnsw = HNSW::default();
//     let vectors = gen_vec(10, 32, 6);
//
//     for (i, vector) in vectors.0.iter().enumerate() {
//         let level = hnsw.get_random_level();
//         hnsw.insert(vector, vec![], level).unwrap();
//     }
//
//     let entry_before = hnsw
//         .get_entry_point()
//         .map(|node| node.node_id.clone())
//         .unwrap();
//
//     hnsw.search(&vectors.0[0], 3, None);
//
//     assert_eq!(
//         hnsw.get_entry_point()
//             .map(|node| node.node_id.clone())
//             .unwrap(),
//         entry_before
//     );
// }

#[test]
fn test_hnsw_multiple_searches() {
    let mut hnsw = HNSW::default();
    let vectors = gen_vec(30, 48, 7);

    for vector in vectors.0.iter() {
        let (idx, meta) = gen_helper();
        let level = hnsw.get_random_level();
        hnsw.insert(idx, vector, meta, level).unwrap();
    }

    for i in 0..10 {
        let idx = (i * 7919) % 30;
        let query = vectors.0[idx].clone();
        let results = hnsw.search(&query, 5, None);
        assert_eq!(results.len(), 5);
    }
}

#[test]
fn test_hnsw_with_large_vectors() {
    let mut hnsw = HNSW::default();
    let vectors = gen_vec(50, 512, 8);

    for vector in vectors.0.iter() {
        let (idx, meta) = gen_helper();
        let level = hnsw.get_random_level();
        hnsw.insert(idx, vector, meta, level).unwrap();
    }

    let query = vectors.0[25].clone();
    let results = hnsw.search(&query, 10, None);

    assert_eq!(results.len(), 10);
}

#[test]
fn test_hnsw_with_high_ef_construction() {
    let mut hnsw = HNSW::new(16, 512, 4, 1.0, Some(Metrics::Cosine), 1000);
    let vectors = gen_vec(20, 32, 9);

    for vector in vectors.0.iter() {
        let (idx, meta) = gen_helper();
        let level = hnsw.get_random_level();
        hnsw.insert(idx, vector, meta, level).unwrap();
    }

    let query = vectors.0[10].clone();
    let results = hnsw.search(&query, 5, None);

    assert_eq!(results.len(), 5);
}

#[test]
fn test_hnsw_search_results_sorted_by_similarity() {
    let mut hnsw = HNSW::default();
    let vectors = gen_vec(20, 32, 10);

    for vector in vectors.0.iter() {
        let (idx, meta) = gen_helper();
        let level = hnsw.get_random_level();
        hnsw.insert(idx, vector, meta, level).unwrap();
    }

    let query = vectors.0[0].clone();
    let results = hnsw.search(&query, 5, None);

    for i in 1..results.len() {
        assert!(results[i - 1].1 >= results[i].1);
    }
}
