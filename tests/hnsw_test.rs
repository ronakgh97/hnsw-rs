use hnsw_rs::maths::Metrics;
use hnsw_rs::prelude::*;

#[test]
fn test_hnsw_basic_insert() {
    let mut hnsw = HNSW::default();
    let vectors = [
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];

    for (i, vector) in vectors.iter().enumerate() {
        let level = hnsw.get_random_level();
        hnsw.insert(i.to_string(), vector, vec![], level).unwrap();
    }

    assert_eq!(hnsw.nodes.len(), 3);
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
    hnsw.insert("0".to_string(), &[1.0, 0.0, 0.0], vec![], 0)
        .unwrap();

    let results = hnsw.search(&[1.0, 0.0, 0.0], 1, None);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "0");
}

#[test]
fn test_hnsw_search_returns_correct_k() {
    let mut hnsw = HNSW::default();
    let num_vectors = 20;
    let dimensions = 32;

    let vectors = generate_random_vectors(num_vectors, dimensions, 42, false);

    for (i, vector) in vectors.iter().enumerate() {
        let level = hnsw.get_random_level();
        hnsw.insert(i.to_string(), vector, vec![], level).unwrap();
    }

    let query = vectors[0].clone();
    let results = hnsw.search(&query, 5, None);

    assert_eq!(results.len(), 5);
    assert_eq!(results[0].0, "0");
}

#[test]
fn test_hnsw_search_with_different_ef() {
    let mut hnsw = HNSW::default();
    let vectors = generate_random_vectors(50, 64, 99, false);

    for (i, vector) in vectors.iter().enumerate() {
        let level = hnsw.get_random_level();
        hnsw.insert(i.to_string(), vector, vec![], level).unwrap();
    }

    let query = vectors[10].clone();

    let results_ef10 = hnsw.search(&query, 5, Some(10));
    let results_ef50 = hnsw.search(&query, 5, Some(50));

    assert_eq!(results_ef10.len(), 5);
    assert_eq!(results_ef50.len(), 5);
}

#[test]
fn test_hnsw_search_with_metadata() {
    let mut hnsw = HNSW::default();

    hnsw.insert("0".to_string(), &[1.0, 0.0], b"meta0".to_vec(), 0)
        .unwrap();
    hnsw.insert("1".to_string(), &[0.0, 1.0], b"meta1".to_vec(), 0)
        .unwrap();

    let results = hnsw.search_with_metadata(&[1.0, 0.0], 1, None);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "0");
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

    for (i, vector) in vectors.iter().enumerate() {
        let level = hnsw.get_random_level();
        hnsw.insert(i.to_string(), vector, vec![], level).unwrap();
    }

    let results = hnsw.search(&[1.0, 0.0, 0.0], 2, None);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, "0");
}

#[test]
fn test_hnsw_search_with_euclidean_metric() {
    let mut hnsw = HNSW::new(16, 64, 4, 1.0, Some(Metrics::Euclidean), 1000);

    let vectors = [vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];

    for (i, vector) in vectors.iter().enumerate() {
        let level = hnsw.get_random_level();
        hnsw.insert(i.to_string(), vector, vec![], level).unwrap();
    }

    let results = hnsw.search(&[0.0, 0.0], 2, None);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, "0");
}

#[test]
fn test_hnsw_search_with_dot_product_metric() {
    let mut hnsw = HNSW::new(16, 64, 4, 1.0, Some(Metrics::DotProduct), 1000);

    let vectors = [
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.5, 0.5, 0.0],
    ];

    for (i, vector) in vectors.iter().enumerate() {
        let level = hnsw.get_random_level();
        hnsw.insert(i.to_string(), vector, vec![], level).unwrap();
    }

    let results = hnsw.search(&[1.0, 0.0, 0.0], 2, None);
    assert_eq!(results.len(), 2);
}

#[test]
fn test_hnsw_brute_force_search() {
    let mut hnsw = HNSW::default();
    let vectors = generate_random_vectors(30, 32, 123, false);

    for (i, vector) in vectors.iter().enumerate() {
        let level = hnsw.get_random_level();
        hnsw.insert(i.to_string(), vector, vec![], level).unwrap();
    }

    let query = vectors[5].clone();

    let hnsw_results = hnsw.search(&query, 5, None);
    let brute_force_results = hnsw.brute_force_search(&query, 5);

    let hnsw_ids: Vec<_> = hnsw_results.iter().map(|(id, _)| id).collect();
    let bf_ids: Vec<_> = brute_force_results.iter().map(|(id, _)| id).collect();

    assert!(hnsw_ids.iter().any(|id| bf_ids.contains(id)));
}

#[test]
fn test_hnsw_delete_node() {
    let mut hnsw = HNSW::default();
    let vectors = generate_random_vectors(10, 32, 1, false);

    for (i, vector) in vectors.iter().enumerate() {
        let level = hnsw.get_random_level();
        hnsw.insert(i.to_string(), vector, vec![], level).unwrap();
    }

    hnsw.delete_node_by_id("5").unwrap();

    assert_eq!(hnsw.active_count(), 9);
    assert_eq!(hnsw.tombstone_count(), 1);
}

#[test]
fn test_hnsw_delete_nonexistent() {
    let mut hnsw = HNSW::default();
    hnsw.insert("0".to_string(), &[1.0], vec![], 0).unwrap();

    let result = hnsw.delete_node_by_id("999");
    assert!(result.is_err());
}

#[test]
fn test_hnsw_tombstone_ratio() {
    let mut hnsw = HNSW::default();
    let vectors = generate_random_vectors(10, 32, 2, false);

    for (i, vector) in vectors.iter().enumerate() {
        let level = hnsw.get_random_level();
        hnsw.insert(i.to_string(), vector, vec![], level).unwrap();
    }

    assert_eq!(hnsw.tombstone_ratio(), 0.0);

    hnsw.delete_node_by_id("0").unwrap();
    hnsw.delete_node_by_id("1").unwrap();
    hnsw.delete_node_by_id("2").unwrap();

    assert!((hnsw.tombstone_ratio() - 0.3).abs() < 0.001);
}

#[test]
fn test_hnsw_reindex() {
    let mut hnsw = HNSW::default();
    let vectors = generate_random_vectors(10, 32, 3, false);

    for (i, vector) in vectors.iter().enumerate() {
        let level = hnsw.get_random_level();
        hnsw.insert(i.to_string(), vector, vec![], level).unwrap();
    }

    hnsw.delete_node_by_id("0").unwrap();
    hnsw.delete_node_by_id("1").unwrap();
    hnsw.delete_node_by_id("2").unwrap();

    hnsw.reindex().unwrap();

    assert_eq!(hnsw.nodes.len(), 7);
    assert_eq!(hnsw.active_count(), 7);
    assert_eq!(hnsw.tombstone_count(), 0);
}

#[test]
fn test_hnsw_reindex_preserves_search() {
    let mut hnsw = HNSW::default();
    let vectors = generate_random_vectors(20, 32, 4, false);

    for (i, vector) in vectors.iter().enumerate() {
        let level = hnsw.get_random_level();
        hnsw.insert(i.to_string(), vector, vec![], level).unwrap();
    }

    let query = vectors[5].clone();
    let before_reindex = hnsw.search(&query, 5, None);

    hnsw.delete_node_by_id("0").unwrap();
    hnsw.delete_node_by_id("1").unwrap();
    hnsw.reindex().unwrap();

    let after_reindex = hnsw.search(&query, 5, None);

    assert_eq!(before_reindex.len(), after_reindex.len());
}

#[test]
fn test_hnsw_get_node_by_id() {
    let mut hnsw = HNSW::default();
    hnsw.insert(
        "test".to_string(),
        &[1.0, 2.0, 3.0],
        b"metadata".to_vec(),
        0,
    )
    .unwrap();

    let node = hnsw.get_node_by_id("test");
    assert!(node.is_some());
    assert_eq!(node.unwrap().node_id, "test");
    assert_eq!(node.unwrap().metadata, b"metadata");
}

#[test]
fn test_hnsw_get_node_by_id_nonexistent() {
    let hnsw = HNSW::default();
    let node = hnsw.get_node_by_id("nonexistent");
    assert!(node.is_none());
}

#[test]
fn test_hnsw_node_is_deleted() {
    let mut hnsw = HNSW::default();
    hnsw.insert("0".to_string(), &[1.0], vec![], 0).unwrap();

    let node = hnsw.get_node_by_id("0").unwrap();
    assert!(!node.is_deleted());

    hnsw.delete_node_by_id("0").unwrap();

    let node = hnsw.get_node_by_id("0").unwrap();
    assert!(node.is_deleted());
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
    let vectors = generate_random_vectors(5, 16, 5, false);

    for (i, vector) in vectors.iter().enumerate() {
        let level = hnsw.get_random_level();
        hnsw.insert(i.to_string(), vector, vec![], level).unwrap();
    }

    assert_eq!(hnsw.active_count(), 5);

    hnsw.delete_node_by_id("1").unwrap();
    hnsw.delete_node_by_id("2").unwrap();

    assert_eq!(hnsw.active_count(), 3);
}

#[test]
fn test_hnsw_search_preserves_entry_point() {
    let mut hnsw = HNSW::default();
    let vectors = generate_random_vectors(10, 32, 6, false);

    for (i, vector) in vectors.iter().enumerate() {
        let level = hnsw.get_random_level();
        hnsw.insert(i.to_string(), vector, vec![], level).unwrap();
    }

    let entry_before = hnsw.entry_point;

    hnsw.search(&vectors[0], 3, None);

    assert_eq!(hnsw.entry_point, entry_before);
}

#[test]
fn test_hnsw_multiple_searches() {
    let mut hnsw = HNSW::default();
    let vectors = generate_random_vectors(30, 48, 7, false);

    for (i, vector) in vectors.iter().enumerate() {
        let level = hnsw.get_random_level();
        hnsw.insert(i.to_string(), vector, vec![], level).unwrap();
    }

    for i in 0..10 {
        let idx = (i * 7919) % 30;
        let query = vectors[idx].clone();
        let results = hnsw.search(&query, 5, None);
        assert_eq!(results.len(), 5);
    }
}

#[test]
fn test_hnsw_with_large_vectors() {
    let mut hnsw = HNSW::default();
    let vectors = generate_random_vectors(50, 512, 8, false);

    for (i, vector) in vectors.iter().enumerate() {
        let level = hnsw.get_random_level();
        hnsw.insert(i.to_string(), vector, vec![], level).unwrap();
    }

    let query = vectors[25].clone();
    let results = hnsw.search(&query, 10, None);

    assert_eq!(results.len(), 10);
}

#[test]
fn test_hnsw_with_high_ef_construction() {
    let mut hnsw = HNSW::new(16, 512, 4, 1.0, Some(Metrics::Cosine), 1000);
    let vectors = generate_random_vectors(20, 32, 9, false);

    for (i, vector) in vectors.iter().enumerate() {
        let level = hnsw.get_random_level();
        hnsw.insert(i.to_string(), vector, vec![], level).unwrap();
    }

    let query = vectors[10].clone();
    let results = hnsw.search(&query, 5, None);

    assert_eq!(results.len(), 5);
}

#[test]
fn test_hnsw_search_results_sorted_by_similarity() {
    let mut hnsw = HNSW::default();
    let vectors = generate_random_vectors(20, 32, 10, false);

    for (i, vector) in vectors.iter().enumerate() {
        let level = hnsw.get_random_level();
        hnsw.insert(i.to_string(), vector, vec![], level).unwrap();
    }

    let query = vectors[0].clone();
    let results = hnsw.search(&query, 5, None);

    for i in 1..results.len() {
        assert!(results[i - 1].1 >= results[i].1);
    }
}
