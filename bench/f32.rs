pub mod utils;

use crate::utils::{
    BenchMetrics, BenchmarkResult, DATASET_CACHE, INDEX_CACHE_F32, compare_recall_at_k, get_vector,
    load_vectors_mmap, plot_bench, write_compact_datasets,
};
use anyhow::Result;
use fastrand::Rng;
use hnsw_rs::prelude::*;
use memmap2::MmapMut;
use std::io::Write;
use std::io::stdout;
use std::path::PathBuf;
use std::time::{Duration, Instant};

// TODO: Harness could be improved
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let path: String = args
        .get(1)
        .expect("Usage: f32 <input_parquet_dir> [num_files]")
        .parse()?;
    let num: usize = args
        .get(2)
        .expect("Usage: f32 <input_parquet_dir> [num_files]")
        .parse()?;

    if !PathBuf::from(DATASET_CACHE).exists() {
        write_compact_datasets(PathBuf::from(path), num)?;
    } else {
        println!(
            "Dataset cache already exists at {:?}, skipping dataset compaction",
            DATASET_CACHE
        );
    }

    let (num, dim, mut mmap) = load_vectors_mmap(PathBuf::from(DATASET_CACHE))?;

    println!("Total vectors: {}, dimension: {}", num, dim);

    // Build largest index and cache it
    {
        if !PathBuf::from(INDEX_CACHE_F32).exists() {
            build_cache_index(num, dim, &mut mmap);
        } else {
            println!(
                "Index cache already exists at {:?}, skipping index build",
                INDEX_CACHE_F32
            );
        }
    }
    drop(mmap); // drop here to save mem

    // Bench starts from here
    let mut bench_results = Vec::<BenchmarkResult>::with_capacity(6);
    {
        let (num, _, mut mmap) = load_vectors_mmap(PathBuf::from(DATASET_CACHE))?;
        let config = wincode::config::Configuration::default()
            .enable_zero_copy_align_check()
            .with_preallocation_size_limit::<PREALLOCATION_SIZE>();
        let hnsw: VectorHnsw =
            IndexStorage::read_from_disk(&PathBuf::from(INDEX_CACHE_F32), config)?;

        let mut rng = Rng::new();
        let query_count = 4096;
        let warmup_count = 2156;
        let ef_values = vec![32, 64, 128, 256, 512, 768];
        let k_values = vec![12, 24, 48, 96, 192, 384];

        let mut queries_idx: Vec<usize> = (0..query_count).map(|_| rng.usize(0..num)).collect();

        // Warm up
        {
            for query_vec in queries_idx.iter_mut().take(warmup_count) {
                let _ = hnsw.search(get_vector(&mut mmap, *query_vec, dim), 32, Some(64));
            }
        }

        std::thread::sleep(Duration::from_millis(100));

        // Varying ef
        {
            let k = 1;
            let mut results = Vec::with_capacity(ef_values.len());
            for &ef in &ef_values {
                let time = Instant::now();

                for query_idx in queries_idx.iter().take(query_count) {
                    let query_vec = get_vector(&mut mmap, *query_idx, dim);
                    let _ = hnsw.search(query_vec, k, Some(ef));
                }
                let elapsed = time.elapsed();
                println!(
                    "Search with ef: {}, QPS: {:.2}",
                    ef,
                    query_count as f64 / elapsed.as_secs_f64()
                );
                results.push((ef as f64, query_count as f64 / elapsed.as_secs_f64()));
            }
            bench_results.push(BenchmarkResult {
                metrics: BenchMetrics::QRSVaryingEF,
                x_y: results,
            });
        }

        // Recall@k
        {
            let recall_sample = 1024;
            let mut results = Vec::with_capacity(k_values.len());
            for &k in &k_values {
                let mut total_recall = 0.0f32;

                let start = Instant::now();
                for query_idx in queries_idx.iter_mut().take(recall_sample) {
                    let ef = k * 4;
                    let query_vec = get_vector(&mut mmap, *query_idx, dim);
                    let hnsw_search = hnsw.search(query_vec, k, Some(ef));
                    let brute_search = hnsw.brute_search(query_vec, k);

                    total_recall += compare_recall_at_k(&hnsw_search, &brute_search, k);
                }
                let elapsed = start.elapsed().as_secs_f32();

                let avg_recall = total_recall / recall_sample as f32;
                println!("Recall@{}: {:.4}, Time: {}", k, avg_recall, elapsed);
                results.push((k as f64, avg_recall as f64));
            }
            bench_results.push(BenchmarkResult {
                metrics: BenchMetrics::RecallVaryingK,
                x_y: results,
            });
        }

        // Build time and recall varying M (Alg. 3 simple, paper default)
        {
            println!("Simple selection");
            let m_values = vec![8, 16, 32, 48, 64, 96];
            let mut build_results = Vec::with_capacity(m_values.len());
            let mut recall_results = Vec::with_capacity(m_values.len());
            let sample_count = 32540;
            let recall_sample = 1024;
            let k = 32;
            let ef = 128;
            let mut rng2 = Rng::new();
            let queries_idx_m: Vec<_> = (0..recall_sample)
                .map(|_| rng2.usize(0..sample_count))
                .collect();

            for &m in &m_values {
                let (hnsw, elapsed) = build_index_with_m(m, sample_count, dim, &mut mmap);
                build_results.push((m as f64, elapsed.as_secs_f64()));

                let mut total_recall = 0.0f32;
                for &query_idx in &queries_idx_m {
                    let query_vec = get_vector(&mut mmap, query_idx, dim);
                    let hnsw_search = hnsw.search(query_vec, k, Some(ef));
                    let brute_search = hnsw.brute_search(query_vec, k);
                    total_recall += compare_recall_at_k(&hnsw_search, &brute_search, k);
                }

                let avg_recall = total_recall / recall_sample as f32;
                println!(
                    "M: {}, Build Time: {:.4}s, Recall@{}: {:.4}",
                    m,
                    elapsed.as_secs_f64(),
                    k,
                    avg_recall
                );
                recall_results.push((m as f64, avg_recall as f64));
            }

            bench_results.push(BenchmarkResult {
                metrics: BenchMetrics::BuildTimeVaryingM,
                x_y: build_results,
            });
            bench_results.push(BenchmarkResult {
                metrics: BenchMetrics::RecallVaryingM,
                x_y: recall_results,
            });
        }

        // Build time and recall varying M (Alg. 4 heuristic)
        {
            println!("Heuristic selection");
            let m_values = vec![8, 16, 32, 48, 64, 96];
            let mut build_results = Vec::with_capacity(m_values.len());
            let mut recall_results = Vec::with_capacity(m_values.len());
            let sample_count = 32540;
            let recall_sample = 1024;
            let k = 32;
            let ef = 128;
            let mut rng2 = Rng::new();
            let queries_idx_m: Vec<_> = (0..recall_sample)
                .map(|_| rng2.usize(0..sample_count))
                .collect();

            for &m in &m_values {
                let (hnsw, elapsed) = build_index_with_m_heuristic(m, sample_count, dim, &mut mmap);
                build_results.push((m as f64, elapsed.as_secs_f64()));

                let mut total_recall = 0.0f32;
                for &query_idx in &queries_idx_m {
                    let query_vec = get_vector(&mut mmap, query_idx, dim);
                    let hnsw_search = hnsw.search(query_vec, k, Some(ef));
                    let brute_search = hnsw.brute_search(query_vec, k);
                    total_recall += compare_recall_at_k(&hnsw_search, &brute_search, k);
                }

                let avg_recall = total_recall / recall_sample as f32;
                println!(
                    "M: {}, Build Time: {:.4}s, Recall@{}: {:.4}",
                    m,
                    elapsed.as_secs_f64(),
                    k,
                    avg_recall
                );
                recall_results.push((m as f64, avg_recall as f64));
            }

            bench_results.push(BenchmarkResult {
                metrics: BenchMetrics::BuildTimeVaryingMHeuristic,
                x_y: build_results,
            });
            bench_results.push(BenchmarkResult {
                metrics: BenchMetrics::RecallVaryingMHeuristic,
                x_y: recall_results,
            });
        }
    }

    plot_bench(bench_results, PathBuf::from("./bench/plot_f32.png"))?;

    Ok(())
}

fn build_index_with_m(
    m: usize,
    num_vectors: usize,
    dim: usize,
    mmap: &mut MmapMut,
) -> (VectorHnsw, Duration) {
    let ef_const = 96.max(2 * m);
    let mut hnsw: VectorHnsw = HNSW::with_options(
        FlatVectorStore::init(dim, Metrics::Cosine, num_vectors),
        m,
        ef_const,
        18,
        1.0 / (m as f32).ln(),
        num_vectors,
        false,
        false,
        true,
        false,
    );

    let time = Instant::now();
    for i in 0..num_vectors {
        let vec = get_vector(mmap, i, dim);
        let level = hnsw.get_random_level();
        hnsw.insert(i as u128, vec, level).ok();
    }
    (hnsw, time.elapsed())
}

fn build_index_with_m_heuristic(
    m: usize,
    num_vectors: usize,
    dim: usize,
    mmap: &mut MmapMut,
) -> (VectorHnsw, Duration) {
    let ef_const = 96.max(2 * m);
    let mut hnsw: VectorHnsw = HNSW::with_options(
        FlatVectorStore::init(dim, Metrics::Cosine, num_vectors),
        m,
        ef_const,
        18,
        1.0 / (m as f32).ln(),
        num_vectors,
        false,
        false,
        true,
        true,
    );

    let time = Instant::now();
    for i in 0..num_vectors {
        let vec = get_vector(mmap, i, dim);
        let level = hnsw.get_random_level();
        hnsw.insert(i as u128, vec, level).ok();
    }
    (hnsw, time.elapsed())
}

fn build_cache_index(num_vectors: usize, dim: usize, mmap: &mut MmapMut) -> VectorHnsw {
    println!("Building index with {} vectors...", num_vectors);
    let _ = stdout().flush();

    let mut hnsw: VectorHnsw = HNSW::new(
        FlatVectorStore::init(dim, Metrics::Cosine, num_vectors),
        16,
        96,
        18,
        1.0 / 16.0_f32.ln(),
        num_vectors,
        true,
        true,
    );

    let time = Instant::now();
    for i in 0..num_vectors {
        print!(
            "\rIndexing: {}/{} ({:.1}%)",
            i + 1,
            num_vectors,
            (i + 1) as f64 / num_vectors as f64 * 100.0
        );
        stdout().flush().ok();

        let vec = get_vector(mmap, i, dim);
        let level = hnsw.get_random_level();
        hnsw.insert(i as u128, vec, level).ok();
    }
    println!();

    println!(
        "Index built in {:?} with {} insert/s",
        time.elapsed(),
        num_vectors as u64 / time.elapsed().as_secs()
    );

    let config = wincode::config::Configuration::default()
        .enable_zero_copy_align_check()
        .with_preallocation_size_limit::<PREALLOCATION_SIZE>();
    IndexStorage::flush_to_disk(&PathBuf::from(INDEX_CACHE_F32), &hnsw, config)
        .expect("Failed to save index to disk");

    let file_size_on_disk = std::fs::metadata(INDEX_CACHE_F32)
        .expect("Failed to get file metadata")
        .len();

    println!(
        "Saved index to disk, size on disk: {} bytes",
        file_size_on_disk
    );
    let _ = stdout().flush();
    hnsw
}
