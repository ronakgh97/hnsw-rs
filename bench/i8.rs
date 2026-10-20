pub mod utils;

use crate::utils::{
    BenchMetrics, BenchmarkResult, DATASET_CACHE, INDEX_CACHE_I8, compare_recall_at_k, get_vector,
    load_vectors_mmap, plot_bench, write_compact_datasets,
};
use anyhow::Result;
use fastrand::Rng;
use hnsw_rs::prelude::*;
use memmap2::MmapMut;
use std::io::{Write, stdout};
use std::path::PathBuf;
use std::time::{Duration, Instant};

// TODO: Harness could be improved
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let path: String = args
        .get(1)
        .expect("Usage: bencher <input_parquet_dir> [num_files]")
        .parse()?;
    let num: usize = args
        .get(2)
        .expect("Usage: bencher <input_parquet_dir> [num_files]")
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

    // Build quantized big index and cache them son of b
    {
        if !PathBuf::from(INDEX_CACHE_I8).exists() {
            build_quant_cache_index(num, dim, &mut mmap);
        } else {
            println!(
                "Index cache already exists at {:?}, skipping index build",
                INDEX_CACHE_I8
            );
        }
    }
    drop(mmap);

    // Load quantized index and run benchmarks
    let mut bench_results = Vec::<BenchmarkResult>::with_capacity(6);
    {
        let (num, _, mut mmap) = load_vectors_mmap(PathBuf::from(DATASET_CACHE))?;
        let config = wincode::config::Configuration::default()
            .enable_zero_copy_align_check()
            .with_preallocation_size_limit::<PREALLOCATION_SIZE>();
        let hnsw: Vector8bitHnsw =
            IndexStorage::read_from_disk(&PathBuf::from(INDEX_CACHE_I8), config)?;

        let mut rng = Rng::new();
        let query_count = 4096;
        let warmup_count = 2156;
        let ef_values = [32, 64, 128, 256, 512, 768];
        let k_values = [12, 24, 48, 96, 192, 384];

        let queries_idx: Vec<usize> = (0..query_count).map(|_| rng.usize(0..num)).collect();

        // Warm up
        {
            let mut i8_vec = vec![0i8; dim];
            for &query_idx in queries_idx.iter().take(warmup_count) {
                let f32_vec = get_vector(&mut mmap, query_idx, dim);
                let _ = quant(f32_vec, &mut i8_vec);
                let _ = hnsw.search(&mut i8_vec, 32, Some(64));
            }
        }

        std::thread::sleep(Duration::from_millis(100));

        // Varying ef
        {
            let k = 1;
            let mut results = Vec::with_capacity(ef_values.len());
            for &ef in &ef_values {
                let mut i8_vec = vec![0i8; dim];

                let time = Instant::now();
                for &query_idx in queries_idx.iter().take(query_count) {
                    let f32_vec = get_vector(&mut mmap, query_idx, dim);
                    quant(f32_vec, &mut i8_vec);
                    let _ = hnsw.search(&mut i8_vec, k, Some(ef));
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

        // Recall@K
        {
            let recall_sample = 1024;
            let mut results = Vec::with_capacity(k_values.len());
            for &k in &k_values {
                let mut total_recall = 0.0f32;
                let mut i8_vec = vec![0i8; dim];

                let start = Instant::now();
                for &query_idx in queries_idx.iter().take(recall_sample) {
                    let ef = k * 4;
                    let query_vec = get_vector(&mut mmap, query_idx, dim);
                    quant(query_vec, &mut i8_vec); // quant
                    let hnsw_search = hnsw.search(&mut i8_vec, k, Some(ef));
                    let brute_search = hnsw.brute_search(&mut i8_vec, k);

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
    }

    plot_bench(bench_results, PathBuf::from("bench/plot_i8.png"))?;

    Ok(())
}

fn build_quant_cache_index(num_vectors: usize, dim: usize, mmap: &mut MmapMut) -> Vector8bitHnsw {
    println!("Building quantized index with {} vectors...", num_vectors);
    let _ = stdout().flush();

    let mut hnsw: Vector8bitHnsw = HNSW::new(
        FlatI8VectorStore::init(dim, Metrics::Cosine, num_vectors),
        16,
        96,
        18,
        1.0 / 16.0_f32.ln(),
        num_vectors,
        true,
        true,
    );

    let mut i8_vec_buf = vec![0i8; dim];

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
        let _scale = quant(vec, &mut i8_vec_buf);
        let level = hnsw.get_random_level();
        hnsw.insert(i as u128, &mut i8_vec_buf, level).ok();
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
    IndexStorage::flush_to_disk(&PathBuf::from(INDEX_CACHE_I8), &hnsw, config)
        .expect("Failed to save index to disk");

    let file_size_on_disk = std::fs::metadata(INDEX_CACHE_I8)
        .expect("Failed to get file metadata")
        .len();

    println!(
        "Saved index to disk, size on disk: {} bytes",
        file_size_on_disk
    );

    let _ = stdout().flush();
    hnsw
}
