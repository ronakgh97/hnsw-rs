use ahash::HashSet;
use anyhow::Result;
use hnsw_rs::prelude::*;
use indicatif::ProgressBar;
use memmap2::Mmap;
use rand::RngExt;
use std::fs;
use std::fs::File;
use std::io::{Seek, Write};
use std::path::PathBuf;
use std::time::Instant;
// TODO: Something wrong, this is too slow, need to figure out later

const DATASET_CACHE: &str = "./examples/bench_data.bin";
const INDEX_CACHE: &str = "./examples/bench_index.bin";

enum BenchMetrics {
    QRSVaryingEF,
    RecallVaryingK,
}

struct BenchmarkResult {
    metrics: BenchMetrics,
    x_y: Vec<(f64, f64)>,
}

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("Usage: bench <input_parquet_dir> [num_files]");
    let num: usize = std::env::args()
        .nth(2)
        .expect("Usage: bench <input_parquet_dir> [num_files]")
        .parse()?;

    if !PathBuf::from(DATASET_CACHE).exists() {
        write_compact_bin(PathBuf::from(path), num)?;
    }

    let (num, dim, mmap) = load_vectors_mmap(PathBuf::from(DATASET_CACHE));

    println!("Total vectors: {}, dimension: {}", num, dim);
    // println!("Sample vector: {:?}", get_vector(&mmap, 0, dim));

    // Build largest index and cache it
    {
        if !PathBuf::from(INDEX_CACHE).exists() {
            cache_index(num, dim, &mmap);
        } else {
            println!(
                "Index cache already exists at {:?}, skipping index build",
                INDEX_CACHE
            );
        }
    }
    drop(mmap);

    let mut bench_results = Vec::<BenchmarkResult>::new();
    // Bench starts from here
    {
        let (num, _, mmap) = load_vectors_mmap(PathBuf::from(DATASET_CACHE));
        let hnsw = Storage::read_from_disk(&PathBuf::from(INDEX_CACHE))?;

        let mut rng = rand::rng();
        let query_count = 4096;
        let warmup_count = 2156;
        let ef_values = vec![32, 64, 128, 256, 512, 768];
        let k_values = vec![12, 24, 48, 96, 192, 384];

        let queries_idx: Vec<_> = (0..query_count).map(|_| rng.random_range(0..num)).collect();

        // Warm up
        {
            for query_vec in queries_idx.iter().take(warmup_count) {
                let _ = hnsw.search(get_vector(&mmap, *query_vec, dim), 32, Some(64));
            }
        }

        std::thread::sleep(std::time::Duration::from_millis(100));

        // Varying ef
        {
            let k = 1;
            let mut results = Vec::new();
            for &ef in &ef_values {
                let time = Instant::now();

                for query_vec in queries_idx.iter().take(query_count) {
                    // let idx = rng.random_range(0..num);
                    let query_vec = get_vector(&mmap, *query_vec, dim);
                    let _ = hnsw.search(query_vec, k, Some(ef));
                }
                let elapsed = time.elapsed();
                println!(
                    "Search with ef: {} took, QPS: {:.2}",
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
            let mut results = Vec::new();
            for &k in &k_values {
                let mut total_recall = 0.0f32;

                let start = Instant::now();
                for query_vec in queries_idx.iter().take(recall_sample) {
                    // let idx = rng.random_range(0..num);
                    let query_vec = get_vector(&mmap, *query_vec, dim);
                    let ef = k * 4;
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
                x_y: results.clone(),
            });
        }
    }

    plot_bench(bench_results, PathBuf::from("plot.png"))?;

    Ok(())
}

/// Reads parquet files from input directory, extracts "openai" column, converts to f32 and writes to compact binary format: [num_vectors: u32][dim: u32][vectors: f32...]
/// Datasets used: https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M
fn write_compact_bin(input: PathBuf, num_files: usize) -> Result<()> {
    use arrow::array::{Array, Float64Array, ListArray};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let mut paths: Vec<PathBuf> = fs::read_dir(&input)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|ext| ext == "parquet").unwrap_or(false))
        .collect();

    paths.sort();
    paths.truncate(num_files);

    println!("Found {} files", paths.len());

    let output = PathBuf::from(DATASET_CACHE);

    fs::create_dir_all(output.parent().unwrap())?;

    let mut writer = std::io::BufWriter::new(File::create(&output)?);

    writer.write_all(&0u32.to_le_bytes())?;
    writer.write_all(&0u32.to_le_bytes())?;

    let mut total = 0usize;
    let mut dim = 0usize;

    for path in paths {
        println!("Reading {:?}", path);

        let file = File::open(path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

        for batch in reader {
            let batch = batch?;

            let column = batch.column_by_name("openai").unwrap();
            let list = column.as_any().downcast_ref::<ListArray>().unwrap();

            let mut buffer = Vec::with_capacity(dim);

            for i in 0..list.len() {
                if list.is_null(i) {
                    continue;
                }

                let values = list.value(i);
                let floats64 = values.as_any().downcast_ref::<Float64Array>().unwrap();

                if dim == 0 {
                    dim = floats64.len();
                    buffer.reserve(dim);
                }

                buffer.clear();

                buffer.extend(floats64.values().iter().map(|&v| v as f32));

                let bytes =
                    unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const u8, dim * 4) };

                writer.write_all(bytes)?;
                total += 1;
            }
        }
    }

    writer.flush()?;
    drop(writer);

    // write header
    let mut file = fs::OpenOptions::new().write(true).open(&output)?;
    file.seek(std::io::SeekFrom::Start(0))?;

    file.write_all(&(total as u32).to_le_bytes())?;
    file.write_all(&(dim as u32).to_le_bytes())?;

    println!(
        "Written {} vectors of dimension {} to {:?}",
        total, dim, output
    );

    Ok(())
}

#[inline]
/// Calculate Recall@K - what fraction of HNSW results are in the true top-k (brute force)
fn compare_recall_at_k(
    hnsw_results: &[(NodeID, f32)],
    brute_results: &[(NodeID, f32)],
    k: usize,
) -> f32 {
    let brute_set: HashSet<NodeID> = brute_results
        .iter()
        .take(k)
        .map(|(id, _)| id.clone())
        .collect();
    let mut hits = 0;
    for (id, _) in hnsw_results.iter().take(k) {
        if brute_set.contains(id) {
            hits += 1;
        }
    }
    hits as f32 / k as f32
}

#[inline]
/// Get a vector slice from mmap data
fn get_vector(mmap: &Mmap, idx: usize, dim: usize) -> &[f32] {
    assert!(8 + (idx + 1) * dim * 4 <= mmap.len(), "Index out of bounds");
    let offset = 8 + idx * dim * 4; // 8 bytes header
    unsafe {
        let ptr = mmap.as_ptr().add(offset) as *const f32;
        std::slice::from_raw_parts(ptr, dim)
    }
}

/// Loads compact vectors, format: [num_vectors: u32][dim: u32][vectors: f32...]
/// Returns (num_vectors, dim, mmap)
fn load_vectors_mmap(path: PathBuf) -> (usize, usize, Mmap) {
    let path = File::open(&path).expect("Failed to open file");
    let mmap = unsafe { Mmap::map(&path).expect("Failed to mmap file") };

    // Read header
    let num_vectors = u32::from_le_bytes([mmap[0], mmap[1], mmap[2], mmap[3]]) as usize;
    let dim = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]) as usize;
    (num_vectors, dim, mmap)
}

fn cache_index(num_vectors: usize, dim: usize, mmap: &Mmap) -> HNSW {
    println!("Building index with {} vectors...", num_vectors);

    let mut hnsw = HNSW::new(
        32,
        256,
        18,
        1.0 / 16.0_f32.ln(),
        Some(Metrics::Cosine),
        num_vectors,
    );

    let pg_bar = ProgressBar::new(num_vectors as u64);

    let time = Instant::now();
    for i in 0..num_vectors {
        let vec = get_vector(mmap, i, dim);
        let level = hnsw.get_random_level();
        let id = format!("uuid_{}", i);
        hnsw.insert(id, vec, vec![], level).ok();
        pg_bar.inc(1);
    }

    Storage::flush_to_disk(&PathBuf::from(INDEX_CACHE), &hnsw)
        .expect("Failed to cache index to disk");
    println!("Index built in {:?} and cached to disk.", time.elapsed());

    hnsw
}

fn plot_bench(benchmark_results: Vec<BenchmarkResult>, output: PathBuf) -> Result<()> {
    use plotters::prelude::*;

    let root = BitMapBackend::new(&output, (1200, 900)).into_drawing_area();
    root.fill(&BLACK)?;

    let chart_area = root.split_evenly((2, 2)).into_iter().enumerate();

    for (idx, area) in chart_area {
        let (_x_label, _y_label, color, label) = match benchmark_results.get(idx) {
            Some(BenchmarkResult { metrics, .. }) => match metrics {
                BenchMetrics::QRSVaryingEF => ("EF", "QPS", RED, "Search Varying EF"),
                BenchMetrics::RecallVaryingK => ("K", "Recall@K", BLUE, "Recall Varying K"),
            },
            None => ("X", "Y", GREEN, "N/A"),
        };

        let (x_range, y_range) = if let Some(result) = benchmark_results.get(idx) {
            let xs: Vec<f64> = result.x_y.iter().map(|(x, _)| *x).collect();
            let ys: Vec<f64> = result.x_y.iter().map(|(_, y)| *y).collect();
            let x_min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
            let x_max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let y_min = ys.iter().cloned().fold(f64::INFINITY, f64::min);
            let y_max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let x_padding = (x_max - x_min) * 0.1;
            let y_padding = (y_max - y_min) * 0.1;
            (
                (x_min - x_padding)..(x_max + x_padding),
                (y_min - y_padding)..(y_max + y_padding),
            )
        } else {
            (0f64..1000f64, 0f64..1000f64)
        };

        let mut chart = ChartBuilder::on(&area)
            .caption(label, ("sans-serif", 20).into_font().color(&color))
            .x_label_area_size(40)
            .y_label_area_size(60)
            .margin(20)
            .build_cartesian_2d(x_range, y_range)?;

        chart
            .configure_mesh()
            .label_style(("sans-serif", 14).into_font().color(&WHITE))
            .axis_desc_style(("sans-serif", 16).into_font().color(&WHITE))
            .x_desc(_x_label)
            .y_desc(_y_label)
            .bold_line_style(WHITE.mix(0.3))
            .light_line_style(WHITE.mix(0.15))
            .draw()?;

        if let Some(result) = benchmark_results.get(idx) {
            chart.draw_series(LineSeries::new(result.x_y.clone().into_iter(), color))?;

            chart.draw_series(
                result
                    .x_y
                    .iter()
                    .map(|(x, y)| (*x, *y))
                    .map(|pos| Circle::new(pos, 5, color.filled())),
            )?;
        }
    }

    root.present()?;
    Ok(())
}
