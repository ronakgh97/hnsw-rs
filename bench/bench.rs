use ahash::HashSet;
use anyhow::Result;
use hnsw_rs::prelude::*;
use memmap2::MmapMut;
use std::fs;
use std::fs::File;
use std::io::stdout;
use std::io::{Seek, Write};
use std::path::PathBuf;
use std::time::Instant;

// TODO: Something wrong, this is way beyond slow, need to figure out later

const DATASET_CACHE: &str = "./bench/bench_data.bin";
const INDEX_CACHE: &str = "./bench/bench_index.bin";

enum BenchMetrics {
    QRSVaryingEF,
    RecallVaryingK,
    BuildTimeVaryingM,
    RecallVaryingM,
    BuildTimeVaryingMHeuristic,
    RecallVaryingMHeuristic,
}

struct BenchmarkResult {
    metrics: BenchMetrics,
    x_y: Vec<(f64, f64)>,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let path = args
        .get(1)
        .expect("Usage: bench <input_parquet_dir> [num_files]")
        .to_owned();
    let num: usize = args
        .get(2)
        .expect("Usage: bench <input_parquet_dir> [num_files]")
        .parse()?;

    if !PathBuf::from(DATASET_CACHE).exists() {
        write_compact_datasets(PathBuf::from(path), num)?;
    }

    let (num, dim, mut mmap) = load_vectors_mmap(PathBuf::from(DATASET_CACHE));

    println!("Total vectors: {}, dimension: {}", num, dim);

    // Build largest index and cache it
    {
        if !PathBuf::from(INDEX_CACHE).exists() {
            cache_index(num, dim, &mut mmap);
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
        let (num, _, mut mmap) = load_vectors_mmap(PathBuf::from(DATASET_CACHE));
        let config = wincode::config::Configuration::default()
            .enable_zero_copy_align_check()
            .with_preallocation_size_limit::<PREALLOCATION_SIZE>();
        let hnsw: VectorHnsw = IndexStorage::read_from_disk(&PathBuf::from(INDEX_CACHE), config)?;

        let mut rng = fastrand::Rng::new();
        let query_count = 4096;
        let warmup_count = 2156;
        let ef_values = vec![32, 64, 128, 256, 512, 768];
        let k_values = vec![12, 24, 48, 96, 192, 384];

        let mut queries_idx: Vec<_> = (0..query_count).map(|_| rng.usize(0..num)).collect();

        // Warm up
        {
            for query_vec in queries_idx.iter_mut().take(warmup_count) {
                let _ = hnsw.search(get_vector(&mut mmap, *query_vec, dim), 32, Some(64));
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
                    let query_vec = get_vector(&mut mmap, *query_vec, dim);
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
                for query_vec in queries_idx.iter_mut().take(recall_sample) {
                    let ef = k * 4;
                    let query_vec = get_vector(&mut mmap, *query_vec, dim);
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
            let sample_count = 32540;
            let m_values = vec![8, 16, 32, 48, 64, 96];
            let mut build_results = Vec::new();
            let mut recall_results = Vec::new();
            let recall_sample = 1024;
            let k = 32;
            let ef = 128;
            let mut rng2 = fastrand::Rng::new();
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
            let sample_count = 32540;
            let m_values = vec![8, 16, 32, 48, 64, 96];
            let mut build_results = Vec::new();
            let mut recall_results = Vec::new();
            let recall_sample = 1024;
            let k = 32;
            let ef = 128;
            let mut rng2 = fastrand::Rng::new();
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

    plot_bench(bench_results, PathBuf::from("./bench/plot.png"))?;

    Ok(())
}

/// Reads parquet files from input directory, extracts "openai" column, converts to f32 and writes to compact binary format: [num_vectors: u32][dim: u32][vectors: f32...]
/// [Datasets](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M) used.
fn write_compact_datasets(input: PathBuf, num_files: usize) -> Result<()> {
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

    let mut writer = std::io::BufWriter::with_capacity(1024 * 1024 * 8, File::create(&output)?);

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

#[inline(always)]
/// Calculate Recall@K - what fraction of HNSW results are in the true top-k (brute force)
fn compare_recall_at_k(
    hnsw_results: &[(NodeUUID, f32)],
    brute_results: &[(NodeUUID, f32)],
    k: usize,
) -> f32 {
    let brute_set: HashSet<NodeUUID> = brute_results.iter().take(k).map(|(id, _)| *id).collect();
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
fn get_vector(mmap: &mut MmapMut, idx: usize, dim: usize) -> &mut [f32] {
    assert!(8 + (idx + 1) * dim * 4 <= mmap.len(), "Index out of bounds");
    let offset = 8 + idx * dim * 4; // 8 bytes header
    unsafe {
        let ptr = mmap.as_mut_ptr().add(offset) as *mut f32;
        std::slice::from_raw_parts_mut(ptr, dim)
    }
}

/// Loads compact vectors, format: [num_vectors: u32][dim: u32][vectors: f32...]
/// Returns (num_vectors, dim, mmap_mut)
fn load_vectors_mmap(path: PathBuf) -> (usize, usize, MmapMut) {
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .expect("Failed to open file");
    let mmap = unsafe { MmapMut::map_mut(&file).expect("Failed to mmap file") };

    // Read header
    let num_vectors = u32::from_le_bytes([mmap[0], mmap[1], mmap[2], mmap[3]]) as usize;
    let dim = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]) as usize;
    (num_vectors, dim, mmap)
}

fn build_index_with_m(
    m: usize,
    num_vectors: usize,
    dim: usize,
    mmap: &mut MmapMut,
) -> (VectorHnsw, std::time::Duration) {
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
        let mut id = [0u8; 32];
        id[0..8].copy_from_slice(&(i as u64).to_le_bytes());
        hnsw.insert(id, vec, vec![], level).ok();
    }
    (hnsw, time.elapsed())
}

fn build_index_with_m_heuristic(
    m: usize,
    num_vectors: usize,
    dim: usize,
    mmap: &mut MmapMut,
) -> (VectorHnsw, std::time::Duration) {
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
        let mut id = [0u8; 32];
        id[0..8].copy_from_slice(&(i as u64).to_le_bytes());
        hnsw.insert(id, vec, vec![], level).ok();
    }
    (hnsw, time.elapsed())
}

fn cache_index(num_vectors: usize, dim: usize, mmap: &mut MmapMut) -> VectorHnsw {
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
        let mut id = [0u8; 32];
        id[0..8].copy_from_slice(&(i as u64).to_le_bytes());
        hnsw.insert(id, vec, vec![], level).ok();
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
    IndexStorage::flush_to_disk(&PathBuf::from(INDEX_CACHE), &hnsw, config)
        .expect("Failed to save index to disk");
    println!("Saved index to disk...");
    let _ = stdout().flush();
    hnsw
}

fn plot_bench(benchmark_results: Vec<BenchmarkResult>, output: PathBuf) -> Result<()> {
    use plotters::prelude::*;

    let root = BitMapBackend::new(&output, (1800, 1200)).into_drawing_area();
    root.fill(&BLACK)?;

    let chart_area = root.split_evenly((3, 2)).into_iter().enumerate();

    for (idx, area) in chart_area {
        let (_x_label, _y_label, color, label) = match benchmark_results.get(idx) {
            Some(BenchmarkResult { metrics, .. }) => match metrics {
                BenchMetrics::QRSVaryingEF => ("EF", "QPS", RED, "Search Varying EF"),
                BenchMetrics::RecallVaryingK => ("K", "Recall@K", BLUE, "Recall Varying K"),
                BenchMetrics::BuildTimeVaryingM => (
                    "M",
                    "Build Time (s)",
                    GREEN,
                    "Build Time vs M (Simple selection)",
                ),
                BenchMetrics::RecallVaryingM => {
                    ("M", "Recall@K", YELLOW, "Recall vs M (Simple selection)")
                }
                BenchMetrics::BuildTimeVaryingMHeuristic => (
                    "M",
                    "Build Time (s)",
                    CYAN,
                    "Build Time vs M (Heuristic selection)",
                ),
                BenchMetrics::RecallVaryingMHeuristic => (
                    "M",
                    "Recall@K",
                    MAGENTA,
                    "Recall vs M (Heuristic selection)",
                ),
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
            .caption(label, ("0xProto Nerd Font", 20).into_font().color(&color))
            .x_label_area_size(40)
            .y_label_area_size(60)
            .margin(20)
            .build_cartesian_2d(x_range, y_range)?;

        chart
            .configure_mesh()
            .label_style(("0xProto Nerd Font", 14).into_font().color(&WHITE))
            .axis_desc_style(("0xProto Nerd Font", 16).into_font().color(&WHITE))
            .x_desc(_x_label)
            .y_desc(_y_label)
            .bold_line_style(WHITE.mix(0.3))
            .light_line_style(WHITE.mix(0.15))
            .draw()?;

        if let Some(result) = benchmark_results.get(idx) {
            chart.draw_series(LineSeries::new(result.x_y.clone(), color))?;

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
