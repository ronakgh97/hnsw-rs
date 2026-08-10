use ahash::HashSet;
use anyhow::Result;
use arrow::array::{Array, Float64Array, ListArray};
use hnsw_rs::hnsw::NodeUUID;
use memmap2::MmapMut;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use plotters::prelude::*;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::PathBuf;

pub const DATASET_CACHE: &str = "./bench/cache/bench_data.bin";
pub const INDEX_CACHE_F32: &str = "./bench/cache/bench_index_f32.bin";
pub const INDEX_CACHE_I8: &str = "./bench/cache/bench_index_i8.bin";

#[allow(unused)]
/// BenchMetrics enum to represent different benchmark metrics
pub enum BenchMetrics {
    QRSVaryingEF,
    RecallVaryingK,
    BuildTimeVaryingM,
    RecallVaryingM,
    BuildTimeVaryingMHeuristic,
    RecallVaryingMHeuristic,
}

/// BenchmarkResult struct to hold benchmark results data
pub struct BenchmarkResult {
    pub metrics: BenchMetrics,
    pub x_y: Vec<(f64, f64)>,
}

/// Reads parquet files from input directory, extracts "openai" column, converts to f32 and writes to compact binary format: [num_vectors: u32][dim: u32][vectors: f32...]
/// [Datasets](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M) used.
pub fn write_compact_datasets(input: PathBuf, num_files: usize) -> Result<()> {
    let mut paths: Vec<PathBuf> = fs::read_dir(&input)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|ext| ext == "parquet").unwrap_or(false))
        .collect();

    paths.sort();
    paths.truncate(num_files);

    println!("Found {} files", paths.len());

    let output = PathBuf::from(DATASET_CACHE);

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }

    let file = File::create(&output)?;
    let mut buf_file_writer = BufWriter::with_capacity(8 * 1024 * 1024, file);

    buf_file_writer.write_all(&0u32.to_le_bytes())?;
    buf_file_writer.write_all(&0u32.to_le_bytes())?;

    let mut total_vector = 0usize;
    let mut vector_dim = 0;

    for path in paths {
        println!("Reading {:?}", path);

        let file = File::open(path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

        let mut scratch_buffer: Vec<f32> = Vec::with_capacity(4096);

        for batch in reader {
            let batch = batch?;

            let column = batch
                .column_by_name("openai")
                .map(|c| c.as_ref())
                .ok_or_else(|| anyhow::anyhow!("Column 'openai' not found"))?;
            let list = column
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| anyhow::anyhow!("Column 'openai' is not a ListArray"))?;

            for i in 0..list.len() {
                if list.is_null(i) {
                    continue;
                }

                let values = list.value(i);
                let floats64 = values
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| anyhow::anyhow!("Values are not Float64Array"))?;

                let dim = floats64.len();

                scratch_buffer.reserve(dim);
                scratch_buffer.clear();
                scratch_buffer.extend(floats64.values().iter().map(|&v| v as f32));

                let bytes = unsafe {
                    std::slice::from_raw_parts(scratch_buffer.as_ptr() as *const u8, dim * 4)
                };

                buf_file_writer.write_all(bytes)?;
                total_vector += 1;
                vector_dim = dim;
            }
        }
    }

    buf_file_writer.flush()?;

    // write header
    buf_file_writer.seek(SeekFrom::Start(0))?;
    buf_file_writer.write_all(&(total_vector as u32).to_le_bytes())?;
    buf_file_writer.write_all(&(vector_dim as u32).to_le_bytes())?;

    drop(buf_file_writer);

    println!(
        "Written {} vectors of dimension {} to {:?}",
        total_vector, vector_dim, output
    );

    Ok(())
}

/// Loads compact vectors, format: [num_vectors: u32][dim: u32][vectors: f32...]
/// Returns (num_vectors, dim, mmap_mut)
pub fn load_vectors_mmap(path: PathBuf) -> Result<(usize, usize, MmapMut)> {
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .expect("Failed to open file");
    let mmap = unsafe { MmapMut::map_mut(&file).expect("Failed to mmap file") };

    // Read header
    let num_vectors = u32::from_le_bytes([mmap[0], mmap[1], mmap[2], mmap[3]]) as usize;
    let dim = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]) as usize;
    Ok((num_vectors, dim, mmap))
}

#[inline(always)]
/// Get a vector slice from mmap data
pub fn get_vector(mmap: &mut MmapMut, idx: usize, dim: usize) -> &mut [f32] {
    assert!(8 + (idx + 1) * dim * 4 <= mmap.len(), "Index out of bounds");
    let offset = 8 + idx * dim * 4; // 8 bytes header
    unsafe {
        let ptr = mmap.as_mut_ptr().add(offset) as *mut f32;
        std::slice::from_raw_parts_mut(ptr, dim)
    }
}

#[inline(always)]
/// Calculate Recall@K - what fraction of HNSW results are in the true top-k (brute force)
pub fn compare_recall_at_k(
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

/// Plot benchmark results [`BenchmarkResult`], empty results will be marked as "N/A", and save to output path
pub fn plot_bench(benchmark_results: Vec<BenchmarkResult>, output: PathBuf) -> Result<()> {
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
