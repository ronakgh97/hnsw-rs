use ahash::HashSet;
use anyhow::Result;
use arrow::array::{Array, Float64Array, ListArray};
use hnsw_rs::hnsw::NodeUUID;
use memmap2::MmapMut;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::PathBuf;

pub const DATASET_CACHE: &str = "./bench/cache/bench_data.bin";
pub const INDEX_CACHE_F32: &str = "./bench/bench_index_f32.bin";
pub const INDEX_CACHE_I8: &str = "./bench/bench_index_i8.bin";

pub enum BenchMetrics {
    QRSVaryingEF,
    RecallVaryingK,
    BuildTimeVaryingM,
    RecallVaryingM,
    BuildTimeVaryingMHeuristic,
    RecallVaryingMHeuristic,
}

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

#[inline]
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
