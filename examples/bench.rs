use anyhow::Result;
use memmap2::Mmap;
use std::fs;
use std::fs::File;
use std::io::{Seek, Write};
use std::path::PathBuf;

const CACHE: &str = "./examples/bench_data.bin";

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("Usage: bench <input_parquet_dir> [num_files]");
    let num: usize = std::env::args()
        .nth(2)
        .expect("Usage: bench <input_parquet_dir> [num_files]")
        .parse()?;

    write_compact_bin(PathBuf::from(path), num)?;

    let (_num, dim, mmap) = load_vectors_mmap(PathBuf::from(CACHE));

    println!("Sample vector: {:?}", get_vector(&mmap, 0, dim));

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

    let output = PathBuf::from(CACHE);

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

            let mut buffer = Vec::new();

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

    println!("Loaded {} vectors of {} dimensions", num_vectors, dim);

    (num_vectors, dim, mmap)
}
