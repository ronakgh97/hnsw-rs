use crate::prelude::HNSW;
use anyhow::Context;
use sha2::Digest;
use std::fs::File;
use std::path;
use wincode::{SchemaRead, SchemaWrite};

pub const PREALLOCATION_SIZE: usize = 512 * 1024 * 1024;

/// Unit struct for handling disk operations related to HNSW index
#[derive(SchemaRead, SchemaWrite)]
pub struct Storage;

impl Storage {
    /// Reads an HNSW index from disk using memory mapping for efficient access.
    pub fn read_from_disk(path: &path::Path) -> anyhow::Result<HNSW> {
        let path = path.to_path_buf();

        let file = File::open(&path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let config = wincode::config::Configuration::default()
            .enable_zero_copy_align_check()
            .with_preallocation_size_limit::<PREALLOCATION_SIZE>();

        let index = wincode::config::deserialize(&mmap[..], config)
            .with_context(|| format!("Failed to deserialize: {:?}", path))?;

        Ok(index)
    }

    /// Writes an HNSW index to disk and returns the sha256 checksum of the index
    pub fn flush_to_disk(path: &path::Path, index: &HNSW) -> anyhow::Result<String> {
        let path = path.to_path_buf();
        let config = wincode::config::Configuration::default()
            .enable_zero_copy_align_check()
            .with_preallocation_size_limit::<PREALLOCATION_SIZE>();

        let bytes = wincode::config::serialize(&index, config).with_context(|| {
            format!(
                "Failed to serialize vector data for disk at {}",
                path.display()
            )
        })?;

        let mut hasher = sha2::Sha256::new();
        hasher.update(&bytes);
        let checksum = hex::encode(hasher.finalize());

        std::fs::write(&path, bytes).with_context(|| {
            format!("Failed to write vector data to disk at {}", path.display())
        })?;

        Ok(checksum)
    }
}
