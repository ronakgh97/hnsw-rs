use crate::prelude::*;
use anyhow::Context;
use anyhow::Result;
use sha2::Digest;
use std::fs::File;
use std::path::PathBuf;
use wincode::{SchemaRead, SchemaWrite};

pub const PREALLOCATION_SIZE: usize = 1024 * 1024 * 1024; // oh, yeah

/// Unit struct for handling disk operations related to HNSW index
pub struct IndexStorage;

impl IndexStorage {
    /// Reads an HNSW index from disk using memory mapping for efficient access.
    /// Takes [wincode::config::Configuration] for deserialization or [`wincode::config::Configuration::default`] for most cases.
    ///
    /// Requires that `I` (the [backend](ItemBackend)) implements `SchemaRead` + `SchemaWrite` with `Dst = I` / `Src = I`.
    /// ([`FlatVectorStore`] has this derives, so [`VectorHnsw`] works *automagically*)
    pub fn read_from_disk<I, C>(path: &PathBuf, config: C) -> Result<HNSW<I>>
    where
        I: ItemBackend + for<'de> SchemaRead<'de, C, Dst = I> + SchemaWrite<C, Src = I>,
        C: wincode::config::Config,
    {
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        let index: HNSW<I> = wincode::config::deserialize(&mmap[..], config)
            .with_context(|| format!("Failed to read from: {:?}", path))?;

        Ok(index)
    }

    /// Writes an `HNSW<I>` index to disk. *(TODO: this is suboptimal implementation, it allocated gigantic memory for the whole index)*
    /// Takes [wincode::config::Configuration] for serialization and returns a Result containing the sha256 checksum as a hex string.
    pub fn flush_to_disk<I, C>(path: &PathBuf, index: &HNSW<I>, config: C) -> Result<String>
    where
        I: ItemBackend + for<'de> SchemaRead<'de, C, Dst = I> + SchemaWrite<C, Src = I>,
        C: wincode::config::Config,
    {
        // TODO; fix this huge alloc later, find some better ways
        let bytes = wincode::config::serialize(index, config)
            .with_context(|| "Failed to serialize the Index".to_string())?;

        std::fs::write(path, &bytes)
            .with_context(|| format!("Failed to write to disk at {}", path.display()))?;

        Ok(hex::encode(sha2::Sha256::digest(&bytes)))
    }
}
