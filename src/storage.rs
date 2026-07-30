use crate::prelude::*;
use anyhow::Context;
use anyhow::Result;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use wincode::io::{WriteResult, Writer};
use wincode::{SchemaRead, SchemaWrite};

pub const PREALLOCATION_SIZE: usize = 1024 * 1024 * 1024; // oh, yeah

/// Unit struct for handling disk operations related to HNSW index
pub struct IndexStorage;

impl IndexStorage {
    /// Reads an HNSW index from disk using memory mapping for writing efficiently.
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

    /// Streams an `HNSW<I>` index to disk with buffer size of 6MB.
    /// Takes [wincode::config::Configuration] for serialization and returns the sha256 checksum as a hex string.
    pub fn flush_to_disk<I, C>(path: &PathBuf, index: &HNSW<I>, config: C) -> Result<String>
    where
        I: ItemBackend + for<'de> SchemaRead<'de, C, Dst = I> + SchemaWrite<C, Src = I>,
        C: wincode::config::Config,
    {
        let file = File::create(path)?;
        let file_buf = BufWriter::with_capacity(6 * 1024 * 1024, file);

        let mut writer = HashWriter {
            inner: file_buf,
            hasher: Sha256::new(),
        };

        wincode::config::serialize_into(&mut writer, index, config)
            .with_context(|| "Failed to buffer index to disk".to_string())?;

        writer.finish()?;

        Ok(writer.finalize())
    }
}

/// A wrapper around a writer that computes the SHA256 hash of the data as its being written.
struct HashWriter<W> {
    inner: W,
    hasher: Sha256,
}

impl<W: Writer> Writer for HashWriter<W> {
    fn finish(&mut self) -> WriteResult<()> {
        self.inner.finish()
    }

    fn write(&mut self, src: &[u8]) -> WriteResult<()> {
        self.inner.write(src)?;
        self.hasher.update(src);
        Ok(())
    }
}

impl<W> HashWriter<W> {
    fn finalize(self) -> String {
        hex::encode(self.hasher.finalize())
    }
}
