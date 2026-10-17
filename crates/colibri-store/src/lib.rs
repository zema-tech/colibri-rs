//! Store dei pesi: file aperti in mmap, lettura a range.
//!
//! In Colibrì `st.h` evita mmap per tenere RSS piatto e usa pread.
//! Qui partiamo con mmap (semplice); si può aggiungere pread dopo.

use std::fs::File;
use std::path::{Path, PathBuf};

use colibri_core::{ColibriError, Result};
use memmap2::Mmap;

/// Un file di pesi mappato (o porzione logica).
pub struct WeightFile {
    path: PathBuf,
    mmap: Mmap,
}

impl WeightFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        // SAFETY: file non modificato mentre mappato (pesi immutabili a runtime).
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { path, mmap })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// Slice [offset, offset+len).
    pub fn slice(&self, offset: u64, len: usize) -> Result<&[u8]> {
        let start = offset as usize;
        let end = start.checked_add(len).ok_or_else(|| {
            ColibriError::InvalidModel(format!(
                "overflow slice {}+{} in {}",
                offset,
                len,
                self.path.display()
            ))
        })?;
        self.mmap.get(start..end).ok_or_else(|| {
            ColibriError::InvalidModel(format!(
                "range [{}, {}) fuori file {} (len {})",
                start,
                end,
                self.path.display(),
                self.mmap.len()
            ))
        })
    }
}

/// Indice grezzo di un tensore su disco (offset + size).
/// In seguito: parser safetensors / header GGUF.
#[derive(Debug, Clone)]
pub struct TensorLoc {
    pub file_index: usize,
    pub offset: u64,
    pub nbytes: u64,
}
