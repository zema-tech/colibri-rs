//! On-demand safetensors index.
//!
//! File layout (same contract Colibri checks in `st.h`):
//! 8-byte little-endian header length, UTF-8 JSON header, then raw tensor bytes.
//! Tensor `data_offsets` are relative to the first byte after the header.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use thiserror::Error;

/// Colibri refuses headers above 256 MiB.
pub const MAX_HEADER_BYTES: u64 = 256 << 20;

#[derive(Debug, Error)]
pub enum StError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad safetensors header: {0}")]
    Header(String),
    #[error("tensor `{0}` not in shard")]
    Missing(String),
}

pub type Result<T> = std::result::Result<T, StError>;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Absolute file offset of the first data byte.
    pub offset: u64,
    pub nbytes: u64,
}

/// One safetensors file, indexed but not fully loaded.
pub struct Shard {
    file: File,
    tensors: HashMap<String, TensorInfo>,
}

impl Shard {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut file = File::open(path.as_ref())?;
        let mut len_buf = [0u8; 8];
        file.read_exact(&mut len_buf)?;
        let header_len = u64::from_le_bytes(len_buf);
        if header_len == 0 || header_len > MAX_HEADER_BYTES {
            return Err(StError::Header(format!(
                "header length {header_len} out of range"
            )));
        }
        let mut header = vec![0u8; header_len as usize];
        file.read_exact(&mut header)?;
        let data_base = 8 + header_len;
        let value: serde_json::Value = serde_json::from_slice(&header)
            .map_err(|e| StError::Header(e.to_string()))?;
        let obj = value
            .as_object()
            .ok_or_else(|| StError::Header("header is not an object".into()))?;

        let mut tensors = HashMap::new();
        for (name, spec) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = spec
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| StError::Header(format!("{name}: missing dtype")))?
                .to_string();
            let shape = spec
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| StError::Header(format!("{name}: missing shape")))?
                .iter()
                .map(|n| {
                    n.as_u64()
                        .ok_or_else(|| StError::Header(format!("{name}: bad shape")))
                })
                .collect::<Result<Vec<_>>>()?;
            let offs = spec
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| StError::Header(format!("{name}: missing data_offsets")))?;
            if offs.len() != 2 {
                return Err(StError::Header(format!("{name}: data_offsets len")));
            }
            let start = offs[0]
                .as_u64()
                .ok_or_else(|| StError::Header(format!("{name}: offset")))?;
            let end = offs[1]
                .as_u64()
                .ok_or_else(|| StError::Header(format!("{name}: offset")))?;
            if end < start {
                return Err(StError::Header(format!("{name}: inverted offsets")));
            }
            tensors.insert(
                name.clone(),
                TensorInfo {
                    dtype,
                    shape,
                    offset: data_base + start,
                    nbytes: end - start,
                },
            );
        }
        Ok(Self { file, tensors })
    }

    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Read the raw bytes of one tensor. Does not decode the dtype.
    pub fn read(&mut self, name: &str) -> Result<Vec<u8>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| StError::Missing(name.to_string()))?
            .clone();
        let mut buf = vec![0u8; info.nbytes as usize];
        self.file.seek(SeekFrom::Start(info.offset))?;
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }
}

/// Write a single-shard safetensors file. `blobs` are raw payload bytes.
pub fn write_shard(path: impl AsRef<Path>, tensors: &[(&str, &str, &[u64], &[u8])]) -> Result<()> {
    let mut header = serde_json::Map::new();
    let mut cursor = 0u64;
    let mut body = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = cursor;
        let end = cursor + bytes.len() as u64;
        cursor = end;
        body.extend_from_slice(bytes);
        header.insert(
            (*name).to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, end],
            }),
        );
    }
    let header_bytes = serde_json::Value::Object(header).to_string().into_bytes();
    let mut file = File::create(path)?;
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
    file.write_all(&header_bytes)?;
    file.write_all(&body)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn roundtrip_two_tensors() {
        let path = env::temp_dir().join("colibri-rs-st-roundtrip.safetensors");
        let a = [1u8, 2, 3, 4];
        let b = [9u8, 8, 7, 6, 5, 4, 3, 2];
        write_shard(
            &path,
            &[
                ("layers.0.experts.0.gate", "U8", &[4], &a),
                ("layers.0.experts.0.up", "U8", &[8], &b),
            ],
        )
        .unwrap();
        let mut shard = Shard::open(&path).unwrap();
        assert_eq!(shard.read("layers.0.experts.0.gate").unwrap(), a);
        assert_eq!(shard.read("layers.0.experts.0.up").unwrap(), b);
        let info = shard.get("layers.0.experts.0.up").unwrap();
        assert_eq!(info.nbytes, 8);
        assert_eq!(info.shape, vec![8]);
        let _ = std::fs::remove_file(path);
    }
}
