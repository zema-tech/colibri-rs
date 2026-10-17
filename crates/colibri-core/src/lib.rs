//! Nucleo condiviso: errori, config, trait Engine.
//!
//! Allineato alla filosofia Colibrì: un engine per famiglia, pesi densità vs expert.

use std::path::PathBuf;

use thiserror::Error;

/// Errori di runtime del motore.
#[derive(Debug, Error)]
pub enum ColibriError {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),

    #[error("modello non valido: {0}")]
    InvalidModel(String),

    #[error("expert cache piena o budget superato (need {need} bytes, budget {budget})")]
    ExpertBudget { need: u64, budget: u64 },

    #[error("tokenizer: {0}")]
    Tokenizer(String),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, ColibriError>;

/// Configurazione di placement (RAM vs disco), come COLI_* in Colibrì.
#[derive(Debug, Clone)]
pub struct PlacementConfig {
    /// Budget massimo della cache expert in byte (RAM).
    pub expert_cache_bytes: u64,
    /// Contesto massimo in token.
    pub max_context: usize,
    /// Thread CPU per matmul / decode.
    pub threads: usize,
}

impl Default for PlacementConfig {
    fn default() -> Self {
        Self {
            // ~4 GiB di cache expert di default su macchina 16 GB
            expert_cache_bytes: 4 * 1024 * 1024 * 1024,
            max_context: 4096,
            threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .min(4),
        }
    }
}

/// Identificatore di un expert (layer, indice).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertId {
    pub layer: u32,
    pub expert: u32,
}

/// Richiesta di generazione minima.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
}

/// Trait che ogni engine (qwen, olmoe, …) implementa.
///
/// Equivalente concettuale di un `*.c` per famiglia in Colibrì.
pub trait Engine: Send {
    fn name(&self) -> &str;

    fn load(model_dir: PathBuf, cfg: PlacementConfig) -> Result<Self>
    where
        Self: Sized;

    fn generate(&mut self, req: &GenerateRequest) -> Result<String>;

    /// Statistiche di placement (hit rate cache, byte residenti, …).
    fn stats(&self) -> EngineStats;
}

#[derive(Debug, Clone, Default)]
pub struct EngineStats {
    pub resident_dense_bytes: u64,
    pub expert_cache_bytes: u64,
    pub expert_hits: u64,
    pub expert_misses: u64,
    pub tokens_generated: u64,
}

impl EngineStats {
    pub fn hit_rate(&self) -> f64 {
        let t = self.expert_hits + self.expert_misses;
        if t == 0 {
            0.0
        } else {
            self.expert_hits as f64 / t as f64
        }
    }
}
