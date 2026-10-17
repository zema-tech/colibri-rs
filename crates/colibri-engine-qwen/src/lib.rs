//! Engine Qwen-MoE (fascia 30B–35B-A3B).
//!
//! Corrisponde concettualmente a `qwen36.c` in Colibrì.
//! Qui: caricamento path + cache expert + generate stub.
//! Il forward reale (attention, router, FFN expert) va implementato fase 3.

use std::path::PathBuf;

use colibri_core::{
    ColibriError, Engine, EngineStats, GenerateRequest, PlacementConfig, Result,
};
use colibri_expert::SharedExpertCache;
use colibri_tok::{Tokenizer, WhitespaceTokenizer};

/// Engine Qwen MoE (scaffold).
pub struct QwenMoeEngine {
    model_dir: PathBuf,
    cfg: PlacementConfig,
    expert_cache: SharedExpertCache,
    tok: WhitespaceTokenizer,
    stats: EngineStats,
}

impl Engine for QwenMoeEngine {
    fn name(&self) -> &str {
        "qwen-moe-scaffold"
    }

    fn load(model_dir: PathBuf, cfg: PlacementConfig) -> Result<Self> {
        if !model_dir.exists() {
            return Err(ColibriError::InvalidModel(format!(
                "directory assente: {}",
                model_dir.display()
            )));
        }
        let expert_cache = SharedExpertCache::new(cfg.expert_cache_bytes);
        Ok(Self {
            model_dir,
            cfg,
            expert_cache,
            tok: WhitespaceTokenizer::new(),
            stats: EngineStats::default(),
        })
    }

    fn generate(&mut self, req: &GenerateRequest) -> Result<String> {
        // Scaffold: non esegue il forward. Solo tokenizer + messaggio chiaro.
        let ids = self.tok.encode(&req.prompt)?;
        self.stats.tokens_generated = self.stats.tokens_generated.saturating_add(1);

        Ok(format!(
            "[colibri-rs scaffold] engine={} dir={} prompt_tokens={} max_new={} temperature={:.2}\n\
             Forward MoE non ancora implementato. Prossimi passi: parser pesi, kernel quant, router+expert FFN.\n\
             Cache expert budget: {} MiB",
            self.name(),
            self.model_dir.display(),
            ids.len(),
            req.max_tokens,
            req.temperature,
            self.cfg.expert_cache_bytes / (1024 * 1024)
        ))
    }

    fn stats(&self) -> EngineStats {
        let (hits, misses, used) = self.expert_cache.stats();
        EngineStats {
            resident_dense_bytes: self.stats.resident_dense_bytes,
            expert_cache_bytes: used,
            expert_hits: hits,
            expert_misses: misses,
            tokens_generated: self.stats.tokens_generated,
        }
    }
}
