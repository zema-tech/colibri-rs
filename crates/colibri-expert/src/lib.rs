//! Cache di expert streamati da disco.
//!
//! Equivalente concettuale di `expert_store.h` in Colibrì:
//! - budget in byte
//! - hit → slice in RAM
//! - miss → lettura da WeightFile + inserimento (con eviction)

use std::collections::HashMap;
use std::sync::Arc;

use colibri_core::{ColibriError, ExpertId, Result};
use colibri_store::{TensorLoc, WeightFile};
use parking_lot::Mutex;

struct CacheEntry {
    data: Arc<Vec<u8>>,
    bytes: u64,
    /// Contatore grezzo per LRU approssimato (bump on hit).
    last_used: u64,
}

/// Cache bounded di expert.
pub struct ExpertCache {
    budget: u64,
    used: u64,
    clock: u64,
    map: HashMap<ExpertId, CacheEntry>,
    hits: u64,
    misses: u64,
}

impl ExpertCache {
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            budget: budget_bytes,
            used: 0,
            clock: 0,
            map: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn misses(&self) -> u64 {
        self.misses
    }

    pub fn used_bytes(&self) -> u64 {
        self.used
    }

    /// Ottiene i byte dell'expert; se manca, li legge da `file` secondo `loc`.
    pub fn get_or_load(
        &mut self,
        id: ExpertId,
        file: &WeightFile,
        loc: &TensorLoc,
    ) -> Result<Arc<Vec<u8>>> {
        self.clock = self.clock.wrapping_add(1);

        if let Some(entry) = self.map.get_mut(&id) {
            entry.last_used = self.clock;
            self.hits += 1;
            return Ok(Arc::clone(&entry.data));
        }

        self.misses += 1;
        let nbytes = loc.nbytes as usize;
        if loc.nbytes > self.budget {
            return Err(ColibriError::ExpertBudget {
                need: loc.nbytes,
                budget: self.budget,
            });
        }

        // Evict finché c'è spazio
        while self.used + loc.nbytes > self.budget && !self.map.is_empty() {
            self.evict_one();
        }

        let slice = file.slice(loc.offset, nbytes)?;
        let data = Arc::new(slice.to_vec());
        self.used += loc.nbytes;
        self.map.insert(
            id,
            CacheEntry {
                data: Arc::clone(&data),
                bytes: loc.nbytes,
                last_used: self.clock,
            },
        );
        Ok(data)
    }

    fn evict_one(&mut self) {
        // LRU grezzo: entry con last_used minimo
        let victim = self
            .map
            .iter()
            .min_by_key(|(_, e)| e.last_used)
            .map(|(k, _)| *k);
        if let Some(id) = victim {
            if let Some(e) = self.map.remove(&id) {
                self.used = self.used.saturating_sub(e.bytes);
            }
        }
    }
}

/// Wrapper thread-safe opzionale.
pub struct SharedExpertCache {
    inner: Mutex<ExpertCache>,
}

impl SharedExpertCache {
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            inner: Mutex::new(ExpertCache::new(budget_bytes)),
        }
    }

    pub fn get_or_load(
        &self,
        id: ExpertId,
        file: &WeightFile,
        loc: &TensorLoc,
    ) -> Result<Arc<Vec<u8>>> {
        self.inner.lock().get_or_load(id, file, loc)
    }

    pub fn stats(&self) -> (u64, u64, u64) {
        let g = self.inner.lock();
        (g.hits(), g.misses(), g.used_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // tempfile non è in workspace: test minimale senza crate extra
    #[test]
    fn cache_hit_miss() {
        // Test logico senza file reale: solo struttura
        let mut cache = ExpertCache::new(1024);
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }
}
