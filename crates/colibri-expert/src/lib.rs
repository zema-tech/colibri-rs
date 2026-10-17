//! Expert store: keep a bounded set of routed experts in RAM and stream the
//! rest from a safetensors shard.
//!
//! Lease contract (from Colibri `expert_store.h`):
//! - `lookup` holds one lease until `release`
//! - a leased slot is never evicted
//! - `prefetch` is advisory and does not take a lease
//! - `release` on an unknown id is a no-op

use std::collections::HashMap;
use std::path::Path;

use colibri_st::Shard;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertKey {
    pub layer: i32,
    pub expert: i32,
}

#[derive(Debug, Clone)]
pub struct ExpertView {
    pub key: ExpertKey,
    pub gate: Vec<u8>,
    pub up: Vec<u8>,
    pub down: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct ExpertStats {
    pub requests: u64,
    pub hits: u64,
    pub misses: u64,
    pub prefetched: u64,
    pub prefetch_hits: u64,
    pub bytes_read: u64,
    pub resident_bytes: u64,
    pub capacity_bytes: u64,
}

#[derive(Debug, Error)]
pub enum ExpertError {
    #[error(transparent)]
    Store(#[from] colibri_st::StError),
    #[error("expert {0:?} does not fit in the cache ({1} bytes > capacity {2})")]
    TooBig(ExpertKey, u64, u64),
    #[error("expert {0:?} missing tensors in shard")]
    Missing(ExpertKey),
}

struct Slot {
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
    bytes: u64,
    leases: u32,
    last_used: u64,
}

pub struct ExpertStore {
    shard: Shard,
    capacity: u64,
    resident: u64,
    clock: u64,
    slots: HashMap<ExpertKey, Slot>,
    requests: u64,
    hits: u64,
    misses: u64,
    prefetched: u64,
    prefetch_hits: u64,
    bytes_read: u64,
}

impl ExpertStore {
    /// Tensor names follow `layers.{L}.experts.{E}.{gate,up,down}`.
    pub fn open(path: impl AsRef<Path>, capacity_bytes: u64) -> Result<Self, ExpertError> {
        Ok(Self {
            shard: Shard::open(path)?,
            capacity: capacity_bytes,
            resident: 0,
            clock: 0,
            slots: HashMap::new(),
            requests: 0,
            hits: 0,
            misses: 0,
            prefetched: 0,
            prefetch_hits: 0,
            bytes_read: 0,
            })
    }

    pub fn lookup(&mut self, key: ExpertKey) -> Result<ExpertView, ExpertError> {
        self.requests += 1;
        self.clock = self.clock.wrapping_add(1);
        if let Some(slot) = self.slots.get_mut(&key) {
            slot.leases += 1;
            slot.last_used = self.clock;
            self.hits += 1;
            return Ok(ExpertView {
                key,
                gate: slot.gate.clone(),
                up: slot.up.clone(),
                down: slot.down.clone(),
            });
        }
        self.misses += 1;
        self.load(key, true)
    }

    pub fn release(&mut self, key: ExpertKey) {
        if let Some(slot) = self.slots.get_mut(&key) {
            slot.leases = slot.leases.saturating_sub(1);
        }
    }

    /// Load keys that are absent. Never evicts a leased slot. Returns how many
    /// were newly read.
    pub fn prefetch(&mut self, keys: &[ExpertKey]) -> Result<usize, ExpertError> {
        let mut loaded = 0;
        for key in keys {
            self.clock = self.clock.wrapping_add(1);
            if self.slots.contains_key(key) {
                self.prefetch_hits += 1;
                if let Some(slot) = self.slots.get_mut(key) {
                    slot.last_used = self.clock;
                }
                continue;
            }
            match self.load(*key, false) {
                Ok(_) => {
                    self.prefetched += 1;
                    loaded += 1;
                }
                Err(ExpertError::TooBig(_, _, _)) => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(loaded)
    }

    pub fn stats(&self) -> ExpertStats {
        ExpertStats {
            requests: self.requests,
            hits: self.hits,
            misses: self.misses,
            prefetched: self.prefetched,
            prefetch_hits: self.prefetch_hits,
            bytes_read: self.bytes_read,
            resident_bytes: self.resident,
            capacity_bytes: self.capacity,
        }
    }

    pub fn leases(&self, key: ExpertKey) -> u32 {
        self.slots.get(&key).map(|s| s.leases).unwrap_or(0)
    }

    fn names(key: ExpertKey) -> (String, String, String) {
        (
            format!("layers.{}.experts.{}.gate", key.layer, key.expert),
            format!("layers.{}.experts.{}.up", key.layer, key.expert),
            format!("layers.{}.experts.{}.down", key.layer, key.expert),
        )
    }

    fn load(&mut self, key: ExpertKey, lease: bool) -> Result<ExpertView, ExpertError> {
        let (g, u, d) = Self::names(key);
        if self.shard.get(&g).is_none() || self.shard.get(&u).is_none() || self.shard.get(&d).is_none()
        {
            return Err(ExpertError::Missing(key));
        }
        let gate = self.shard.read(&g)?;
        let up = self.shard.read(&u)?;
        let down = self.shard.read(&d)?;
        let bytes = (gate.len() + up.len() + down.len()) as u64;
        if bytes > self.capacity {
            return Err(ExpertError::TooBig(key, bytes, self.capacity));
        }
        self.evict_until(bytes)?;
        self.bytes_read += bytes;
        self.resident += bytes;
        self.clock = self.clock.wrapping_add(1);
        self.slots.insert(
            key,
            Slot {
                gate: gate.clone(),
                up: up.clone(),
                down: down.clone(),
                bytes,
                leases: if lease { 1 } else { 0 },
                last_used: self.clock,
            },
        );
        Ok(ExpertView {
            key,
            gate,
            up,
            down,
        })
    }

    fn evict_until(&mut self, need: u64) -> Result<(), ExpertError> {
        while self.resident + need > self.capacity {
            let victim = self
                .slots
                .iter()
                .filter(|(_, s)| s.leases == 0)
                .min_by_key(|(_, s)| s.last_used)
                .map(|(k, _)| *k);
            let Some(id) = victim else {
                return Err(ExpertError::TooBig(ExpertKey { layer: -1, expert: -1 }, need, self.capacity));
            };
            if let Some(slot) = self.slots.remove(&id) {
                self.resident = self.resident.saturating_sub(slot.bytes);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use colibri_st::write_shard;
    use std::env;

    fn fixture() -> std::path::PathBuf {
        let path = env::temp_dir().join("colibri-rs-experts.safetensors");
        let payload = |n: u8| vec![n; 8];
        let a = payload(1);
        let b = payload(2);
        let c = payload(3);
        let d = payload(4);
        let e = payload(5);
        let f = payload(6);
        write_shard(
            &path,
            &[
                ("layers.0.experts.0.gate", "U8", &[8], &a),
                ("layers.0.experts.0.up", "U8", &[8], &b),
                ("layers.0.experts.0.down", "U8", &[8], &c),
                ("layers.0.experts.1.gate", "U8", &[8], &d),
                ("layers.0.experts.1.up", "U8", &[8], &e),
                ("layers.0.experts.1.down", "U8", &[8], &f),
            ],
        )
        .unwrap();
        path
    }

    #[test]
    fn hit_miss_and_no_evict_while_leased() {
        let path = fixture();
        // Each expert is 24 bytes. Capacity fits exactly one.
        let mut store = ExpertStore::open(&path, 24).unwrap();
        let k0 = ExpertKey { layer: 0, expert: 0 };
        let k1 = ExpertKey { layer: 0, expert: 1 };
        let v = store.lookup(k0).unwrap();
        assert_eq!(v.gate, vec![1; 8]);
        assert_eq!(store.leases(k0), 1);

        let err = store.lookup(k1).unwrap_err();
        assert!(matches!(err, ExpertError::TooBig(_, _, _)));
        assert_eq!(store.leases(k0), 1);

        store.release(k0);
        assert_eq!(store.leases(k0), 0);
        let v1 = store.lookup(k1).unwrap();
        assert_eq!(v1.down, vec![6; 8]);
        assert_eq!(store.leases(k0), 0);
        assert_eq!(store.leases(k1), 1);

        let s = store.stats();
        // k0 miss, k1 miss that could not evict the lease, k1 miss after release.
        assert_eq!(s.misses, 3);
        assert_eq!(s.resident_bytes, 24);
        assert!(s.bytes_read >= 48);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn second_lookup_is_a_hit() {
        let path = fixture();
        let mut store = ExpertStore::open(&path, 128).unwrap();
        let k0 = ExpertKey { layer: 0, expert: 0 };
        store.lookup(k0).unwrap();
        store.lookup(k0).unwrap();
        let s = store.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
        assert_eq!(store.leases(k0), 2);
        let _ = std::fs::remove_file(path);
    }
}
