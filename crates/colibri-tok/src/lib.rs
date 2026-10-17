//! Tokenizer minimo per lo scaffold.
//!
//! In produzione: BPE/SentencePiece allineato al modello (come tok.h in Colibrì).

use colibri_core::{ColibriError, Result};

/// Tokenizer a spazi (solo per far compilare chat stub).
pub struct WhitespaceTokenizer;

impl WhitespaceTokenizer {
    pub fn new() -> Self {
        Self
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // ID fittizi: hash dei token
        Ok(text
            .split_whitespace()
            .map(|t| {
                let mut h = 0u32;
                for b in t.bytes() {
                    h = h.wrapping_mul(31).wrapping_add(b as u32);
                }
                h
            })
            .collect())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        // Non invertibile: stub
        Ok(format!("<{} tokens>", ids.len()))
    }
}

impl Default for WhitespaceTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait per tokenizer reali.
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, ids: &[u32]) -> Result<String>;
}

impl Tokenizer for WhitespaceTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        WhitespaceTokenizer::encode(self, text)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        WhitespaceTokenizer::decode(self, ids)
    }
}

// silenzia warning unused
#[allow(dead_code)]
fn _use_error() -> ColibriError {
    ColibriError::Tokenizer("stub".into())
}
