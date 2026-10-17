//! CLI minimale stile `coli` di Colibrì.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use colibri_core::{Engine, GenerateRequest, PlacementConfig};
use colibri_engine_qwen::QwenMoeEngine;

#[derive(Parser, Debug)]
#[command(name = "coli-rs", about = "colibri-rs — Motore MoE Rust (scaffold)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Info su build e placement di default
    Info,
    /// Chat stub (richiede path modello; forward non ancora implementato)
    Chat {
        /// Directory del modello / container
        #[arg(long)]
        model: PathBuf,
        /// Prompt iniziale (opzionale)
        #[arg(long)]
        prompt: Option<String>,
        /// Budget cache expert in MiB
        #[arg(long, default_value_t = 4096)]
        expert_cache_mib: u64,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Info => {
            let cfg = PlacementConfig::default();
            println!("colibri-rs 0.1.0 (scaffold)");
            println!("  ispirato a JustVugg/colibri + antirez/ds4");
            println!("  target: Qwen MoE ~27-35B su 16GB RAM");
            println!();
            println!("Placement default:");
            println!(
                "  expert_cache: {} MiB",
                cfg.expert_cache_bytes / (1024 * 1024)
            );
            println!("  max_context:  {}", cfg.max_context);
            println!("  threads:      {}", cfg.threads);
            println!();
            println!("Comandi:");
            println!("  coli-rs info");
            println!("  coli-rs chat --model /path/to/model [--prompt '...']");
            println!();
            println!("Stato: forward MoE non implementato — solo architettura e cache expert.");
        }
        Cmd::Chat {
            model,
            prompt,
            expert_cache_mib,
        } => {
            let cfg = PlacementConfig {
                expert_cache_bytes: expert_cache_mib * 1024 * 1024,
                ..PlacementConfig::default()
            };
            let mut engine = QwenMoeEngine::load(model, cfg)?;
            let req = GenerateRequest {
                prompt: prompt.unwrap_or_else(|| "Ciao".into()),
                max_tokens: 64,
                temperature: 0.7,
            };
            let out = engine.generate(&req)?;
            println!("{}", out);
            let st = engine.stats();
            println!(
                "\n[stats] expert cache used={} B  hits={} misses={} hit_rate={:.1}%",
                st.expert_cache_bytes,
                st.expert_hits,
                st.expert_misses,
                st.hit_rate() * 100.0
            );
        }
    }
    Ok(())
}
