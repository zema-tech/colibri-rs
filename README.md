# colibri-rs

**Riscrittura in Rust** dello spirito di [Colibrì](https://github.com/JustVugg/colibri) e [ds4](https://github.com/antirez/ds4):

> Motore MoE minimale · expert streamati da disco · parte densa in RAM · zero runtime pesanti

**Stato:** scaffold / work in progress.  
Non è ancora un engine completo: è l’architettura e i pezzi base su cui costruire.

Target hardware di riferimento: **16 GB RAM · 4 core · 50 GB disco** (Qwen MoE ~23–35B).

---

## Perché Rust (e non un port 1:1)

Colibrì upstream è pure C, un `.c` per famiglia di modello + header condivisi.  
Qui:

- stessa **filosofia** (streaming expert, engine specializzati, disco come memoria)
- safety + toolchain Rust
- crate separate al posto dei monolitici `.c`
- un engine alla volta (prima Qwen3.6-style MoE)

Una riscrittura completa di tutti gli engine (GLM 744B, DeepSeek V4, Kimi, …) è un progetto di mesi. Questo repo parte dal nucleo riusabile e da un engine Qwen-MoE.

---

## Architettura (mappa da Colibrì)

| Colibrì (C) | colibri-rs |
|-------------|------------|
| `st.h` safetensors / container | `colibri-store` |
| `expert_store.h` cache + stream | `colibri-expert` |
| `quant.h` / `idot.h` kernel | `colibri-quant` |
| `tok.h` tokenizer | `colibri-tok` |
| `qwen36.c` / `olmoe.c` … | `colibri-engine-qwen` (poi altri) |
| `coli` CLI | `coli-rs` binary |

```
colibri-rs/
├── Cargo.toml                 # workspace
├── crates/
│   ├── colibri-core/          # tipi comuni, errori, config
│   ├── colibri-store/         # lettura pesi da file (mmap / pread)
│   ├── colibri-expert/        # cache LRU + stream expert da disco
│   ├── colibri-quant/         # dequant / matmul int4-int8 (stub)
│   ├── colibri-tok/           # tokenizer (stub BPE)
│   ├── colibri-engine-qwen/   # forward MoE stile Qwen3.6 (stub)
│   └── coli-rs/               # CLI: chat / info / serve (minimo)
└── docs/
```

---

## Principio operativo (come Colibrì)

1. **Dense** (attention, shared, embed) → residente in RAM  
2. **Routed experts** → su disco, letti on-demand  
3. **Cache** per-layer (budget in byte) con eviction  
4. Il router decide quali expert servono → miss = I/O da SSD  

Su 16 GB / 50 GB il modello realistico è **Qwen3.6-35B-A3B** (~23 GB container), non GLM/DeepSeek full.

---

## Build

```bash
git clone https://github.com/zema-tech/colibri-rs.git
cd colibri-rs
cargo build --release
```

CLI (quando i pezzi saranno collegati):

```bash
cargo run -p coli-rs -- info
cargo run -p coli-rs -- chat --model /path/to/container
```

---

## Roadmap onesta

| Fase | Contenuto |
|------|-----------|
| **0** (ora) | Workspace, trait `Engine`, expert cache, store mmap, CLI stub |
| **1** | Lettura container safetensors/GGUF minima, tokenizer reale |
| **2** | Kernel quant CPU (int4/int8 matvec) |
| **3** | Forward MoE Qwen-like (router + expert FFN + residual) |
| **4** | `coli chat` usable su modello piccolo (OLMoE / Qwen3.6) |
| **5** | API OpenAI-compatible, prefetch expert, metriche |

Non promettiamo parità con Colibrì C nel breve. L’obiettivo è un motore **corretto e misurabile** sulla fascia 16 GB.

---

## Licenza

Apache-2.0 (allineata a Colibrì).  
I pesi dei modelli restano sotto le licenze originali degli autori.

---

## Riferimenti

- [JustVugg/colibri](https://github.com/JustVugg/colibri) — design e idea originale  
- [antirez/ds4](https://github.com/antirez/ds4) — motore C specializzato + SSD streaming  
