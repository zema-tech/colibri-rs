# Architettura colibri-rs

## Da Colibrì C a Rust

Colibrì: un file `.c` per famiglia + header condivisi (`st.h`, `expert_store.h`, `quant.h`, `tok.h`).

colibri-rs: crate per responsabilità, stesso flusso a runtime.

```
prompt
  → tok (colibri-tok)
  → embed + layers densità (in RAM, colibri-store + quant)
  → per layer MoE:
        router → ExpertId[]
        expert_cache.get_or_load (miss = lettura disco)
        FFN expert (colibri-quant)
  → lm_head → token
```

## Expert streaming

`colibri-expert::ExpertCache`:

- budget in byte (es. 4 GiB su macchina 16 GB)
- `get_or_load(id, file, loc)` → hit in RAM o read + insert
- eviction LRU grezza quando il budget è pieno

È il pezzo che permette modelli più grandi della RAM, come in Colibrì/ds4.

## Engine Qwen

`colibri-engine-qwen` implementa il trait `Engine`.  
Oggi: load path + generate stub.  
Domani: parser tensori, attention, router, expert FFN allineati a Qwen3.6-MoE.

## Cosa non fare

- Non copiare tutti gli engine (GLM, Kimi, …) prima di avere **un** forward corretto.
- Non aggiungere GPU finché il path CPU non è misurabile e giusto.
