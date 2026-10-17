# colibri-rs

Incremental Rust port of [Colibrì](https://github.com/JustVugg/colibri).

This is not a full engine. The first slice is the part every model family shares:

| C (Colibrì) | Rust |
|---|---|
| `c/st.h` header + tensor reads | `colibri-st` |
| `c/expert_store.h` lease + cache | `colibri-expert` |

Experts stay on disk inside a safetensors shard. A lookup copies `gate` / `up` / `down` into a byte-budget cache. A slot with an active lease is never evicted. `prefetch` fills the cache without taking a lease.

Tensor names, for now:

```text
layers.{L}.experts.{E}.gate
layers.{L}.experts.{E}.up
layers.{L}.experts.{E}.down
```

```bash
cargo test
```

Not ported yet: quant kernels, tokenizer, a real Qwen/OLMoE forward, GPU tiers.
