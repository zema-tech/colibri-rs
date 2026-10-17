# Roadmap

1. **Parser pesi** — safetensors o container Colibrì; indice `TensorLoc` reale.
2. **Tokenizer** — BPE/Unigram del modello Qwen (vocab + merges).
3. **Kernel quant** — int4 group 64 / int8 matvec CPU (rayon).
4. **Forward layer** — RMSNorm, attention (o DeltaNet se serve), router top-k, expert FFN, residual.
5. **Prefetch** — leggere gli expert del layer N+1 mentre si calcola N.
6. **CLI chat** interattiva + `serve` HTTP OpenAI-compatible.
7. **Test** — oracle su modello tiny-random (come fa Colibrì).

Priorità: correttezza su un MoE piccolo (OLMoE / Qwen3.6 quant) prima della velocità.
