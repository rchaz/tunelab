# LoRA vs QLoRA (and why your Mac can train an 8B model)

**Full fine-tuning** updates every weight in the model. For an 8B model that
means holding the weights, their gradients, and optimizer state in memory —
tens of GB. Not laptop territory.

**LoRA** (Low-Rank Adaptation) freezes the base model entirely and trains
small *adapter* matrices alongside a subset of layers. The bet: the change a
narrow task requires is low-rank — expressible in far fewer numbers than the
full weight matrix. Instead of updating a 4096×4096 matrix (16.7M numbers),
LoRA learns two thin matrices of rank r (say 16): 4096×16 + 16×4096 ≈ 131K
numbers — about 0.8% of the original. Only adapters get gradients and
optimizer state, so training memory collapses. The artifact is a few MB of
adapter weights you can fuse into the base later or keep separate.

**QLoRA** = LoRA with the frozen base model *quantized* to 4-bit. The base
weights are already frozen — nobody is updating them — so storing them at
reduced precision costs little, and they shrink 4×. The adapters themselves
stay fp16, so the part that's actually learning keeps full precision.

## The memory math (why 8B fits in 16GB)

```
Qwen3-8B at 4-bit:        ~4.6 GB   (frozen, quantized base)
LoRA adapters (fp16):     ~10s of MB
gradients + optimizer:     adapter-sized, not model-sized
working memory:           ~2–4 GB
                          ─────────
                          fits a 16GB Mac
```

The same model in fp16 would need ~16GB for weights alone before training
started.

## How this maps to mlx-lm

mlx-lm doesn't have a QLoRA switch — the distinction is implicit in the
checkpoint you load:

- Load a 4-bit checkpoint (`mlx-community/...-4bit`) → you're doing **QLoRA**.
  This is tunelab's default.
- Load an fp16 model → plain **LoRA**. Slightly better quality ceiling,
  ~4× the memory.

Two dials worth knowing: `--num-layers` (how many layers get adapters,
default 16 — fewer means less capacity, useful against overfitting on tiny
datasets) and the rank (default 8). Rank lives in a YAML config file
(`lora_parameters`), not a CLI flag — tunelab generates that file when a run
tunes rank. Higher rank = more capacity = more to overfit with; the
rank-sweep experiment in research mode makes that trade visible.

One honest cost: quantizing the base loses a little quality versus fp16, and a
QLoRA-fused model can't export straight to GGUF without dequantizing first.
For the narrow tasks tunelab targets, the eval — tuned vs base vs teacher —
tells you whether the loss mattered, instead of guessing.
