# PyTorch pipeline (Linux / Windows / macOS · CPU / CUDA / MPS)

A self-contained, **cross-platform PyTorch** port of the Swahili LLM: build a
corpus from a HuggingFace dataset, train a GPT-style Transformer from scratch, and
generate text. The original MLX scripts in the repo root are untouched.

Everything here is **generic** — no machine-specific paths. Artifacts default to
`./data` inside this folder, and the dataset comes from **HuggingFace** (no local
corpus needed). Override anything with environment variables.

## Install
```bash
pip install -r requirements.txt
# For a GPU build of PyTorch, see https://pytorch.org/get-started/locally/
```

## 1. Build the corpus (HuggingFace → tokenizer → token binary)
Streams a HF text dataset, trains a 32k SentencePiece BPE tokenizer (with
`byte_fallback`, so no `<unk>`), and writes `data/train.bin` + `data/val.bin`:
```bash
python build_corpus.py
```
Configure via env (defaults shown):
```bash
HF_DATASET=Alfaxad/Inkuba-Mono-Swahili   # any HF text dataset
HF_SPLIT=train
TEXT_COLUMN=text
DATA_DIR=./data
TOK_SAMPLE=4000000                        # sentences sampled to train the tokenizer
```
The build is **resumable** — re-run it and it continues from `manifest.json`.

## 2. Train (resumable)
```bash
python train.py
```
Warmup + cosine LR, gradient clipping, AdamW(0.9, 0.95), weight decay 0.1.
A checkpoint is saved every 1,000 steps to `data/checkpoints/ckpt.pt`; rerun to
**resume from the exact step**. Key env vars:
```bash
DATA_DIR=./data  BATCH_SIZE=8  GRAD_ACCUM=4  BLOCK_SIZE=128
MAX_ITERS=600000  LEARNING_RATE=6e-4
```
Runs on CPU / CUDA / Apple MPS (auto-detected). On a small GPU, keep
`BATCH_SIZE`/`BLOCK_SIZE` modest and raise `GRAD_ACCUM` for a larger effective batch.

> Tip for long runs: launch under `nohup` or a systemd/Task-Scheduler service. Since
> training auto-resumes from the checkpoint, it survives restarts.

## 3. Generate
Runs sample prompts, then interactive input:
```bash
python generate.py
```
It's a **base/completion** model — give it the *start* of a sentence (e.g.
`"Leo asubuhi nilikwenda"`), not a question.

## Files
| File | Purpose |
|------|---------|
| `model.py` | Model definition (decoder-only Transformer) + load/generate helpers |
| `build_corpus.py` | HuggingFace dataset → tokenizer → `uint16` token binary (resumable) |
| `train.py` | Resumable trainer (memmap, nanoGPT schedule) |
| `generate.py` | Inference: sample prompts + interactive |
| `model_config.json` | Architecture (vocab 32k, 12 layers, 8 heads, hidden 512) |

## Architecture
Decoder-only Transformer (GPT-2 style): 12 layers, 8 heads (head dim 64), hidden
512, GELU FFN, pre-norm LayerNorm, learned positions, 32k BPE tokenizer
(`byte_fallback`). ~71M parameters at the default config.
