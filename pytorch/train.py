"""
Resumable training for the PyTorch Swahili LLM.

Reads the uint16 token binaries from build_corpus.py via np.memmap and trains with
a nanoGPT-style schedule (warmup + cosine decay, gradient clipping, AdamW betas
0.9/0.95, weight decay 0.1). A full checkpoint (model + optimizer + step + RNG) is
saved every CKPT_INTERVAL steps to DATA_DIR/checkpoints/ckpt.pt; on startup it
resumes from there — so the run survives interruptions and reboots.

All paths/hyperparameters are environment variables (no machine-specific paths):

    DATA_DIR        where train.bin/val.bin live   (default: ./data)
    BATCH_SIZE      micro-batch                     (default: 8)
    GRAD_ACCUM      gradient accumulation steps     (default: 4)
    BLOCK_SIZE      context length per step         (default: 128)
    MAX_ITERS       total optimizer steps           (default: 600000)
    LEARNING_RATE / MIN_LR / WARMUP_ITERS / WEIGHT_DECAY / GRAD_CLIP
    EVAL_INTERVAL / EVAL_ITERS / CKPT_INTERVAL / LOG_INTERVAL / SEED

Usage:  python train.py
"""
import os
import math
import time
import json
import numpy as np
import torch
import torch.nn.functional as F

from model import KiswahiliLLM, load_config, pick_device

HERE = os.path.dirname(os.path.abspath(__file__))


def _i(name, d): return int(os.environ.get(name, d))
def _f(name, d): return float(os.environ.get(name, d))


cfg = load_config()
VOCAB_SIZE = cfg["vocab_size"]

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(HERE, "data"))
CKPT_DIR = os.path.join(DATA_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN = os.path.join(DATA_DIR, "val.bin")
CKPT_PATH = os.path.join(CKPT_DIR, "ckpt.pt")
BEST_PATH = os.path.join(CKPT_DIR, "ckpt_best.pt")
FINAL_PATH = os.path.join(DATA_DIR, "model_final.pt")

BATCH_SIZE = _i("BATCH_SIZE", 8)
GRAD_ACCUM = _i("GRAD_ACCUM", 4)
BLOCK_SIZE = _i("BLOCK_SIZE", 128)
MAX_ITERS = _i("MAX_ITERS", 600_000)
WARMUP_ITERS = _i("WARMUP_ITERS", 2000)
LR_DECAY_ITERS = _i("LR_DECAY_ITERS", MAX_ITERS)
LEARNING_RATE = _f("LEARNING_RATE", 6e-4)
MIN_LR = _f("MIN_LR", 6e-5)
WEIGHT_DECAY = _f("WEIGHT_DECAY", 0.1)
BETA1, BETA2 = _f("BETA1", 0.9), _f("BETA2", 0.95)
GRAD_CLIP = _f("GRAD_CLIP", 1.0)
EVAL_INTERVAL = _i("EVAL_INTERVAL", 1000)
EVAL_ITERS = _i("EVAL_ITERS", 100)
LOG_INTERVAL = _i("LOG_INTERVAL", 20)
CKPT_INTERVAL = _i("CKPT_INTERVAL", 1000)
SEED = _i("SEED", 1337)

DEVICE = pick_device()
USE_AMP = os.environ.get("USE_AMP", "0") == "1" and DEVICE.type == "cuda"

_CACHE, _CALLS = {}, {}


def get_batch(split):
    path = TRAIN_BIN if split == "train" else VAL_BIN
    refresh = path not in _CACHE
    if not refresh:
        _CALLS[path] = _CALLS.get(path, 0) + 1
        if _CALLS[path] % 500 == 0 and os.path.getsize(path) // 2 > len(_CACHE[path]) + 1_000_000:
            refresh = True
    if refresh:
        _CACHE[path] = np.memmap(path, dtype=np.uint16, mode="r"); _CALLS[path] = 0
    data = _CACHE[path]
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i:i + BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + BLOCK_SIZE].astype(np.int64)) for i in ix])
    if DEVICE.type == "cuda":
        return x.pin_memory().to(DEVICE, non_blocking=True), y.pin_memory().to(DEVICE, non_blocking=True)
    return x.to(DEVICE), y.to(DEVICE)


def get_lr(it):
    if it < WARMUP_ITERS:
        return LEARNING_RATE * (it + 1) / (WARMUP_ITERS + 1)
    if it > LR_DECAY_ITERS:
        return MIN_LR
    r = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    return MIN_LR + 0.5 * (1 + math.cos(math.pi * r)) * (LEARNING_RATE - MIN_LR)


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        if split == "val" and not os.path.exists(VAL_BIN):
            continue
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split)
            losses[k] = F.cross_entropy(model(x).reshape(-1, VOCAB_SIZE), y.reshape(-1)).item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_ckpt(path, model, opt, it, best):
    ckpt = {"model": model.state_dict(), "optimizer": opt.state_dict(), "iter_num": it,
            "best_val_loss": best, "config": cfg, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if DEVICE.type == "cuda" else None,
            "numpy_rng": np.random.get_state()}
    tmp = path + ".tmp"; torch.save(ckpt, tmp); os.replace(tmp, path)


def main():
    print(f"Device: {DEVICE} | AMP: {USE_AMP} | DATA_DIR: {DATA_DIR}")
    if not os.path.exists(TRAIN_BIN):
        raise FileNotFoundError(f"{TRAIN_BIN} not found — run build_corpus.py first.")
    print(f"train tokens: {os.path.getsize(TRAIN_BIN)//2:,} | vocab: {VOCAB_SIZE}")
    torch.manual_seed(SEED); np.random.seed(SEED)

    model = KiswahiliLLM(cfg).to(DEVICE)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    it, best = 0, float("inf")
    if os.path.exists(CKPT_PATH):
        ck = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        it = ck["iter_num"]; best = ck.get("best_val_loss", float("inf"))
        try:
            torch.set_rng_state(ck["torch_rng"].cpu())
            if ck.get("numpy_rng"):
                np.random.set_state(ck["numpy_rng"])
            if ck.get("cuda_rng") and DEVICE.type == "cuda":
                torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
        except Exception as e:
            print(f"(rng restore skipped: {e})")
        print(f"Resumed at iter {it}, best_val {best:.4f}")
    else:
        print("No checkpoint — starting fresh.")

    model.train()
    t0, running, n = time.time(), 0.0, 0
    while it < MAX_ITERS:
        lr = get_lr(it)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(GRAD_ACCUM):
            x, y = get_batch("train")
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                loss = F.cross_entropy(model(x).reshape(-1, VOCAB_SIZE), y.reshape(-1)) / GRAD_ACCUM
            scaler.scale(loss).backward()
            acc += loss.item()
        if GRAD_CLIP > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt); scaler.update()
        running += acc; n += 1; it += 1

        if it % LOG_INTERVAL == 0:
            dt = time.time() - t0
            tps = LOG_INTERVAL * BATCH_SIZE * GRAD_ACCUM * BLOCK_SIZE / dt
            print(f"iter {it} | loss {running/n:.4f} | lr {lr:.2e} | {tps/1e3:.1f}k tok/s", flush=True)
            running, n, t0 = 0.0, 0, time.time()

        if it % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            print(f"[eval] iter {it} | " + " | ".join(f"{k} {v:.4f}" for k, v in losses.items()), flush=True)
            if "val" in losses and losses["val"] < best:
                best = losses["val"]; save_ckpt(BEST_PATH, model, opt, it, best)
                print(f"[eval] new best val {best:.4f}", flush=True)
            t0 = time.time()

        if it % CKPT_INTERVAL == 0:
            save_ckpt(CKPT_PATH, model, opt, it, best)
            torch.save(model.state_dict(), FINAL_PATH)

    save_ckpt(CKPT_PATH, model, opt, it, best)
    torch.save(model.state_dict(), FINAL_PATH)
    print(f"Training complete at iter {it}. Final weights -> {FINAL_PATH}")


if __name__ == "__main__":
    main()
