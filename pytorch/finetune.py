"""
Instruction fine-tuning (SFT) of the base Swahili model.

Continues training from the base checkpoint on the instruction dataset built by
build_instruct.py, with two differences from pre-training:
  1. LOSS MASKING — only response tokens contribute to the loss; the prompt is
     context only (positions before prompt_len are ignored).
  2. A short, low-LR cosine schedule so the model learns to follow instructions
     without washing out what it learned in pre-training.

GRADIENT CHECKPOINTING lets this run full-length (512-token) sequences on small
GPUs (e.g. 4GB). The base run is untouched: this reads the base checkpoint
read-only and writes its own to DATA_DIR/instruct/.

RESUMABLE: a full checkpoint (model + optimizer + step + RNG) is saved every
CKPT_EVERY steps; on startup it resumes from there, so the run survives reboots.

All paths/hyperparameters are environment variables (no machine-specific paths):

    DATA_DIR     where instruct/ + checkpoints live   (default: ./data)
    BASE_CKPT    base weights to start from            (default: ckpt_best.pt -> model_final.pt)
    EPOCHS       passes over the data                  (default: 3)
    LR / MIN_LR  peak / floor learning rate            (default: 2e-5 / 2e-6)
    BATCH_SIZE / GRAD_ACCUM                            (default: 4 / 8 -> eff. 32)
    FT_MAX_LEN   cap sequence length to fit VRAM       (default: 512)
    WARMUP_FRAC / WEIGHT_DECAY / GRAD_CLIP / CKPT_EVERY / EVAL_EVERY / SEED

Usage:  python finetune.py
"""
import os
import math
import time
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint

from model import KiswahiliLLM, load_config, pick_device, create_causal_mask

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(HERE, "data"))
INSTRUCT_DIR = os.path.join(DATA_DIR, "instruct")
CKPT_DIR = os.path.join(DATA_DIR, "checkpoints")

cfg = load_config()
VOCAB_SIZE = cfg["vocab_size"]
DEVICE = pick_device()

EPOCHS = int(os.environ.get("EPOCHS", 3))
LR = float(os.environ.get("LR", 2e-5))
MIN_LR = float(os.environ.get("MIN_LR", 2e-6))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", 8))
FT_MAX_LEN = int(os.environ.get("FT_MAX_LEN", 512))
WARMUP_FRAC = float(os.environ.get("WARMUP_FRAC", 0.03))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.1))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", 1.0))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", 100))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 500))
SEED = int(os.environ.get("SEED", 1337))
PAD_ID, IGNORE = 0, -100

OUT_CKPT = os.path.join(INSTRUCT_DIR, "ckpt_instruct.pt")
OUT_FINAL = os.path.join(INSTRUCT_DIR, "instruct_final.pt")


def find_base_ckpt():
    env = os.environ.get("BASE_CKPT")
    if env:
        return env
    for p in (os.path.join(CKPT_DIR, "ckpt_best.pt"),
              os.path.join(CKPT_DIR, "ckpt.pt"),
              os.path.join(DATA_DIR, "model_final.pt")):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No base checkpoint found to fine-tune from.")


def forward_checkpointed(model, input_ids):
    """Forward that recomputes each block in backward — flat activation memory
    in the number of layers, so full-length sequences fit on a small GPU."""
    T = input_ids.shape[1]
    pos = torch.arange(T, device=input_ids.device)[None, :]
    x = model.embedding(input_ids) + model.pos_embedding(pos)
    mask = create_causal_mask(T, input_ids.device)
    for layer in model.layers:
        x = checkpoint(layer, x, mask, use_reentrant=False)
    return model.output(model.norm(x))


class InstructDataset(Dataset):
    def __init__(self, path, max_len=FT_MAX_LEN):
        raw = torch.load(path)
        self.examples = []
        for ex in raw:
            ids = ex["ids"][:max_len]
            if ex["prompt_len"] >= len(ids):   # nothing to learn -> skip
                continue
            self.examples.append({"ids": ids, "prompt_len": ex["prompt_len"]})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        ids, p_len = ex["ids"], ex["prompt_len"]
        x = ids[:-1]
        y = ids[1:]
        # learn only response tokens: y[j] predicts ids[j+1]; mask while prompt.
        labels = [t if (j + 1) >= p_len else IGNORE for j, t in enumerate(y)]
        return torch.tensor(x, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate(batch):
    maxlen = max(len(x) for x, _ in batch)
    xs, ys = [], []
    for x, y in batch:
        pad = maxlen - len(x)
        xs.append(F.pad(x, (0, pad), value=PAD_ID))
        ys.append(F.pad(y, (0, pad), value=IGNORE))
    return torch.stack(xs), torch.stack(ys)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB_SIZE), y.reshape(-1),
                               ignore_index=IGNORE)
        tot += loss.item(); n += 1
    model.train()
    return tot / max(n, 1)


def save_resume(model, optimizer, global_step, total_steps):
    ckpt = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "global_step": global_step, "total_steps": total_steps,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if DEVICE.type == "cuda" else None,
    }
    tmp = OUT_CKPT + ".tmp"; torch.save(ckpt, tmp); os.replace(tmp, OUT_CKPT)
    ftmp = OUT_FINAL + ".tmp"; torch.save(model.state_dict(), ftmp); os.replace(ftmp, OUT_FINAL)


def main():
    torch.manual_seed(SEED); random.seed(SEED)
    train_path = os.path.join(INSTRUCT_DIR, "train.pt")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} not found — run build_instruct.py first.")

    train_ds = InstructDataset(train_path)
    val_path = os.path.join(INSTRUCT_DIR, "val.pt")
    val_ds = InstructDataset(val_path) if os.path.exists(val_path) else None
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate, drop_last=True)
    val_loader = (DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate)
                  if val_ds else None)

    base = find_base_ckpt()
    print(f"Device: {DEVICE} | base: {base}")
    print(f"Train examples: {len(train_ds):,} | val: {len(val_ds) if val_ds else 0:,}")

    model = KiswahiliLLM(cfg).to(DEVICE)
    state = torch.load(base, map_location=DEVICE)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                                  weight_decay=WEIGHT_DECAY)

    steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM)
    total_steps = steps_per_epoch * EPOCHS
    warmup = max(1, int(total_steps * WARMUP_FRAC))

    def lr_at(step):
        if step < warmup:
            return LR * (step + 1) / (warmup + 1)
        ratio = (step - warmup) / max(1, total_steps - warmup)
        return MIN_LR + 0.5 * (1 + math.cos(math.pi * min(1.0, ratio))) * (LR - MIN_LR)

    # ---- resume ----
    global_step = 0
    if os.path.exists(OUT_CKPT):
        ck = torch.load(OUT_CKPT, map_location=DEVICE)
        if "optimizer" in ck and "global_step" in ck:
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            global_step = ck["global_step"]
            try:
                torch.set_rng_state(ck["torch_rng"].cpu())
                if ck.get("cuda_rng") is not None and DEVICE.type == "cuda":
                    torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
            except Exception as e:
                print(f"(rng restore skipped: {e})")
            print(f"RESUMING from step {global_step}/{total_steps}")
    print(f"Steps/epoch: {steps_per_epoch} | total: {total_steps} | warmup: {warmup}")

    if global_step >= total_steps:
        print("Already complete — nothing to do.")
        torch.save(model.state_dict(), OUT_FINAL)
        return

    def infinite(loader):
        while True:
            for b in loader:
                yield b

    data_iter = infinite(train_loader)
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    while global_step < total_steps:
        running = 0.0
        for _ in range(GRAD_ACCUM):
            x, y = next(data_iter)
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = forward_checkpointed(model, x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1),
                                   ignore_index=IGNORE) / GRAD_ACCUM
            loss.backward()
            running += loss.item()

        lr = lr_at(global_step)
        for g in optimizer.param_groups:
            g["lr"] = lr
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if global_step % 20 == 0:
            dt = time.time() - t0
            print(f"step {global_step}/{total_steps} "
                  f"(epoch {global_step/steps_per_epoch:.2f}) | loss {running:.4f} "
                  f"| lr {lr:.2e} | {20/dt:.2f} steps/s", flush=True)
            t0 = time.time()
        if global_step % EVAL_EVERY == 0 and val_loader is not None:
            print(f"[eval] step {global_step} | val loss {evaluate(model, val_loader):.4f}",
                  flush=True)
            t0 = time.time()
        if global_step % CKPT_EVERY == 0:
            save_resume(model, optimizer, global_step, total_steps)

    save_resume(model, optimizer, global_step, total_steps)
    print(f"\nFine-tuning complete ({global_step} steps). Model -> {OUT_FINAL}")
    print("Chat with it:  python chat.py")


if __name__ == "__main__":
    main()
