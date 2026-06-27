"""
Generate Swahili text with the trained model.

Runs a few built-in sample prompts first, then hands over to interactive input.
This is a BASE / completion model: give it the START of a sentence and it continues.

Paths are environment-configurable (no machine-specific hardcoding):
    DATA_DIR   where the tokenizer + checkpoints live   (default: ./data)
    DEVICE     cpu | cuda | mps                          (default: auto)

Usage:  python generate.py
"""
import os
import torch
import sentencepiece as spm

from model import load_model, load_config, generate, pick_device

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(HERE, "data"))
DEVICE = os.environ.get("DEVICE") or str(pick_device())

TOKENIZER = os.path.join(DATA_DIR, "tokenizer.model")
CANDIDATES = [
    os.path.join(DATA_DIR, "checkpoints", "ckpt_best.pt"),
    os.path.join(DATA_DIR, "checkpoints", "ckpt.pt"),
    os.path.join(DATA_DIR, "model_final.pt"),
]

SAMPLE_PROMPTS = [
    "Habari za leo, leo nataka kuzungumza kuhusu",
    "Serikali ya Tanzania imetangaza kuwa",
    "Elimu ni muhimu kwa sababu",
    "Mpira wa miguu ni mchezo",
]


def main():
    weights = next((p for p in CANDIDATES if os.path.exists(p)), None)
    if weights is None:
        raise SystemExit(
            f"No checkpoint found in {DATA_DIR}. Train first (python train.py); the "
            "first checkpoint is written at iter 1000."
        )
    if not os.path.exists(TOKENIZER):
        raise SystemExit(f"Tokenizer not found: {TOKENIZER} (run build_corpus.py first)")

    cfg = load_config()
    model = load_model(weights, cfg, device=DEVICE)
    sp = spm.SentencePieceProcessor(); sp.load(TOKENIZER)
    print(f"Loaded {os.path.basename(weights)} "
          f"({sum(p.numel() for p in model.parameters())/1e6:.1f}M params) on {DEVICE}")
    print("=" * 70)

    # 1) Built-in sample prompts.
    print("Running default sample prompts...\n")
    for p in SAMPLE_PROMPTS:
        print(f"PROMPT : {p}\nOUTPUT : {generate(model, sp, p, device=DEVICE)}\n" + "-" * 70)

    # 2) Interactive.
    print("\nType the START of a sentence and it continues. 'exit' to quit.\n")
    while True:
        try:
            prompt = input("Wewe > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKwaheri!"); break
        if not prompt or prompt.lower() in ("exit", "quit", "toka"):
            print("Kwaheri!"); break
        print(f"SwahiliGPT > {generate(model, sp, prompt, device=DEVICE)}\n" + "-" * 70)


if __name__ == "__main__":
    main()
