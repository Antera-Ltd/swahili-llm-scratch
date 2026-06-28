"""
Chat with the instruction fine-tuned Swahili model (from finetune.py).

Wraps your message in the same template build_instruct.py used, streams the
response token-by-token, and shows only the answer. Runs a few sample
instructions first, then hands over to interactive input.

Paths are environment-configurable (no machine-specific hardcoding):
    DATA_DIR       where tokenizer + instruct checkpoint live  (default: ./data)
    INSTRUCT_CKPT  the fine-tuned weights         (default: DATA_DIR/instruct/instruct_final.pt)
    DEVICE         cpu | cuda | mps                             (default: auto)

Usage:  python chat.py
"""
import os
import sys

import torch
import torch.nn.functional as F
import sentencepiece as spm

from model import KiswahiliLLM, load_config, pick_device

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(HERE, "data"))
TOKENIZER = os.environ.get("TOKENIZER", os.path.join(DATA_DIR, "tokenizer.model"))
CKPT = os.environ.get("INSTRUCT_CKPT", os.path.join(DATA_DIR, "instruct", "instruct_final.pt"))
DEVICE = os.environ.get("DEVICE") or str(pick_device())

MAX_NEW, TEMP, TOP_K, TOP_P = 200, 0.7, 40, 0.9

T_INSTR = "### Maagizo:\n{instruction}\n"
T_INPUT = "### Ingizo:\n{input}\n"
T_ANSWER = "### Jibu:\n"


def build_prompt(instruction, inp=""):
    p = T_INSTR.format(instruction=instruction)
    if inp:
        p += T_INPUT.format(input=inp)
    return p + T_ANSWER


@torch.no_grad()
def generate(model, sp, prompt, stream=True):
    ids = torch.tensor(sp.encode(prompt), dtype=torch.long, device=DEVICE)[None, :]
    gen_ids, printed, text = [], 0, ""
    for _ in range(MAX_NEW):
        logits = model(ids)[:, -1, :] / TEMP
        if TOP_K:
            kth = torch.topk(logits, min(TOP_K, logits.shape[-1])).values[:, -1, None]
            logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
        sl, si = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
        rm = cum > TOP_P
        rm[:, 1:] = rm[:, :-1].clone(); rm[:, 0] = False
        sl[rm] = float("-inf")
        logits = sl.gather(-1, si.argsort(-1))
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        tok = nxt.item()
        ids = torch.cat([ids, nxt], dim=1)
        if tok == sp.eos_id():
            break
        gen_ids.append(tok)
        # Re-decode the whole response each step (SentencePiece needs full
        # context for spacing) and stream only the newly-revealed text.
        text = sp.decode(gen_ids).replace("⁇", "")
        cut = len(text)
        for s in ("### Maagizo:", "### Ingizo:"):
            j = text.find(s)
            if j != -1:
                cut = min(cut, j)
        visible = text[:cut]
        if stream and len(visible) > printed:
            sys.stdout.write(visible[printed:]); sys.stdout.flush()
            printed = len(visible)
        if cut < len(text):
            text = visible
            break
    if stream:
        sys.stdout.write("\n"); sys.stdout.flush()
    return text.strip()


def main():
    if not os.path.exists(CKPT):
        sys.exit(f"No instruct checkpoint at {CKPT}\nRun finetune.py first.")
    if not os.path.exists(TOKENIZER):
        sys.exit(f"Tokenizer not found: {TOKENIZER} (run build_corpus.py first)")
    sp = spm.SentencePieceProcessor(); sp.load(TOKENIZER)

    model = KiswahiliLLM(load_config()).to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded instruct model: {CKPT} on {DEVICE}\n" + "=" * 70)

    demos = [
        "Nini mji mkuu wa Tanzania?",
        "Andika sentensi fupi kuhusu umuhimu wa elimu.",
        "Eleza maana ya neno 'ujamaa'.",
    ]
    print("Mifano (sample instructions):\n")
    for d in demos:
        print(f"Maagizo : {d}")
        print("Jibu    : ", end="", flush=True)
        generate(model, sp, build_prompt(d))
        print("-" * 70)

    print("\nUliza chochote (type your instruction). 'toka' to quit.\n")
    while True:
        try:
            msg = input("Wewe > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKwaheri!"); break
        if not msg or msg.lower() in ("exit", "quit", "toka"):
            print("Kwaheri!"); break
        print("Swahili > ", end="", flush=True)
        generate(model, sp, build_prompt(msg))
        print("-" * 70)


if __name__ == "__main__":
    main()
