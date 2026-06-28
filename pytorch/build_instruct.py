"""
Build a Swahili instruction-tuning dataset from open HuggingFace sources.

Pulls instruction/response data straight from the Hub, normalises it to one
prompt/response template, OVERSAMPLES the small native set so it isn't drowned
out by larger translated sets, tokenises with the tokenizer built by
build_corpus.py, and writes:

    DATA_DIR/instruct/train.pt , val.pt   tokenised examples {"ids", "prompt_len"}
    DATA_DIR/instruct/instruct.jsonl      human-readable rows
    DATA_DIR/instruct/meta.json           counts + provenance

Only the RESPONSE tokens are learned during fine-tuning; prompt_len marks where
the prompt ends so finetune.py can mask it.

All paths/sources are environment variables (no machine-specific hardcoding):

    DATA_DIR     where tokenizer.model lives + output goes  (default: ./data)
    OVERSAMPLE   times to repeat the native (Aya) set        (default: 5)
    MAX_LEN      truncate examples to this many tokens        (default: 512)
    VAL_FRAC     fraction held out for validation             (default: 0.01)
    SEED         shuffle seed                                 (default: 1337)

Sources (all from HuggingFace; credit them if you publish):
    - CohereLabs/aya_dataset             (Swahili subset, native human-written)
    - MBZUAI/Bactrian-X                  (sw split, instruction-translated)
    - NabajyotiPathak/kiswahili-ai-blended  (FineTome-sw + KenSwQuAD +
                                            Code-170k-sw + Swahili-Corpus)

Usage:  python build_instruct.py
"""
import os
import gzip
import json
import random

import torch
import sentencepiece as spm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(HERE, "data"))
OUT_DIR = os.path.join(DATA_DIR, "instruct")
os.makedirs(OUT_DIR, exist_ok=True)

TOKENIZER = os.environ.get("TOKENIZER", os.path.join(DATA_DIR, "tokenizer.model"))
OVERSAMPLE = int(os.environ.get("OVERSAMPLE", 5))
MAX_LEN = int(os.environ.get("MAX_LEN", 512))
VAL_FRAC = float(os.environ.get("VAL_FRAC", 0.01))
SEED = int(os.environ.get("SEED", 1337))

# Swahili prompt template. The prompt is everything up to and including
# "### Jibu:\n"; the model is trained to produce what comes after.
T_INSTR = "### Maagizo:\n{instruction}\n"
T_INPUT = "### Ingizo:\n{input}\n"
T_ANSWER = "### Jibu:\n"


def load_aya_swahili():
    from datasets import load_dataset
    last = None
    for repo in ("CohereLabs/aya_dataset", "CohereForAI/aya_dataset"):
        try:
            ds = load_dataset(repo, split="train")
            break
        except Exception as e:
            last = e
    else:
        raise RuntimeError(f"could not load Aya dataset: {last}")
    rows = []
    for r in ds:
        if r.get("language") != "Swahili":
            continue
        instr = (r.get("inputs") or "").strip()
        out = (r.get("targets") or "").strip()
        if instr and out:
            rows.append({"instruction": instr, "input": "", "output": out})
    print(f"  Aya (Swahili, native): {len(rows)} rows")
    return rows


def load_bactrian_swahili():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("MBZUAI/Bactrian-X", "data/sw.json.gz", repo_type="dataset")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for r in data:
        instr = (r.get("instruction") or "").strip()
        inp = (r.get("input") or "").strip()
        out = (r.get("output") or "").strip()
        if instr and out:
            rows.append({"instruction": instr, "input": inp, "output": out})
    print(f"  Bactrian-X (sw, translated): {len(rows)} rows")
    return rows


def load_blended_swahili():
    """Uses the Adaptive-Data ENHANCED fields (quality-improved), falling back
    to the raw instruction/response when an enhanced field is missing."""
    from datasets import load_dataset
    ds = load_dataset("NabajyotiPathak/kiswahili-ai-blended", split="train")
    rows = []
    for r in ds:
        instr = (r.get("enhanced_prompt") or r.get("instruction") or "").strip()
        out = (r.get("enhanced_completion") or r.get("response") or "").strip()
        if instr and out:
            rows.append({"instruction": instr, "input": "", "output": out})
    print(f"  kiswahili-ai-blended (enhanced): {len(rows)} rows")
    return rows


def render(row):
    prompt = T_INSTR.format(instruction=row["instruction"])
    if row["input"]:
        prompt += T_INPUT.format(input=row["input"])
    prompt += T_ANSWER
    return prompt, row["output"]


def main():
    if not os.path.exists(TOKENIZER):
        raise SystemExit(f"Tokenizer not found: {TOKENIZER} (run build_corpus.py first)")
    random.seed(SEED)
    sp = spm.SentencePieceProcessor(); sp.load(TOKENIZER)
    eos = sp.eos_id()

    print("Loading instruction sources from HuggingFace...")
    aya = load_aya_swahili()
    bactrian = load_bactrian_swahili()
    blended = load_blended_swahili()

    mixed = bactrian + blended + aya * OVERSAMPLE
    random.shuffle(mixed)
    print(f"Mix: {len(bactrian)} Bactrian + {len(blended)} blended + "
          f"{len(aya)}x{OVERSAMPLE} Aya = {len(mixed)} examples")

    examples, jsonl_rows, skipped = [], [], 0
    for row in mixed:
        prompt, response = render(row)
        p_ids = sp.encode(prompt)
        r_ids = sp.encode(response) + [eos]
        if len(p_ids) >= MAX_LEN:
            skipped += 1
            continue
        ids = (p_ids + r_ids)[:MAX_LEN]
        if len(ids) <= len(p_ids):
            skipped += 1
            continue
        examples.append({"ids": ids, "prompt_len": len(p_ids)})
        jsonl_rows.append({
            "instruction": row["instruction"], "input": row["input"],
            "output": row["output"], "text": prompt + response,
        })

    random.shuffle(examples)
    n_val = max(1, int(len(examples) * VAL_FRAC))
    val, train = examples[:n_val], examples[n_val:]

    torch.save(train, os.path.join(OUT_DIR, "train.pt"))
    torch.save(val, os.path.join(OUT_DIR, "val.pt"))

    seen, clean = set(), []
    for r in jsonl_rows:
        if r["text"] in seen:
            continue
        seen.add(r["text"]); clean.append(r)
    with open(os.path.join(OUT_DIR, "instruct.jsonl"), "w", encoding="utf-8") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "template": T_INSTR + T_INPUT + T_ANSWER + "{output}</s>",
        "oversample_aya": OVERSAMPLE, "max_len": MAX_LEN,
        "counts": {
            "aya_native": len(aya), "bactrian": len(bactrian),
            "blended": len(blended), "tokenised_examples": len(examples),
            "unique_jsonl_rows": len(clean), "skipped": skipped,
            "train": len(train), "val": len(val),
        },
        "sources": {
            "aya": "CohereLabs/aya_dataset (Swahili subset, native)",
            "bactrian": "MBZUAI/Bactrian-X (sw split, translated)",
            "blended": "NabajyotiPathak/kiswahili-ai-blended (enhanced fields)",
        },
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nWrote -> {OUT_DIR}")
    print(f"  train.pt : {len(train):,} | val.pt : {len(val):,} | "
          f"instruct.jsonl : {len(clean):,} | skipped : {skipped:,}")


if __name__ == "__main__":
    main()
