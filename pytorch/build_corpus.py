"""
Build training data from a HuggingFace dataset.

Streams a HF text dataset, trains a SentencePiece BPE tokenizer on a sample, and
tokenizes everything into a flat uint16 token-id binary (train.bin / val.bin) that
train.py reads via np.memmap. Resumable: progress is tracked in manifest.json.

Everything is configurable by environment variable (no hardcoded local paths):

    HF_DATASET   HuggingFace dataset id          (default: Alfaxad/Inkuba-Mono-Swahili)
    HF_SPLIT     split to stream                 (default: train)
    TEXT_COLUMN  column holding the text         (default: text)
    DATA_DIR     where to write the binaries     (default: ./data next to this file)
    TOK_SAMPLE   sentences sampled for tokenizer (default: 4000000)

Usage:  python build_corpus.py
"""
import os
import io
import json
import time
import tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

HF_DATASET = os.environ.get("HF_DATASET", "Alfaxad/Inkuba-Mono-Swahili")
HF_SPLIT = os.environ.get("HF_SPLIT", "train")
TEXT_COLUMN = os.environ.get("TEXT_COLUMN", "text")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(HERE, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

with open(os.path.join(HERE, "model_config.json"), encoding="utf-8") as f:
    VOCAB_SIZE = json.load(f)["vocab_size"]

TOK_SAMPLE = int(os.environ.get("TOK_SAMPLE", 4_000_000))
CHAR_COVERAGE = 0.9995          # rest -> byte tokens (byte_fallback)
MIN_CHARS = 8
FLUSH_TOKENS = 4_000_000
VAL_TOKENS = 20_000

TOKENIZER_PREFIX = os.path.join(DATA_DIR, "tokenizer")
TOKENIZER_MODEL = TOKENIZER_PREFIX + ".model"
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN = os.path.join(DATA_DIR, "val.bin")
MANIFEST = os.path.join(DATA_DIR, "manifest.json")
META = os.path.join(DATA_DIR, "meta.json")

assert VOCAB_SIZE <= 65535, "uint16 storage requires vocab <= 65535"


def iter_hf(skip=0):
    """Stream the dataset's text column, self-healing across dropped connections."""
    from datasets import load_dataset
    seen = skip
    while True:
        try:
            ds = load_dataset(HF_DATASET, split=HF_SPLIT, streaming=True)
            if seen:
                ds = ds.skip(seen)
            for ex in ds:
                seen += 1
                text = (ex.get(TEXT_COLUMN) or "").strip()
                if len(text) >= MIN_CHARS:
                    yield text
            return
        except Exception as e:
            print(f"[hf] stream dropped after {seen:,} rows: {e!r}; reconnecting in 20s",
                  flush=True)
            time.sleep(20)


def train_tokenizer():
    import sentencepiece as spm
    if os.path.exists(TOKENIZER_MODEL):
        print(f"[tok] {TOKENIZER_MODEL} exists, skipping")
        return
    print(f"[tok] sampling {TOK_SAMPLE:,} sentences from {HF_DATASET} for tokenizer")
    sample_path = os.path.join(DATA_DIR, "_tok_sample.txt")
    n = 0
    with io.open(sample_path, "w", encoding="utf-8") as out:
        for text in iter_hf():
            out.write(text.replace("\n", " ") + "\n")
            n += 1
            if n >= TOK_SAMPLE:
                break
    print(f"[tok] training BPE vocab={VOCAB_SIZE} (byte_fallback) on {n:,} sentences")
    spm.SentencePieceTrainer.train(
        input=sample_path, model_prefix=TOKENIZER_PREFIX, vocab_size=VOCAB_SIZE,
        character_coverage=CHAR_COVERAGE, model_type="bpe", byte_fallback=True,
        train_extremely_large_corpus=True, num_threads=os.cpu_count() or 4,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",
    )
    os.remove(sample_path)
    print(f"[tok] saved {TOKENIZER_MODEL}")


def load_manifest():
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as f:
            return json.load(f)
    return {"bytes_written": 0, "rows_done": 0, "total_tokens": 0, "val_done": False}


def save_manifest(m):
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(m, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, MANIFEST)


def build_val(sp, manifest):
    if manifest["val_done"]:
        return
    ids = []
    for text in iter_hf():
        ids.extend(sp.encode(text)); ids.append(sp.eos_id())
        if len(ids) >= VAL_TOKENS:
            break
    np.array(ids, dtype=np.uint16).tofile(VAL_BIN)
    print(f"[val] wrote {len(ids):,} tokens -> {VAL_BIN}")
    manifest["val_done"] = True
    save_manifest(manifest)


def build_train(sp, manifest):
    # Drop any partial tail from a previous crash, then append.
    open(TRAIN_BIN, "ab").close()
    if os.path.getsize(TRAIN_BIN) > manifest["bytes_written"]:
        with open(TRAIN_BIN, "r+b") as f:
            f.truncate(manifest["bytes_written"])
    f = open(TRAIN_BIN, "ab"); f.seek(0, os.SEEK_END)
    buf = []
    rows = manifest["rows_done"]
    t0 = time.time()

    def flush():
        nonlocal buf
        if not buf:
            return
        np.array(buf, dtype=np.uint16).tofile(f); f.flush(); os.fsync(f.fileno())
        manifest["bytes_written"] = f.tell()
        manifest["total_tokens"] += len(buf)
        manifest["rows_done"] = rows
        save_manifest(manifest)
        buf = []

    print(f"[train] streaming {HF_DATASET} (resume @ {rows:,} rows)")
    for text in iter_hf(skip=rows):
        rows += 1
        buf.extend(sp.encode(text)); buf.append(sp.eos_id())
        if len(buf) >= FLUSH_TOKENS:
            flush()
            tot = manifest["total_tokens"]
            print(f"[train] {tot:,} tokens, {rows:,} rows "
                  f"({(tot)/max(time.time()-t0,1e-9)/1e3:.0f}k tok/s)", flush=True)
    flush(); f.close()
    print(f"[train] DONE — {manifest['total_tokens']:,} tokens")


def write_meta():
    meta = {
        "vocab_size": VOCAB_SIZE,
        "train_tokens": os.path.getsize(TRAIN_BIN) // 2 if os.path.exists(TRAIN_BIN) else 0,
        "val_tokens": os.path.getsize(VAL_BIN) // 2 if os.path.exists(VAL_BIN) else 0,
        "dtype": "uint16", "hf_dataset": HF_DATASET,
    }
    with open(META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[meta] {meta}")


def main():
    import sentencepiece as spm
    print(f"DATA_DIR={DATA_DIR}  HF_DATASET={HF_DATASET}")
    train_tokenizer()
    sp = spm.SentencePieceProcessor(); sp.load(TOKENIZER_MODEL)
    manifest = load_manifest()
    build_val(sp, manifest)
    build_train(sp, manifest)
    write_meta()
    print("BUILD COMPLETE")


if __name__ == "__main__":
    main()
