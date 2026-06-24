"""
Kiswahili LLM Training (PyTorch)
Uses the dataset and tokenizer
Author: Shadrackovsky
"""

import os
import json
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sentencepiece as spm
import jsonlines


# Load configuration
with open("model_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

VOCAB_SIZE = config["vocab_size"]
NUM_LAYERS = config["num_layers"]
NUM_HEADS = config["num_attention_heads"]
HIDDEN_SIZE = config["hidden_size"]
INTERMEDIATE_SIZE = config["intermediate_size"]
MAX_SEQ_LEN = config["max_seq_len"]
DROPOUT = config["dropout"]
LN_EPS = config["layer_norm_epsilon"]
BLOCK_SIZE = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def create_causal_mask(seq_len: int, device) -> torch.Tensor:
    # Float mask: 0 where allowed, -inf where masked (upper triangle, k=1)
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=HIDDEN_SIZE,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(HIDDEN_SIZE, eps=LN_EPS)
        self.norm2 = nn.LayerNorm(HIDDEN_SIZE, eps=LN_EPS)
        self.ffn = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE),
            nn.GELU(),
            nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE),
            nn.Dropout(DROPOUT),
        )
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, mask=None):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(
            norm_x, norm_x, norm_x, attn_mask=mask, need_weights=False
        )
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        return x


class KiswahiliLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.pos_embedding = nn.Embedding(MAX_SEQ_LEN, HIDDEN_SIZE)
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(NUM_LAYERS)])
        self.norm = nn.LayerNorm(HIDDEN_SIZE, eps=LN_EPS)
        self.output = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)

    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device)[None, :]
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        mask = create_causal_mask(seq_len, input_ids.device)
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits


def load_dataset():
    path = "full_dataset.jsonl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    texts = []
    with jsonlines.open(path, "r") as reader:
        for entry in reader:
            text = entry.get("text", "").strip()
            if len(text) > 10:
                texts.append(text)
    print(f"Loaded {len(texts)} samples from dataset")
    return texts


def load_tokenizer():
    path = "swahili_tokenizer.model"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tokenizer file not found: {path}")
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    print(f"Tokenizer loaded, vocab size: {sp.vocab_size()}")
    return sp


def get_batch(texts, tokenizer, batch_size=8, seq_len=128):
    x_batch = []
    y_batch = []
    for _ in range(batch_size):
        text = random.choice(texts)
        tokens = tokenizer.encode(text)
        if len(tokens) < seq_len + 1:
            tokens = tokens * ((seq_len + 1) // len(tokens) + 1)
        start = random.randint(0, len(tokens) - seq_len - 1)
        seq = tokens[start : start + seq_len + 1]
        x_batch.append(seq[:-1])
        y_batch.append(seq[1:])
    x = torch.tensor(x_batch, dtype=torch.long, device=DEVICE)
    y = torch.tensor(y_batch, dtype=torch.long, device=DEVICE)
    return x, y


if __name__ == "__main__":
    print("Starting training process")
    print(f"Using device: {DEVICE}")

    texts = load_dataset()
    tokenizer = load_tokenizer()
    model = KiswahiliLLM().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    batch_size = 8
    epochs = 8
    steps_per_epoch = 2000

    print(f"Settings: batch={batch_size}, sequence length={BLOCK_SIZE}, epochs={epochs}")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        progress = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")

        for step in progress:
            inputs, targets = get_batch(texts, tokenizer, batch_size, BLOCK_SIZE)

            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / steps_per_epoch
        print(f"Epoch {epoch+1} complete, average loss: {avg_loss:.4f}")

        save_path = f"model_epoch_{epoch+1:02d}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    torch.save(model.state_dict(), "swahili_llm_final.pt")
    print("Training complete. Final model saved as swahili_llm_final.pt")
