"""
Kiswahili LLM Training
Uses the dataset and tokenizer
Author: Shadrackovsky
"""

import os
import json
import random
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
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


def create_causal_mask(seq_len: int) -> mx.array:
    mask = mx.triu(mx.ones((seq_len, seq_len)), k=1)
    mask = mask * -1e9
    return mask


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiHeadAttention(dims=HIDDEN_SIZE, num_heads=NUM_HEADS)
        self.norm1 = nn.LayerNorm(dims=HIDDEN_SIZE, eps=LN_EPS)
        self.norm2 = nn.LayerNorm(dims=HIDDEN_SIZE, eps=LN_EPS)
        self.ffn = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE),
            nn.GELU(),
            nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE),
            nn.Dropout(DROPOUT)
        )
        self.dropout = nn.Dropout(DROPOUT)

    def __call__(self, x, mask=None):
        norm_x = self.norm1(x)
        # MLX MultiHeadAttention call: query, key, value
        attn_out = self.attn(norm_x, norm_x, norm_x, mask=mask)
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        return x


class KiswahiliLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.pos_embedding = nn.Embedding(MAX_SEQ_LEN, HIDDEN_SIZE)
        self.layers = [TransformerBlock() for _ in range(NUM_LAYERS)]
        self.norm = nn.LayerNorm(dims=HIDDEN_SIZE, eps=LN_EPS)
        self.output = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)

    def __call__(self, input_ids):
        seq_len = input_ids.shape[1]
        positions = mx.arange(seq_len)[None, :]
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        mask = create_causal_mask(seq_len)
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
        x = seq[:-1]
        y = seq[1:]
        x_batch.append(x)
        y_batch.append(y)
    return mx.array(x_batch, dtype=mx.int32), mx.array(y_batch, dtype=mx.int32)


if __name__ == "__main__":
    print("Starting training process")

    texts = load_dataset()
    tokenizer = load_tokenizer()
    model = KiswahiliLLM()
    mx.eval(model.parameters())

    optimizer = optim.AdamW(learning_rate=3e-4, weight_decay=0.01)
    loss_fn = nn.losses.cross_entropy

    batch_size = 8
    epochs = 8
    steps_per_epoch = 2000

    print(f"Settings: batch={batch_size}, sequence length={BLOCK_SIZE}, epochs={epochs}")

    for epoch in range(epochs):
        total_loss = 0.0
        progress = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")

        for step in progress:
            inputs, targets = get_batch(texts, tokenizer, batch_size, BLOCK_SIZE)

            def loss_and_grad(model, x, y):
                logits = model(x)
                return mx.mean(loss_fn(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1)))

            loss, grads = mx.value_and_grad(loss_and_grad)(model, inputs, targets)
            model.update(optimizer.apply_gradients(grads, model))
            mx.eval(model.parameters(), optimizer.state)

            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / steps_per_epoch
        print(f"Epoch {epoch+1} complete, average loss: {avg_loss:.4f}")

        save_path = f"model_epoch_{epoch+1:02d}.npz"
        model.save_weights(save_path)
        print(f"Model saved to {save_path}")

    model.save_weights("swahili_llm_final.npz")
    print("Training complete. Final model saved as swahili_llm_final.npz")